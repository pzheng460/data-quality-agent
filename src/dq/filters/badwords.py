"""C4 bad words filter.

Aligned with datatrove's C4BadWordsFilter. Detects documents containing
profanity/NSFW content using language-specific word lists from LDNOOBW.

Word lists are downloaded on first use and cached locally.
"""

import logging
import re
from pathlib import Path
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

logger = logging.getLogger(__name__)

# ── LDNOOBW GitHub URLs (matching datatrove exactly) ─────────────────
_EN_BADWORDS_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/25e679f03d96baa721cde20db9944649e8d0a844/en"
_OTHER_BADWORDS_URL = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/5faf2ba42d7b1c0977169ec3611df25a3c08eb13/"

# Languages supported by LDNOOBW (matching datatrove)
BADWORDS_LANGS = frozenset({
    "ar", "cs", "da", "de", "en", "eo", "es", "fa", "fi", "fil",
    "fr", "fr-CA-u-sd-caqc", "hi", "hu", "it", "ja", "kab", "ko",
    "nl", "no", "pl", "pt", "ru", "sv", "th", "tlh", "tr", "zh",
})

# CJK languages use substring matching (no word boundaries)
_CJK_LANGS = {"ja", "th", "zh"}

# Per-language allowlists (matching datatrove) — false positives to exclude
_ALLOWLISTS: dict[str, set[str]] = {
    "ja": {"sm", "グロ", "女の子"},
    "zh": {"性"},
}

# Cache directory
_CACHE_DIR = Path.home() / ".cache" / "dq" / "badwords"


def _download_wordlist(lang: str) -> list[str]:
    """Download bad words list for a language, return as list of words."""
    url = _EN_BADWORDS_URL if lang == "en" else _OTHER_BADWORDS_URL + lang

    cache_path = _CACHE_DIR / f"{lang}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip().splitlines()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading badwords list for '%s'...", lang)

    import urllib.request
    try:
        data = urllib.request.urlopen(url).read().decode("utf-8")
        cache_path.write_text(data, encoding="utf-8")
        return data.strip().splitlines()
    except Exception as e:
        logger.warning("Failed to download badwords for '%s': %s", lang, e)
        return []


def _compile_badwords_regex(lang: str) -> re.Pattern | None:
    """Compile a regex for detecting bad words in the given language.

    Matching datatrove's C4BadWordsFilter:
    - CJK (ja/th/zh): simple alternation (substring match)
    - Others: word-boundary pattern (?:\\W|^)(word1|word2|...)(?:\\W|$)
    """
    words = _download_wordlist(lang)
    if not words:
        return None

    # Apply allowlist
    allowlist = _ALLOWLISTS.get(lang, set())
    words = [w for w in words if w not in allowlist]

    if not words:
        return None

    # Escape regex special characters
    escaped = [re.escape(w) for w in words]

    if lang in _CJK_LANGS:
        # CJK: no word boundaries, substring match
        pattern = "|".join(escaped)
    else:
        # Western: word boundary matching
        pattern = r"(?:\W|^)(" + "|".join(escaped) + r")(?:\W|$)"

    return re.compile(pattern)


@register_filter("badwords")
class BadWordsFilter(BaseFilter):
    """C4 bad words filter (aligned with datatrove).

    Detects and removes documents containing profanity/NSFW words.
    Supports 28 languages via LDNOOBW word lists.

    Args:
        default_language: Language to assume if not detected (default "en").
        language_field: Metadata field containing detected language code.
            If doc has metadata.language (e.g. from LanguageFilter), it's used
            automatically. Falls back to default_language.
    """

    def __init__(
        self,
        text_field: str = "text",
        default_language: str = "en",
        language_field: str = "language",
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.default_language = default_language
        self.language_field = language_field
        # Lazy cache: lang -> compiled regex (or None)
        self._regex_cache: dict[str, re.Pattern | None] = {}

    def _get_regex(self, lang: str) -> re.Pattern | None:
        """Get compiled regex for language, with caching."""
        if lang not in self._regex_cache:
            if lang not in BADWORDS_LANGS:
                self._regex_cache[lang] = None
            else:
                self._regex_cache[lang] = _compile_badwords_regex(lang)
        return self._regex_cache[lang]

    def _detect_language(self, doc: dict) -> str:
        """Extract language from doc metadata, falling back to default."""
        metadata = doc.get("metadata", {})
        lang = metadata.get(self.language_field)
        if lang and lang in BADWORDS_LANGS:
            return lang
        return self.default_language

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        if not text.strip():
            return True, {}  # Empty docs pass (other filters handle them)

        lang = self._detect_language(doc)
        regex = self._get_regex(lang)

        if regex is None:
            # No badwords list for this language — pass through
            return True, {}

        # Matching datatrove: search lowercase text
        if regex.search(text.lower()):
            return False, {
                "filter": self.name,
                "reason": "badwords",
                "language": lang,
            }

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        keep, info = self.filter(doc)
        if keep:
            return True, []
        return False, [{
            "filter": self.name,
            "rule": "badwords",
            "value": info.get("language", "unknown"),
            "threshold": "none",
        }]
