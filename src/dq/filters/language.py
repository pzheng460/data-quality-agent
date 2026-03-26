"""Language identification filter.

Aligned with datatrove's LanguageFilter. Uses fasttext models (ft176 or glotlid)
to detect document language and filter by confidence threshold.

Requires: fasttext-numpy2-wheel (installed on first use).
"""

import logging
import re
from pathlib import Path
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

logger = logging.getLogger(__name__)

# ── Model URLs (matching datatrove) ──────────────────────────────────
_FT176_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
_GLOTLID_URLS = {
    "v1": "https://huggingface.co/cis-lmu/glotlid/resolve/main/model.bin",
    "v2": "https://huggingface.co/cis-lmu/glotlid/resolve/main/model_v2.bin",
    "v3": "https://huggingface.co/cis-lmu/glotlid/resolve/main/model_v3.bin",
}

# Cache directory
_CACHE_DIR = Path.home() / ".cache" / "dq" / "lid"


def _download_model(url: str, dest: Path) -> Path:
    """Download a model file if not cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    logger.info("Downloading language ID model to %s ...", dest)
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        raise
    logger.info("Download complete: %s", dest)
    return dest


def _load_fasttext_model(path: Path):
    """Load a fasttext model, suppressing its stderr warning."""
    try:
        import fasttext
    except ImportError:
        raise ImportError(
            "fasttext is required for language detection. "
            "Install with: uv add fasttext-numpy2-wheel"
        )
    # fasttext prints a warning about deprecated load_model; suppress it
    import io
    import contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        return fasttext.load_model(str(path))


class _LIDModel:
    """Lazy-loading language identification model wrapper."""

    def __init__(self, backend: str = "ft176", glotlid_version: str = "v3"):
        self.backend = backend
        self.glotlid_version = glotlid_version
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if self.backend == "ft176":
            url = _FT176_URL
            cache_path = _CACHE_DIR / "ft176" / "lid.176.bin"
        else:
            url = _GLOTLID_URLS[self.glotlid_version]
            cache_path = _CACHE_DIR / "glotlid" / f"model_{self.glotlid_version}.bin"

        model_path = _download_model(url, cache_path)
        self._model = _load_fasttext_model(model_path)

    def predict(self, text: str) -> tuple[tuple[str, float], dict[str, float]]:
        """Predict language(s) for text.

        Returns:
            (best_pair, lang_scores) where:
            - best_pair = (lang_code, score)
            - lang_scores = {lang_code: score, ...}
        """
        self._ensure_model()
        # fasttext expects single-line input
        clean_text = text.replace("\n", " ")
        labels, scores = self._model.predict(clean_text, k=-1)

        lang_pairs: dict[str, float] = {}
        for label, score in zip(labels, scores):
            # Labels are like "__label__en" or "__label__zho_Hans"
            code = label.split("__")[2]
            lang_pairs[code] = float(score)

        if lang_pairs:
            best_code = max(lang_pairs, key=lang_pairs.get)
            return (best_code, lang_pairs[best_code]), lang_pairs
        return ("unknown", 0.0), {}


@register_filter("language")
class LanguageFilter(BaseFilter):
    """Language identification filter (aligned with datatrove).

    Uses fasttext to detect document language and filters by confidence.

    Args:
        languages: List of language codes to keep (e.g. ["en", "fr"]).
                   None means keep any language above threshold.
        language_threshold: Minimum confidence score to keep (default 0.65).
        backend: "ft176" (176 languages, ISO 639-1) or "glotlid" (1000+ lang+script).
        label_only: If True, only add language metadata without filtering.
    """

    def __init__(
        self,
        text_field: str = "text",
        languages: list[str] | str | None = None,
        language_threshold: float = 0.65,
        backend: str = "ft176",
        label_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        if isinstance(languages, str):
            languages = [languages]
        self.languages = languages
        self.language_threshold = language_threshold
        self.backend = backend
        self.label_only = label_only
        self._model = _LIDModel(backend=backend)

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        if not text.strip():
            return False, {"filter": self.name, "reason": "empty_text"}

        best_pair, lang_pairs = self._model.predict(text)
        lang, score = best_pair

        # Parse GlotLID lang+script codes
        lang_code = lang
        lang_script = None
        if self.backend == "glotlid" and "_" in lang:
            parts = lang.split("_", 1)
            lang_code = parts[0]
            lang_script = parts[1]

        # Enrich doc metadata
        doc.setdefault("metadata", {})
        doc["metadata"]["language"] = lang_code
        doc["metadata"]["language_score"] = score
        if lang_script:
            doc["metadata"]["language_script"] = lang_script

        if self.label_only:
            return True, {}

        # Filter logic (matching datatrove exactly)
        if self.languages:
            # Keep if ANY specified language is above threshold
            keep = any(
                lang_pairs.get(l, 0.0) > self.language_threshold
                for l in self.languages
            )
        else:
            # Keep if best language is above threshold
            keep = score > self.language_threshold

        if keep:
            return True, {}
        return False, {
            "filter": self.name,
            "reason": "language_below_threshold",
            "language": lang_code,
            "score": round(score, 4),
            "threshold": self.language_threshold,
        }

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        keep, info = self.filter(doc)
        if keep:
            return True, []
        return False, [{
            "filter": self.name,
            "rule": "language_threshold",
            "value": info.get("score", 0.0),
            "threshold": self.language_threshold,
        }]
