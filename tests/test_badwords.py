"""Tests for bad words filter."""

from unittest.mock import patch
import re

import pytest

from dq.filters import ensure_registered; ensure_registered()
from dq.pipeline import get_filter_class


class TestBadWordsFilter:
    """Tests for BadWordsFilter using inline word lists (no network)."""

    def _make_filter_with_words(self, words, lang="en"):
        """Create a BadWordsFilter with pre-loaded regex cache."""
        from dq.filters.badwords import BadWordsFilter, _CJK_LANGS
        f = BadWordsFilter(default_language=lang)

        escaped = [re.escape(w) for w in words]
        if lang in _CJK_LANGS:
            pattern = "|".join(escaped)
        else:
            pattern = r"(?:\W|^)(" + "|".join(escaped) + r")(?:\W|$)"

        f._regex_cache[lang] = re.compile(pattern)
        return f

    def test_registered(self):
        cls = get_filter_class("badwords")
        assert cls is not None
        assert cls.name == "badwords"

    def test_clean_text_passes(self):
        f = self._make_filter_with_words(["badword", "nastything"])
        doc = {"text": "This is a perfectly clean document about science."}
        keep, info = f.filter(doc)
        assert keep is True

    def test_text_with_badword_fails(self):
        f = self._make_filter_with_words(["badword", "nastything"])
        doc = {"text": "This document contains badword in it."}
        keep, info = f.filter(doc)
        assert keep is False
        assert info["reason"] == "badwords"

    def test_case_insensitive(self):
        """Bad words matching should be case-insensitive (via text.lower())."""
        f = self._make_filter_with_words(["badword"])
        doc = {"text": "This has BADWORD in uppercase."}
        keep, info = f.filter(doc)
        assert keep is False

    def test_word_boundary_western(self):
        """Western languages should use word boundaries."""
        f = self._make_filter_with_words(["bad"])
        # "bad" as standalone word
        doc1 = {"text": "This is bad content."}
        keep1, _ = f.filter(doc1)
        assert keep1 is False

        # "bad" as substring of another word — should NOT match
        # because the regex uses \W boundaries
        doc2 = {"text": "This badge is nice."}
        keep2, _ = f.filter(doc2)
        assert keep2 is True

    def test_cjk_substring_match(self):
        """CJK languages should match substrings (no word boundaries)."""
        f = self._make_filter_with_words(["脏话"], lang="zh")
        doc = {"text": "这里有脏话内容"}
        keep, info = f.filter(doc)
        assert keep is False

    def test_empty_text_passes(self):
        f = self._make_filter_with_words(["badword"])
        doc = {"text": ""}
        keep, info = f.filter(doc)
        assert keep is True

    def test_uses_doc_language_metadata(self):
        """Should use language from doc metadata if available."""
        from dq.filters.badwords import BadWordsFilter
        f = BadWordsFilter(default_language="en")
        # Pre-load both English and French
        f._regex_cache["en"] = re.compile(r"(?:\W|^)(badword)(?:\W|$)")
        f._regex_cache["fr"] = re.compile(r"(?:\W|^)(grosmet)(?:\W|$)")

        # Doc with French language metadata
        doc = {"text": "This has badword in English.", "metadata": {"language": "fr"}}
        keep, info = f.filter(doc)
        # Should use French regex, which doesn't match "badword"
        assert keep is True

    def test_unknown_language_passes(self):
        """Unknown language (no wordlist) should pass through."""
        from dq.filters.badwords import BadWordsFilter
        f = BadWordsFilter(default_language="xx_unknown")
        doc = {"text": "Any content here."}
        keep, info = f.filter(doc)
        assert keep is True

    def test_filter_detailed(self):
        f = self._make_filter_with_words(["badword"])
        doc = {"text": "Contains badword here."}
        passed, failures = f.filter_detailed(doc)
        assert passed is False
        assert len(failures) == 1
        assert failures[0]["rule"] == "badwords"

    def test_filter_detailed_passes(self):
        f = self._make_filter_with_words(["badword"])
        doc = {"text": "Clean text here."}
        passed, failures = f.filter_detailed(doc)
        assert passed is True
        assert failures == []


class TestBadWordsConstants:
    """Test constants and helper functions."""

    def test_supported_languages(self):
        from dq.filters.badwords import BADWORDS_LANGS
        assert "en" in BADWORDS_LANGS
        assert "zh" in BADWORDS_LANGS
        assert "ja" in BADWORDS_LANGS
        assert len(BADWORDS_LANGS) == 28

    def test_allowlist_entries(self):
        from dq.filters.badwords import _ALLOWLISTS
        assert "ja" in _ALLOWLISTS
        assert "zh" in _ALLOWLISTS
        assert "性" in _ALLOWLISTS["zh"]
