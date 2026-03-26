"""Tests for language identification filter."""

from unittest.mock import MagicMock, patch

import pytest

from dq.filters import ensure_registered; ensure_registered()
from dq.pipeline import get_filter_class


class TestLanguageFilter:
    """Tests for LanguageFilter using mocked fasttext model."""

    def _make_mock_model(self, lang="en", score=0.95):
        """Create a mock _LIDModel that returns predictable results."""
        mock = MagicMock()
        mock.predict.return_value = (
            (lang, score),
            {lang: score, "fr": 0.02, "de": 0.01},
        )
        return mock

    def test_registered(self):
        cls = get_filter_class("language")
        assert cls is not None
        assert cls.name == "language"

    def test_high_confidence_passes(self):
        from dq.filters.language import LanguageFilter
        f = LanguageFilter(languages=["en"], language_threshold=0.65)
        f._model = self._make_mock_model("en", 0.95)

        doc = {"text": "Hello world, this is a test document."}
        keep, info = f.filter(doc)
        assert keep is True
        assert doc["metadata"]["language"] == "en"
        assert doc["metadata"]["language_score"] == 0.95

    def test_low_confidence_fails(self):
        from dq.filters.language import LanguageFilter
        f = LanguageFilter(languages=["en"], language_threshold=0.65)
        f._model = self._make_mock_model("en", 0.30)

        doc = {"text": "asdfghjkl qwerty zxcvbn"}
        keep, info = f.filter(doc)
        assert keep is False
        assert info["reason"] == "language_below_threshold"
        assert info["score"] == 0.30

    def test_wrong_language_fails(self):
        from dq.filters.language import LanguageFilter
        f = LanguageFilter(languages=["en"], language_threshold=0.65)
        # Model predicts French, not English
        mock = MagicMock()
        mock.predict.return_value = (
            ("fr", 0.90),
            {"fr": 0.90, "en": 0.05, "de": 0.03},
        )
        f._model = mock

        doc = {"text": "Bonjour le monde"}
        keep, info = f.filter(doc)
        assert keep is False

    def test_no_language_filter_keeps_high_confidence(self):
        """languages=None means keep any language above threshold."""
        from dq.filters.language import LanguageFilter
        f = LanguageFilter(languages=None, language_threshold=0.65)
        f._model = self._make_mock_model("ja", 0.90)

        doc = {"text": "テスト文書"}
        keep, info = f.filter(doc)
        assert keep is True

    def test_label_only_always_passes(self):
        from dq.filters.language import LanguageFilter
        f = LanguageFilter(label_only=True)
        f._model = self._make_mock_model("en", 0.10)  # Very low score

        doc = {"text": "some text"}
        keep, info = f.filter(doc)
        assert keep is True
        assert doc["metadata"]["language"] == "en"

    def test_empty_text_fails(self):
        from dq.filters.language import LanguageFilter
        f = LanguageFilter()
        f._model = self._make_mock_model()

        doc = {"text": ""}
        keep, info = f.filter(doc)
        assert keep is False
        assert info["reason"] == "empty_text"

    def test_multiple_languages_any_above_threshold(self):
        """With languages=['en', 'fr'], keep if either is above threshold."""
        from dq.filters.language import LanguageFilter
        f = LanguageFilter(languages=["en", "fr"], language_threshold=0.5)
        mock = MagicMock()
        mock.predict.return_value = (
            ("fr", 0.80),
            {"fr": 0.80, "en": 0.10, "de": 0.05},
        )
        f._model = mock

        doc = {"text": "Bonjour le monde"}
        keep, info = f.filter(doc)
        assert keep is True

    def test_filter_detailed(self):
        from dq.filters.language import LanguageFilter
        f = LanguageFilter(languages=["en"], language_threshold=0.65)
        f._model = self._make_mock_model("en", 0.30)

        doc = {"text": "asdf qwerty"}
        passed, failures = f.filter_detailed(doc)
        assert passed is False
        assert len(failures) == 1
        assert failures[0]["rule"] == "language_threshold"
        assert failures[0]["value"] == 0.30

    def test_glotlid_parses_script(self):
        from dq.filters.language import LanguageFilter
        f = LanguageFilter(backend="glotlid", languages=None, language_threshold=0.5)
        mock = MagicMock()
        mock.predict.return_value = (
            ("zho_Hans", 0.95),
            {"zho_Hans": 0.95, "zho_Hant": 0.03},
        )
        f._model = mock

        doc = {"text": "你好世界"}
        keep, info = f.filter(doc)
        assert keep is True
        assert doc["metadata"]["language"] == "zho"
        assert doc["metadata"]["language_script"] == "Hans"
