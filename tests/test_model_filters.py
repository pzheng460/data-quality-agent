"""Tests for model-based quality filters (Phase 2).

Uses mocks to avoid requiring actual models or GPU in CI.
Tests graceful degradation when dependencies are missing.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# FastTextQualityFilter
# ---------------------------------------------------------------------------

class TestFastTextQualityFilter:
    """Tests for FastTextQualityFilter."""

    def test_skip_when_no_fasttext(self):
        """Filter should pass all docs when fasttext is not installed."""
        with patch("dq.model_filters.fasttext_quality._check_fasttext", return_value=False):
            with patch("dq.model_filters.fasttext_quality._fasttext_available", False):
                from dq.model_filters.fasttext_quality import FastTextQualityFilter
                f = FastTextQualityFilter.__new__(FastTextQualityFilter)
                f.text_field = "text"
                f.params = {}
                f.model_path = ""
                f.threshold = 0.5
                f.label = "__label__hq"
                f._model = None
                f._available = False

                keep, info = f.filter({"text": "some document text here"})
                assert keep is True
                assert info["reason"] == "no model"

    def test_skip_when_no_model_path(self):
        """Filter should pass all docs when no model path is configured."""
        with patch("dq.model_filters.fasttext_quality._check_fasttext", return_value=True):
            from dq.model_filters.fasttext_quality import FastTextQualityFilter
            f = FastTextQualityFilter.__new__(FastTextQualityFilter)
            f.text_field = "text"
            f.params = {}
            f.model_path = ""
            f.threshold = 0.5
            f.label = "__label__hq"
            f._model = None
            f._available = True

            keep, info = f.filter({"text": "some document text here"})
            assert keep is True
            assert info["reason"] == "no model"

    def test_filter_with_mock_model(self):
        """Filter should use the model to score and filter docs."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (["__label__hq"], [0.85])

        with patch("dq.model_filters.fasttext_quality._check_fasttext", return_value=True):
            from dq.model_filters.fasttext_quality import FastTextQualityFilter
            f = FastTextQualityFilter.__new__(FastTextQualityFilter)
            f.text_field = "text"
            f.params = {}
            f.model_path = "/fake/model.bin"
            f.threshold = 0.5
            f.label = "__label__hq"
            f._model = mock_model
            f._available = True

            keep, info = f.filter({"text": "High quality text about science."})
            assert keep is True
            assert info["score"] == 0.85
            assert info["label"] == "__label__hq"

    def test_filter_drops_low_score(self):
        """Filter should drop docs below threshold."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (["__label__lq"], [0.9])

        with patch("dq.model_filters.fasttext_quality._check_fasttext", return_value=True):
            from dq.model_filters.fasttext_quality import FastTextQualityFilter
            f = FastTextQualityFilter.__new__(FastTextQualityFilter)
            f.text_field = "text"
            f.params = {}
            f.model_path = "/fake/model.bin"
            f.threshold = 0.5
            f.label = "__label__hq"
            f._model = mock_model
            f._available = True

            keep, info = f.filter({"text": "bad low quality text"})
            assert keep is False
            assert info["score"] == 0.1  # 1.0 - 0.9

    def test_filter_registered(self):
        """FastTextQualityFilter should be registered in the pipeline registry."""
        import dq.model_filters  # noqa: F401
        from dq.pipeline import get_filter_class
        cls = get_filter_class("fasttext_quality")
        assert cls.__name__ == "FastTextQualityFilter"


# ---------------------------------------------------------------------------
# PerplexityFilter
# ---------------------------------------------------------------------------

class TestPerplexityFilter:
    """Tests for PerplexityFilter."""

    def test_skip_when_no_transformers(self):
        """Filter should pass all docs when transformers is not installed."""
        with patch("dq.model_filters.perplexity._check_transformers", return_value=False):
            from dq.model_filters.perplexity import PerplexityFilter
            f = PerplexityFilter.__new__(PerplexityFilter)
            f.text_field = "text"
            f.params = {}
            f.model_name = "test-model"
            f.max_perplexity = 1000.0
            f.min_perplexity = 5.0
            f.batch_size = 8
            f.device = "cpu"
            f._available = False
            f._loaded = False

            keep, info = f.filter({"text": "some text"})
            assert keep is True
            assert info["reason"] == "skipped"

    def test_filter_rejects_empty(self):
        """Filter should reject empty text."""
        with patch("dq.model_filters.perplexity._check_transformers", return_value=True):
            from dq.model_filters.perplexity import PerplexityFilter
            f = PerplexityFilter.__new__(PerplexityFilter)
            f.text_field = "text"
            f.params = {}
            f.model_name = "test-model"
            f.max_perplexity = 1000.0
            f.min_perplexity = 5.0
            f.batch_size = 8
            f.device = "cpu"
            f._available = True
            f._loaded = True

            keep, info = f.filter({"text": ""})
            assert keep is False
            assert "empty" in info["reason"]

    def test_filter_with_mock_perplexity(self):
        """Filter should keep docs within perplexity range."""
        with patch("dq.model_filters.perplexity._check_transformers", return_value=True):
            from dq.model_filters.perplexity import PerplexityFilter
            f = PerplexityFilter.__new__(PerplexityFilter)
            f.text_field = "text"
            f.params = {}
            f.model_name = "test-model"
            f.max_perplexity = 1000.0
            f.min_perplexity = 5.0
            f.batch_size = 8
            f.device = "cpu"
            f._available = True
            f._loaded = True

            with patch.object(f, "_compute_perplexity", return_value=50.0):
                keep, info = f.filter({"text": "normal text"})
                assert keep is True
                assert info["perplexity"] == 50.0

    def test_filter_rejects_high_perplexity(self):
        """Filter should reject gibberish (high perplexity)."""
        with patch("dq.model_filters.perplexity._check_transformers", return_value=True):
            from dq.model_filters.perplexity import PerplexityFilter
            f = PerplexityFilter.__new__(PerplexityFilter)
            f.text_field = "text"
            f.params = {}
            f.model_name = "test-model"
            f.max_perplexity = 1000.0
            f.min_perplexity = 5.0
            f.batch_size = 8
            f.device = "cpu"
            f._available = True
            f._loaded = True

            with patch.object(f, "_compute_perplexity", return_value=5000.0):
                keep, info = f.filter({"text": "asdlkfj asdlfkj"})
                assert keep is False
                assert "too high" in info["reason"]

    def test_filter_rejects_low_perplexity(self):
        """Filter should reject boilerplate (low perplexity)."""
        with patch("dq.model_filters.perplexity._check_transformers", return_value=True):
            from dq.model_filters.perplexity import PerplexityFilter
            f = PerplexityFilter.__new__(PerplexityFilter)
            f.text_field = "text"
            f.params = {}
            f.model_name = "test-model"
            f.max_perplexity = 1000.0
            f.min_perplexity = 5.0
            f.batch_size = 8
            f.device = "cpu"
            f._available = True
            f._loaded = True

            with patch.object(f, "_compute_perplexity", return_value=2.0):
                keep, info = f.filter({"text": "the the the the the"})
                assert keep is False
                assert "too low" in info["reason"]

    def test_filter_registered(self):
        """PerplexityFilter should be registered in the pipeline registry."""
        import dq.model_filters  # noqa: F401
        from dq.pipeline import get_filter_class
        cls = get_filter_class("perplexity")
        assert cls.__name__ == "PerplexityFilter"


