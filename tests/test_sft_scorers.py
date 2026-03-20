"""Tests for SFT scoring modules (DEITA-style).

Uses mocks to avoid requiring actual API keys or models in CI.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# ComplexityScorer
# ---------------------------------------------------------------------------

class TestComplexityScorer:
    """Tests for DEITA ComplexityScorer."""

    def test_skip_when_no_openai(self):
        """Scorer should return -1 scores when openai is not installed."""
        with patch("dq.sft.complexity._check_openai", return_value=False):
            from dq.sft.complexity import ComplexityScorer
            scorer = ComplexityScorer.__new__(ComplexityScorer)
            scorer.api_url = None
            scorer.api_key = None
            scorer.model = "gpt-4o-mini"
            scorer.batch_size = 10
            scorer.max_retries = 3
            scorer.retry_delay = 0.01
            scorer._client = None
            scorer._available = False

            doc = {"instruction": "Write a poem about trees."}
            result = scorer.score(doc)
            assert result["complexity_score"] == -1.0

    def test_score_with_mock_api(self):
        """Scorer should call API and parse integer response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "4"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("dq.sft.complexity._check_openai", return_value=True):
            from dq.sft.complexity import ComplexityScorer
            scorer = ComplexityScorer.__new__(ComplexityScorer)
            scorer.api_url = None
            scorer.api_key = None
            scorer.model = "gpt-4o-mini"
            scorer.batch_size = 10
            scorer.max_retries = 3
            scorer.retry_delay = 0.01
            scorer._client = mock_client
            scorer._available = True

            doc = {"instruction": "Design a distributed caching system with consistency guarantees."}
            result = scorer.score(doc)
            assert result["complexity_score"] == 4.0
            mock_client.chat.completions.create.assert_called_once()

    def test_score_empty_instruction(self):
        """Scorer should return -1 for empty instruction."""
        with patch("dq.sft.complexity._check_openai", return_value=True):
            from dq.sft.complexity import ComplexityScorer
            scorer = ComplexityScorer.__new__(ComplexityScorer)
            scorer.api_url = None
            scorer.api_key = None
            scorer.model = "gpt-4o-mini"
            scorer.batch_size = 10
            scorer.max_retries = 3
            scorer.retry_delay = 0.01
            scorer._client = MagicMock()
            scorer._available = True

            doc = {"instruction": ""}
            result = scorer.score(doc)
            assert result["complexity_score"] == -1.0

    def test_score_clamps_to_range(self):
        """Scorer should clamp response to 1-6 range."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "8"  # Out of range
        mock_client.chat.completions.create.return_value = mock_response

        with patch("dq.sft.complexity._check_openai", return_value=True):
            from dq.sft.complexity import ComplexityScorer
            scorer = ComplexityScorer.__new__(ComplexityScorer)
            scorer.api_url = None
            scorer.api_key = None
            scorer.model = "gpt-4o-mini"
            scorer.batch_size = 10
            scorer.max_retries = 3
            scorer.retry_delay = 0.01
            scorer._client = mock_client
            scorer._available = True

            doc = {"instruction": "hello"}
            result = scorer.score(doc)
            assert result["complexity_score"] == 6.0

    def test_score_batch(self):
        """Batch scoring should add scores to all docs."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "3"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("dq.sft.complexity._check_openai", return_value=True):
            from dq.sft.complexity import ComplexityScorer
            scorer = ComplexityScorer.__new__(ComplexityScorer)
            scorer.api_url = None
            scorer.api_key = None
            scorer.model = "gpt-4o-mini"
            scorer.batch_size = 10
            scorer.max_retries = 3
            scorer.retry_delay = 0.01
            scorer._client = mock_client
            scorer._available = True

            docs = [
                {"instruction": "What is 2+2?"},
                {"instruction": "Explain quantum computing."},
            ]
            results = scorer.score_batch(docs)
            assert len(results) == 2
            assert all("complexity_score" in d for d in results)


# ---------------------------------------------------------------------------
# QualityScorer
# ---------------------------------------------------------------------------

class TestQualityScorer:
    """Tests for DEITA QualityScorer."""

    def test_skip_when_no_openai(self):
        """Scorer should return -1 scores when openai is not installed."""
        with patch("dq.sft.quality._check_openai", return_value=False):
            from dq.sft.quality import QualityScorer
            scorer = QualityScorer.__new__(QualityScorer)
            scorer.api_url = None
            scorer.api_key = None
            scorer.model = "gpt-4o-mini"
            scorer.batch_size = 10
            scorer.max_retries = 3
            scorer.retry_delay = 0.01
            scorer._client = None
            scorer._available = False

            doc = {"instruction": "Write hello", "output": "Hello!"}
            result = scorer.score(doc)
            assert result["quality_score"] == -1.0

    def test_score_with_mock_api(self):
        """Scorer should call API with instruction + output."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "5"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("dq.sft.quality._check_openai", return_value=True):
            from dq.sft.quality import QualityScorer
            scorer = QualityScorer.__new__(QualityScorer)
            scorer.api_url = None
            scorer.api_key = None
            scorer.model = "gpt-4o-mini"
            scorer.batch_size = 10
            scorer.max_retries = 3
            scorer.retry_delay = 0.01
            scorer._client = mock_client
            scorer._available = True

            doc = {"instruction": "Explain gravity", "output": "Gravity is a force..."}
            result = scorer.score(doc)
            assert result["quality_score"] == 5.0

    def test_score_missing_fields(self):
        """Scorer should return -1 when required fields are missing."""
        with patch("dq.sft.quality._check_openai", return_value=True):
            from dq.sft.quality import QualityScorer
            scorer = QualityScorer.__new__(QualityScorer)
            scorer.api_url = None
            scorer.api_key = None
            scorer.model = "gpt-4o-mini"
            scorer.batch_size = 10
            scorer.max_retries = 3
            scorer.retry_delay = 0.01
            scorer._client = MagicMock()
            scorer._available = True

            doc = {"instruction": "hello"}  # missing output
            result = scorer.score(doc)
            assert result["quality_score"] == -1.0

    def test_score_batch(self):
        """Batch scoring should add scores to all docs."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "4"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("dq.sft.quality._check_openai", return_value=True):
            from dq.sft.quality import QualityScorer
            scorer = QualityScorer.__new__(QualityScorer)
            scorer.api_url = None
            scorer.api_key = None
            scorer.model = "gpt-4o-mini"
            scorer.batch_size = 10
            scorer.max_retries = 3
            scorer.retry_delay = 0.01
            scorer._client = mock_client
            scorer._available = True

            docs = [
                {"instruction": "q1", "output": "a1"},
                {"instruction": "q2", "output": "a2"},
            ]
            results = scorer.score_batch(docs)
            assert len(results) == 2
            assert all("quality_score" in d for d in results)


# ---------------------------------------------------------------------------
# DiversityFilter
# ---------------------------------------------------------------------------

class TestDiversityFilter:
    """Tests for DEITA DiversityFilter."""

    def test_skip_when_no_sentence_transformers(self):
        """Filter should return all docs when sentence-transformers not installed."""
        with patch("dq.sft.diversity._check_sentence_transformers", return_value=False):
            from dq.sft.diversity import DiversityFilter
            f = DiversityFilter.__new__(DiversityFilter)
            f.model_name = "test-model"
            f.threshold = 0.95
            f.batch_size = 32
            f.instruction_field = "instruction"
            f._model = None
            f._available = False

            docs = [{"instruction": "q1"}, {"instruction": "q2"}]
            result = f.filter_batch(docs)
            assert len(result) == 2

    def test_empty_batch(self):
        """Filter should handle empty input."""
        with patch("dq.sft.diversity._check_sentence_transformers", return_value=True):
            from dq.sft.diversity import DiversityFilter
            f = DiversityFilter.__new__(DiversityFilter)
            f.model_name = "test-model"
            f.threshold = 0.95
            f.batch_size = 32
            f.instruction_field = "instruction"
            f._model = None
            f._available = True

            result = f.filter_batch([])
            assert result == []

    def test_diversity_filter_removes_duplicates(self):
        """Filter should remove near-duplicate instructions."""
        import numpy as np

        # Mock embeddings: doc 0 and 1 are near-identical, doc 2 is different
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],  # Near-duplicate of doc 0
            [0.0, 0.0, 1.0],    # Different
        ])
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        with patch("dq.sft.diversity._check_sentence_transformers", return_value=True):
            from dq.sft.diversity import DiversityFilter
            f = DiversityFilter.__new__(DiversityFilter)
            f.model_name = "test-model"
            f.threshold = 0.95
            f.batch_size = 32
            f.instruction_field = "instruction"
            f._model = MagicMock()
            f._available = True

            with patch.object(f, "_compute_embeddings", return_value=embeddings):
                docs = [
                    {"instruction": "What is AI?", "complexity_score": 3.0, "quality_score": 4.0},
                    {"instruction": "What is AI?", "complexity_score": 2.0, "quality_score": 3.0},
                    {"instruction": "Explain gravity.", "complexity_score": 4.0, "quality_score": 5.0},
                ]
                result = f.filter_batch(docs)
                # Should keep doc 0 (higher score) and doc 2 (different)
                assert len(result) == 2
                assert result[0]["complexity_score"] == 3.0  # doc 0 kept
                assert result[1]["instruction"] == "Explain gravity."

    def test_diversity_filter_keeps_higher_scoring_doc(self):
        """When two docs are similar, keep the one with higher composite score."""
        import numpy as np

        # Near-identical embeddings
        embeddings = np.array([
            [1.0, 0.0],
            [0.999, 0.001],
        ])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        with patch("dq.sft.diversity._check_sentence_transformers", return_value=True):
            from dq.sft.diversity import DiversityFilter
            f = DiversityFilter.__new__(DiversityFilter)
            f.model_name = "test-model"
            f.threshold = 0.95
            f.batch_size = 32
            f.instruction_field = "instruction"
            f._model = MagicMock()
            f._available = True

            with patch.object(f, "_compute_embeddings", return_value=embeddings):
                # Doc 1 has higher scores, should be kept
                docs = [
                    {"instruction": "Q", "complexity_score": 2.0, "quality_score": 2.0},  # score = 4
                    {"instruction": "Q", "complexity_score": 5.0, "quality_score": 5.0},  # score = 25
                ]
                result = f.filter_batch(docs)
                assert len(result) == 1
                assert result[0]["complexity_score"] == 5.0

    def test_doc_score_handles_missing_scores(self):
        """_doc_score should treat missing/negative scores as neutral (3.0)."""
        with patch("dq.sft.diversity._check_sentence_transformers", return_value=True):
            from dq.sft.diversity import DiversityFilter
            f = DiversityFilter.__new__(DiversityFilter)
            f.model_name = "test-model"
            f.threshold = 0.95
            f.batch_size = 32
            f.instruction_field = "instruction"
            f._model = None
            f._available = True

            # Missing scores → both default to 3.0 → 9.0
            assert f._doc_score({}) == 9.0
            # Negative scores → treated as neutral
            assert f._doc_score({"complexity_score": -1.0, "quality_score": -1.0}) == 9.0
            # Normal scores
            assert f._doc_score({"complexity_score": 4.0, "quality_score": 5.0}) == 20.0
