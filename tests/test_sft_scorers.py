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

    def _make_scorer(self, mock_client=None):
        """Create a scorer with optional mock client."""
        from dq.sft.complexity import ComplexityScorer
        scorer = ComplexityScorer(api_key="test-key", model="test-model")
        scorer.max_retries = 1
        scorer.retry_delay = 0.01
        if mock_client is not None:
            # Patch get_client to return our mock
            scorer._get_client = lambda: mock_client
        return scorer

    def _mock_response(self, content: str):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    def test_skip_when_no_client(self):
        """Scorer should return -1 scores when no API client available."""
        from dq.sft.complexity import ComplexityScorer
        scorer = ComplexityScorer()
        scorer._get_client = lambda: None

        doc = {"instruction": "Write a poem about trees."}
        result = scorer.score(doc)
        assert result["complexity_score"] == -1.0

    def test_score_with_mock_api(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("4")
        scorer = self._make_scorer(mock_client)

        doc = {"instruction": "Design a distributed caching system."}
        result = scorer.score(doc)
        assert result["complexity_score"] == 4.0
        mock_client.chat.completions.create.assert_called_once()

    def test_score_empty_instruction(self):
        scorer = self._make_scorer(MagicMock())
        doc = {"instruction": ""}
        result = scorer.score(doc)
        assert result["complexity_score"] == -1.0

    def test_score_clamps_to_range(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("8")
        scorer = self._make_scorer(mock_client)

        doc = {"instruction": "hello"}
        result = scorer.score(doc)
        assert result["complexity_score"] == 6.0  # 8 clamped to 6

    def test_score_batch(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("3")
        scorer = self._make_scorer(mock_client)

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

    def _make_scorer(self, mock_client=None):
        from dq.sft.quality import QualityScorer
        scorer = QualityScorer(api_key="test-key", model="test-model")
        scorer.max_retries = 1
        scorer.retry_delay = 0.01
        if mock_client is not None:
            scorer._get_client = lambda: mock_client
        return scorer

    def _mock_response(self, content: str):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    def test_skip_when_no_client(self):
        from dq.sft.quality import QualityScorer
        scorer = QualityScorer()
        scorer._get_client = lambda: None

        doc = {"instruction": "Write hello", "output": "Hello!"}
        result = scorer.score(doc)
        assert result["quality_score"] == -1.0

    def test_score_with_mock_api(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("5")
        scorer = self._make_scorer(mock_client)

        doc = {"instruction": "Explain gravity", "output": "Gravity is a force..."}
        result = scorer.score(doc)
        assert result["quality_score"] == 5.0

    def test_score_missing_fields(self):
        scorer = self._make_scorer(MagicMock())
        doc = {"instruction": "hello"}  # missing output
        result = scorer.score(doc)
        assert result["quality_score"] == -1.0

    def test_score_batch(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("4")
        scorer = self._make_scorer(mock_client)

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
        import numpy as np

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 0.0, 1.0],
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
                docs = [
                    {"instruction": "What is AI?", "complexity_score": 3.0, "quality_score": 4.0},
                    {"instruction": "What is AI?", "complexity_score": 2.0, "quality_score": 3.0},
                    {"instruction": "Explain gravity.", "complexity_score": 4.0, "quality_score": 5.0},
                ]
                result = f.filter_batch(docs)
                assert len(result) == 2
                assert result[0]["complexity_score"] == 3.0
                assert result[1]["instruction"] == "Explain gravity."

    def test_diversity_filter_keeps_higher_scoring_doc(self):
        import numpy as np

        embeddings = np.array([[1.0, 0.0], [0.999, 0.001]])
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
                    {"instruction": "Q", "complexity_score": 2.0, "quality_score": 2.0},
                    {"instruction": "Q", "complexity_score": 5.0, "quality_score": 5.0},
                ]
                result = f.filter_batch(docs)
                assert len(result) == 1
                assert result[0]["complexity_score"] == 5.0

    def test_doc_score_handles_missing_scores(self):
        with patch("dq.sft.diversity._check_sentence_transformers", return_value=True):
            from dq.sft.diversity import DiversityFilter
            f = DiversityFilter.__new__(DiversityFilter)
            f.model_name = "test-model"
            f.threshold = 0.95
            f.batch_size = 32
            f.instruction_field = "instruction"
            f._model = None
            f._available = True

            assert f._doc_score({}) == 9.0
            assert f._doc_score({"complexity_score": -1.0, "quality_score": -1.0}) == 9.0
            assert f._doc_score({"complexity_score": 4.0, "quality_score": 5.0}) == 20.0
