"""Tests for Phase 3 contamination detection.

Tests n-gram overlap, Min-K% Prob (mocked), TS-Guessing (mocked),
report generation, and CLI command.
"""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dq.contamination.report import (
    BenchmarkContamination,
    ContaminationReport,
    ContaminationResult,
)
from dq.contamination.ngram import (
    NgramContaminationDetector,
    _normalize,
    _extract_ngrams,
)


# ---------------------------------------------------------------------------
# Text normalization and n-gram extraction
# ---------------------------------------------------------------------------


class TestNormalization:
    """Tests for text normalization and n-gram extraction."""

    def test_normalize_lowercase(self):
        assert _normalize("Hello World") == "hello world"

    def test_normalize_strip_punctuation(self):
        assert _normalize("Hello, World!") == "hello world"

    def test_normalize_collapse_whitespace(self):
        assert _normalize("hello   world\t\nfoo") == "hello world foo"

    def test_normalize_empty(self):
        assert _normalize("") == ""

    def test_extract_ngrams_basic(self):
        ngrams = _extract_ngrams("one two three four five", n=3)
        assert ("one", "two", "three") in ngrams
        assert ("two", "three", "four") in ngrams
        assert ("three", "four", "five") in ngrams
        assert len(ngrams) == 3

    def test_extract_ngrams_short_text(self):
        """Text shorter than n should return empty set."""
        ngrams = _extract_ngrams("hello world", n=13)
        assert ngrams == set()

    def test_extract_ngrams_normalizes(self):
        """N-grams should be extracted from normalized text."""
        ngrams = _extract_ngrams("Hello, World! Foo Bar Baz", n=3)
        assert ("hello", "world", "foo") in ngrams


# ---------------------------------------------------------------------------
# NgramContaminationDetector
# ---------------------------------------------------------------------------


class TestNgramContaminationDetector:
    """Tests for the N-gram contamination detector."""

    def test_build_index(self):
        detector = NgramContaminationDetector(n=3, threshold=0.5)
        detector.build_index(["the quick brown fox jumps over the lazy dog"], benchmark_name="test")
        assert "test" in detector._benchmark_indices
        assert len(detector._benchmark_indices["test"]) > 0

    def test_check_contamination_overlapping(self):
        """Document that overlaps heavily with benchmark should be flagged."""
        detector = NgramContaminationDetector(n=3, threshold=0.5)
        benchmark_text = "the quick brown fox jumps over the lazy dog near the river bank"
        detector.build_index([benchmark_text], benchmark_name="test_bm")

        # Same text should have 100% overlap
        result = detector.check_contamination(benchmark_text)
        assert result.is_contaminated is True
        assert result.overlap_ratio == 1.0
        assert result.method == "ngram"
        assert result.matched_benchmark == "test_bm"

    def test_check_contamination_no_overlap(self):
        """Completely different text should not be flagged."""
        detector = NgramContaminationDetector(n=3, threshold=0.5)
        detector.build_index(
            ["the quick brown fox jumps over the lazy dog"],
            benchmark_name="test_bm",
        )

        result = detector.check_contamination(
            "python programming involves writing code for various applications and systems"
        )
        assert result.is_contaminated is False
        assert result.overlap_ratio < 0.5

    def test_check_contamination_partial_overlap(self):
        """Partial overlap below threshold should not be flagged."""
        detector = NgramContaminationDetector(n=3, threshold=0.8)
        detector.build_index(
            ["alpha beta gamma delta epsilon zeta eta theta iota kappa"],
            benchmark_name="greek",
        )

        # Mix some overlapping and non-overlapping text
        result = detector.check_contamination(
            "alpha beta gamma delta completely different words here now and more"
        )
        assert result.overlap_ratio < 0.8  # below threshold

    def test_check_contamination_empty_doc(self):
        """Empty document should return safe result."""
        detector = NgramContaminationDetector(n=3, threshold=0.5)
        detector.build_index(["some benchmark text here"], benchmark_name="test")

        result = detector.check_contamination("")
        assert result.is_contaminated is False
        assert result.total_ngrams == 0

    def test_check_contamination_specific_benchmark(self):
        """Can check against a specific benchmark by name."""
        detector = NgramContaminationDetector(n=3, threshold=0.5)
        detector.build_index(["alpha beta gamma delta epsilon"], benchmark_name="bm1")
        detector.build_index(["one two three four five six"], benchmark_name="bm2")

        result = detector.check_contamination(
            "alpha beta gamma delta epsilon",
            benchmark_name="bm1",
        )
        assert result.is_contaminated is True
        assert result.matched_benchmark == "bm1"

        result2 = detector.check_contamination(
            "alpha beta gamma delta epsilon",
            benchmark_name="bm2",
        )
        assert result2.is_contaminated is False

    def test_scan_dataset(self):
        """Scan a dataset against benchmarks and get a report."""
        detector = NgramContaminationDetector(n=3, threshold=0.5)
        benchmark_text = "the quick brown fox jumps over the lazy dog near the river"
        benchmarks = {"test_bm": [benchmark_text]}

        docs = [
            {"text": benchmark_text},  # contaminated
            {"text": "completely different text about machine learning and neural networks"},  # clean
            {"text": "the quick brown fox jumps over the lazy dog near the river bank"},  # contaminated
        ]

        report = detector.scan_dataset(docs, benchmarks=benchmarks, dataset_name="my_dataset")

        assert report.dataset_name == "my_dataset"
        assert report.total_docs == 3
        assert report.contaminated_docs >= 1
        assert report.contamination_rate > 0
        assert "test_bm" in report.per_benchmark
        assert report.method == "ngram"

    def test_scan_dataset_no_index_raises(self):
        """Should raise if no benchmark indices exist."""
        detector = NgramContaminationDetector(n=3)
        with pytest.raises(ValueError, match="No benchmark indices"):
            detector.scan_dataset([{"text": "hello"}])

    def test_multiple_benchmarks(self):
        """Scanning against multiple benchmarks tracks each separately."""
        detector = NgramContaminationDetector(n=3, threshold=0.5)
        benchmarks = {
            "bm1": ["alpha beta gamma delta epsilon zeta eta theta"],
            "bm2": ["one two three four five six seven eight"],
        }

        docs = [
            {"text": "alpha beta gamma delta epsilon zeta eta theta"},
            {"text": "one two three four five six seven eight"},
            {"text": "completely unrelated text for testing purposes"},
        ]

        report = detector.scan_dataset(docs, benchmarks=benchmarks, dataset_name="test")
        assert "bm1" in report.per_benchmark
        assert "bm2" in report.per_benchmark
        assert report.per_benchmark["bm1"].contaminated_docs >= 1
        assert report.per_benchmark["bm2"].contaminated_docs >= 1

    def test_default_n13_threshold(self):
        """Default parameters should be n=13, threshold=0.8."""
        detector = NgramContaminationDetector()
        assert detector.n == 13
        assert detector.threshold == 0.8


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------


class TestLoadBenchmark:
    """Tests for benchmark loading from files."""

    def test_load_from_text_file(self, tmp_path):
        """Load benchmark from a plain text file."""
        from dq.contamination.ngram import load_benchmark

        txt_file = tmp_path / "benchmark.txt"
        txt_file.write_text("question one\nquestion two\nquestion three\n")

        texts = load_benchmark(str(txt_file))
        assert len(texts) == 3
        assert texts[0] == "question one"

    def test_load_from_jsonl_file(self, tmp_path):
        """Load benchmark from a JSONL file."""
        from dq.contamination.ngram import load_benchmark

        jsonl_file = tmp_path / "benchmark.jsonl"
        lines = [
            json.dumps({"text": "first question"}),
            json.dumps({"text": "second question"}),
        ]
        jsonl_file.write_text("\n".join(lines))

        texts = load_benchmark(str(jsonl_file))
        assert len(texts) == 2
        assert texts[0] == "first question"

    def test_load_from_jsonl_question_field(self, tmp_path):
        """Load benchmark from JSONL with 'question' field."""
        from dq.contamination.ngram import load_benchmark

        jsonl_file = tmp_path / "benchmark.jsonl"
        lines = [json.dumps({"question": "What is AI?"})]
        jsonl_file.write_text("\n".join(lines))

        texts = load_benchmark(str(jsonl_file))
        assert texts[0] == "What is AI?"

    def test_load_unknown_benchmark_raises(self):
        """Unknown benchmark name should raise ValueError."""
        from dq.contamination.ngram import load_benchmark

        with pytest.raises(ValueError, match="Unknown benchmark"):
            load_benchmark("nonexistent_benchmark_xyz")


# ---------------------------------------------------------------------------
# Min-K% Prob Detector (Mocked)
# ---------------------------------------------------------------------------


class TestMinKProbDetector:
    """Tests for MinKProbDetector with mocked model."""

    def test_unavailable_graceful_skip(self):
        """Should return safe result when transformers not installed."""
        with patch("dq.contamination.min_k_prob._TRANSFORMERS_AVAILABLE", False):
            from dq.contamination.min_k_prob import MinKProbDetector

            detector = MinKProbDetector.__new__(MinKProbDetector)
            detector.model_name = "test"
            detector.k_percent = 20.0
            detector.threshold = -0.5
            detector._device_str = "cpu"
            detector._model = None
            detector._tokenizer = None
            detector._device = None

            assert detector.available is False
            result = detector.check_contamination("some text here")
            assert result.is_contaminated is False
            assert result.method == "min_k_prob"

    def test_available_property(self):
        """Available should reflect transformers availability."""
        with patch("dq.contamination.min_k_prob._TRANSFORMERS_AVAILABLE", True):
            from dq.contamination.min_k_prob import MinKProbDetector

            detector = MinKProbDetector.__new__(MinKProbDetector)
            assert detector.available is True

    def test_compute_min_k_prob_unavailable(self):
        """Should return -inf when transformers not available."""
        with patch("dq.contamination.min_k_prob._TRANSFORMERS_AVAILABLE", False):
            from dq.contamination.min_k_prob import MinKProbDetector

            detector = MinKProbDetector.__new__(MinKProbDetector)
            detector.model_name = "test"
            detector.k_percent = 20.0
            detector.threshold = -0.5
            detector._device_str = "cpu"
            detector._model = None
            detector._tokenizer = None
            detector._device = None

            score = detector.compute_min_k_prob("hello world")
            assert score == float("-inf")

    def test_check_contamination_high_score(self):
        """High score (less negative) should flag as contaminated."""
        from dq.contamination.min_k_prob import MinKProbDetector

        detector = MinKProbDetector.__new__(MinKProbDetector)
        detector.model_name = "test"
        detector.k_percent = 20.0
        detector.threshold = -0.5
        detector._device_str = "cpu"
        detector._model = None
        detector._tokenizer = None
        detector._device = None

        # Mock compute_min_k_prob to return a high score
        with patch.object(detector, "compute_min_k_prob", return_value=-0.2):
            result = detector.check_contamination("memorized text")
            assert result.is_contaminated is True
            assert result.score == -0.2

    def test_check_contamination_low_score(self):
        """Low score (very negative) should not flag."""
        from dq.contamination.min_k_prob import MinKProbDetector

        detector = MinKProbDetector.__new__(MinKProbDetector)
        detector.model_name = "test"
        detector.k_percent = 20.0
        detector.threshold = -0.5
        detector._device_str = "cpu"
        detector._model = None
        detector._tokenizer = None
        detector._device = None

        with patch.object(detector, "compute_min_k_prob", return_value=-3.5):
            result = detector.check_contamination("novel text not in training")
            assert result.is_contaminated is False
            assert result.score == -3.5

    def test_check_batch(self):
        """Batch check should process all texts."""
        from dq.contamination.min_k_prob import MinKProbDetector

        detector = MinKProbDetector.__new__(MinKProbDetector)
        detector.model_name = "test"
        detector.k_percent = 20.0
        detector.threshold = -0.5
        detector._device_str = "cpu"
        detector._model = None
        detector._tokenizer = None
        detector._device = None

        scores = [-0.2, -3.5, -0.1]
        with patch.object(detector, "compute_min_k_prob", side_effect=scores):
            results = detector.check_batch(["a", "b", "c"])
            assert len(results) == 3
            assert results[0].is_contaminated is True
            assert results[1].is_contaminated is False
            assert results[2].is_contaminated is True


# ---------------------------------------------------------------------------
# TS-Guessing Detector (Mocked)
# ---------------------------------------------------------------------------


class TestTSGuessingDetector:
    """Tests for TSGuessingDetector with mocked API."""

    def test_unavailable_graceful_skip(self):
        """Should return empty result when openai not installed."""
        with patch("dq.contamination.ts_guessing._OPENAI_AVAILABLE", False):
            from dq.contamination.ts_guessing import TSGuessingDetector

            detector = TSGuessingDetector.__new__(TSGuessingDetector)
            detector.model = "test"
            detector.max_retries = 1
            detector.retry_delay = 0.0
            detector._client = None

            assert detector.available is False
            result = detector.check_mcq_contamination(
                question="What is 2+2?",
                choices=["3", "4", "5", "6"],
                correct_idx=1,
            )
            assert result.is_correct is False
            assert result.num_choices == 4

    def test_check_mcq_correct_guess(self):
        """Should detect when model guesses correctly."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "B"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("dq.contamination.ts_guessing._OPENAI_AVAILABLE", True):
            from dq.contamination.ts_guessing import TSGuessingDetector

            detector = TSGuessingDetector.__new__(TSGuessingDetector)
            detector.model = "test"
            detector.max_retries = 1
            detector.retry_delay = 0.0
            detector._client = mock_client

            result = detector.check_mcq_contamination(
                question="What is the capital of France?",
                choices=["London", "Paris", "Berlin", "Madrid"],
                correct_idx=1,  # B = Paris
            )
            assert result.is_correct is True
            assert result.guessed_idx == 1

    def test_check_mcq_wrong_guess(self):
        """Should detect when model guesses incorrectly."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("dq.contamination.ts_guessing._OPENAI_AVAILABLE", True):
            from dq.contamination.ts_guessing import TSGuessingDetector

            detector = TSGuessingDetector.__new__(TSGuessingDetector)
            detector.model = "test"
            detector.max_retries = 1
            detector.retry_delay = 0.0
            detector._client = mock_client

            result = detector.check_mcq_contamination(
                question="What is 2+2?",
                choices=["3", "4", "5", "6"],
                correct_idx=1,
            )
            assert result.is_correct is False
            assert result.guessed_idx == 0

    def test_scan_mcq_dataset(self):
        """Should scan a dataset and produce a report."""
        # Model always guesses "B" (index 1)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "B"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("dq.contamination.ts_guessing._OPENAI_AVAILABLE", True):
            from dq.contamination.ts_guessing import TSGuessingDetector

            detector = TSGuessingDetector.__new__(TSGuessingDetector)
            detector.model = "test"
            detector.max_retries = 1
            detector.retry_delay = 0.0
            detector._client = mock_client

            items = [
                {"question": "Q1", "choices": ["a", "b", "c", "d"], "correct_idx": 1},
                {"question": "Q2", "choices": ["a", "b", "c", "d"], "correct_idx": 0},
                {"question": "Q3", "choices": ["a", "b", "c", "d"], "correct_idx": 1},
            ]

            report = detector.scan_mcq_dataset(items, dataset_name="test_mcq")
            assert report.dataset_name == "test_mcq"
            assert report.total_docs == 3
            assert report.method == "ts_guessing"

    def test_scan_mcq_unavailable_empty_report(self):
        """Should return empty report when API not available."""
        with patch("dq.contamination.ts_guessing._OPENAI_AVAILABLE", False):
            from dq.contamination.ts_guessing import TSGuessingDetector

            detector = TSGuessingDetector.__new__(TSGuessingDetector)
            detector.model = "test"
            detector.max_retries = 1
            detector.retry_delay = 0.0
            detector._client = None

            items = [{"question": "Q1", "choices": ["a", "b"], "correct_idx": 0}]
            report = detector.scan_mcq_dataset(items)
            assert report.total_docs == 1
            assert report.contaminated_docs == 0

    def test_api_failure_retry(self):
        """Should retry on API failure."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            Exception("rate limit"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="A"))]),
        ]

        with patch("dq.contamination.ts_guessing._OPENAI_AVAILABLE", True):
            from dq.contamination.ts_guessing import TSGuessingDetector

            detector = TSGuessingDetector.__new__(TSGuessingDetector)
            detector.model = "test"
            detector.max_retries = 3
            detector.retry_delay = 0.01  # fast for tests
            detector._client = mock_client

            result = detector.check_mcq_contamination(
                question="Q", choices=["a", "b"], correct_idx=0,
            )
            assert result.guessed_idx == 0


# ---------------------------------------------------------------------------
# Binomial test
# ---------------------------------------------------------------------------


class TestBinomialTest:
    """Tests for the binomial test helper."""

    def test_significant_result(self):
        """High accuracy should yield low p-value."""
        from dq.contamination.ts_guessing import _binomial_test

        # 80 correct out of 100 with baseline 0.25 -> very significant
        p = _binomial_test(80, 100, 0.25)
        assert p < 0.001

    def test_random_result(self):
        """Random accuracy should yield high p-value."""
        from dq.contamination.ts_guessing import _binomial_test

        # 25 correct out of 100 with baseline 0.25 -> not significant
        p = _binomial_test(25, 100, 0.25)
        assert p > 0.05

    def test_zero_trials(self):
        from dq.contamination.ts_guessing import _binomial_test
        assert _binomial_test(0, 0, 0.25) == 1.0


# ---------------------------------------------------------------------------
# ContaminationReport
# ---------------------------------------------------------------------------


class TestContaminationReport:
    """Tests for the ContaminationReport dataclass."""

    def _make_report(self) -> ContaminationReport:
        return ContaminationReport(
            dataset_name="test_dataset",
            total_docs=100,
            contaminated_docs=5,
            contamination_rate=0.05,
            method="ngram",
            per_benchmark={
                "mmlu": BenchmarkContamination(
                    benchmark_name="mmlu",
                    total_docs=100,
                    contaminated_docs=3,
                    contamination_rate=0.03,
                    avg_overlap=0.12,
                ),
                "hellaswag": BenchmarkContamination(
                    benchmark_name="hellaswag",
                    total_docs=100,
                    contaminated_docs=2,
                    contamination_rate=0.02,
                    avg_overlap=0.08,
                ),
            },
            sample_contaminated=[
                {"text": "sample contaminated doc", "overlap_ratio": 0.9, "matched_benchmark": "mmlu"},
            ],
        )

    def test_to_dict(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["dataset_name"] == "test_dataset"
        assert d["total_docs"] == 100
        assert d["contaminated_docs"] == 5
        assert "mmlu" in d["per_benchmark"]
        assert "hellaswag" in d["per_benchmark"]

    def test_to_json(self, tmp_path):
        report = self._make_report()
        json_path = tmp_path / "report.json"
        json_str = report.to_json(str(json_path))

        assert json_path.exists()
        data = json.loads(json_str)
        assert data["dataset_name"] == "test_dataset"
        assert data["contamination_rate"] == 0.05

    def test_to_markdown(self, tmp_path):
        report = self._make_report()
        md_path = tmp_path / "report.md"
        md = report.to_markdown(str(md_path))

        assert md_path.exists()
        assert "# Contamination Report: test_dataset" in md
        assert "mmlu" in md
        assert "hellaswag" in md

    def test_print_rich(self):
        """print_rich should not raise."""
        from rich.console import Console
        from io import StringIO

        report = self._make_report()
        buf = StringIO()
        c = Console(file=buf, force_terminal=True)
        report.print_rich(console=c)
        output = buf.getvalue()
        assert "test_dataset" in output


# ---------------------------------------------------------------------------
# ContaminationResult dataclass
# ---------------------------------------------------------------------------


class TestContaminationResult:
    """Tests for the ContaminationResult dataclass."""

    def test_defaults(self):
        r = ContaminationResult(is_contaminated=False)
        assert r.overlap_ratio == 0.0
        assert r.matched_ngrams == 0
        assert r.score == 0.0
        assert r.method == ""

    def test_contaminated_result(self):
        r = ContaminationResult(
            is_contaminated=True,
            overlap_ratio=0.95,
            matched_ngrams=50,
            total_ngrams=53,
            matched_benchmark="mmlu",
            method="ngram",
        )
        assert r.is_contaminated is True
        assert r.overlap_ratio == 0.95


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for the contamination CLI command."""

    def test_contamination_command_with_benchmark_file(self, tmp_path):
        """CLI contamination command should work with a benchmark file."""
        from click.testing import CliRunner
        from dq.cli import main

        # Create input data
        input_file = tmp_path / "data.jsonl"
        docs = [
            {"text": "the quick brown fox jumps over the lazy dog near the river bank by the old mill"},
            {"text": "machine learning is a subset of artificial intelligence that focuses on algorithms"},
        ]
        input_file.write_text("\n".join(json.dumps(d) for d in docs))

        # Create benchmark file
        bm_file = tmp_path / "benchmark.txt"
        bm_file.write_text("the quick brown fox jumps over the lazy dog near the river bank by the old mill\n")

        runner = CliRunner()
        result = runner.invoke(main, [
            "contamination",
            str(input_file),
            "--benchmark-file", str(bm_file),
            "--ngram-size", "3",
            "--threshold", "0.5",
        ])
        assert result.exit_code == 0

    def test_contamination_command_no_benchmarks(self, tmp_path):
        """CLI should fail if no benchmarks provided."""
        from click.testing import CliRunner
        from dq.cli import main

        input_file = tmp_path / "data.jsonl"
        input_file.write_text(json.dumps({"text": "hello"}))

        runner = CliRunner()
        result = runner.invoke(main, ["contamination", str(input_file)])
        assert result.exit_code == 1

    def test_contamination_command_with_output(self, tmp_path):
        """CLI should save reports when output dir specified."""
        from click.testing import CliRunner
        from dq.cli import main

        input_file = tmp_path / "data.jsonl"
        docs = [{"text": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"}]
        input_file.write_text("\n".join(json.dumps(d) for d in docs))

        bm_file = tmp_path / "benchmark.txt"
        bm_file.write_text("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu\n")

        out_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(main, [
            "contamination",
            str(input_file),
            "--benchmark-file", str(bm_file),
            "--ngram-size", "3",
            "-o", str(out_dir),
        ])
        assert result.exit_code == 0
        assert (out_dir / "contamination.json").exists()
        assert (out_dir / "contamination.md").exists()


# ---------------------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------------------


class TestExports:
    """Tests that the contamination package exports everything."""

    def test_imports(self):
        from dq.contamination import (
            NgramContaminationDetector,
            MinKProbDetector,
            TSGuessingDetector,
            ContaminationReport,
            ContaminationResult,
            BenchmarkContamination,
            load_benchmark,
        )
        # All should be importable
        assert NgramContaminationDetector is not None
        assert MinKProbDetector is not None
        assert TSGuessingDetector is not None
        assert ContaminationReport is not None
        assert ContaminationResult is not None
        assert BenchmarkContamination is not None
        assert load_benchmark is not None
