"""Tests for contamination detection (n-gram overlap) and report generation."""

import json
from pathlib import Path

import pytest

from dq.stages.curation.contamination.report import (
    BenchmarkContamination,
    ContaminationReport,
    ContaminationResult,
)
from dq.stages.curation.contamination.ngram import (
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
        from dq.stages.curation.contamination.ngram import load_benchmark

        txt_file = tmp_path / "benchmark.txt"
        txt_file.write_text("question one\nquestion two\nquestion three\n")

        texts = load_benchmark(str(txt_file))
        assert len(texts) == 3
        assert texts[0] == "question one"

    def test_load_from_jsonl_file(self, tmp_path):
        """Load benchmark from a JSONL file."""
        from dq.stages.curation.contamination.ngram import load_benchmark

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
        from dq.stages.curation.contamination.ngram import load_benchmark

        jsonl_file = tmp_path / "benchmark.jsonl"
        lines = [json.dumps({"question": "What is AI?"})]
        jsonl_file.write_text("\n".join(lines))

        texts = load_benchmark(str(jsonl_file))
        assert texts[0] == "What is AI?"

    def test_load_unknown_benchmark_raises(self):
        """Unknown benchmark name should raise ValueError."""
        from dq.stages.curation.contamination.ngram import load_benchmark

        with pytest.raises(ValueError, match="Unknown benchmark"):
            load_benchmark("nonexistent_benchmark_xyz")


# ---------------------------------------------------------------------------
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


class TestExports:
    """Tests that the contamination package exports everything."""

    def test_imports(self):
        from dq.stages.curation.contamination import (
            NgramContaminationDetector,
            ContaminationReport,
            ContaminationResult,
            BenchmarkContamination,
            load_benchmark,
        )
        assert NgramContaminationDetector is not None
        assert ContaminationReport is not None
        assert ContaminationResult is not None
        assert BenchmarkContamination is not None
        assert load_benchmark is not None
