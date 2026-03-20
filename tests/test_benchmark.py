"""Tests for the benchmark module using mock datasets (no network downloads)."""

import json

import dq.filters  # noqa: F401 — trigger filter registration
from dq.benchmark import (
    BenchmarkReport,
    BenchmarkResult,
    DatasetResult,
    FilterResult,
    _merge_alpaca_fields,
    run_benchmark,
)
from dq.benchmark_report import (
    STRONG_THRESHOLD,
    WEAK_THRESHOLD,
    benchmark_to_json,
    benchmark_to_markdown,
    print_benchmark_report,
)
from dq.config import FilterConfig, PipelineConfig


def _make_good_docs(n: int = 50) -> list[dict]:
    """Create mock 'good' documents that should pass most filters."""
    docs = []
    for i in range(n):
        # Well-formed English prose with enough words, proper punctuation,
        # stopwords, and reasonable structure.
        text = (
            f"This is a well-written article about topic number {i}. "
            "The researchers discovered several important findings in their study. "
            "According to the data, the results were statistically significant. "
            "Furthermore, the analysis showed that the proposed method outperforms "
            "existing approaches on multiple benchmarks. The experiment was conducted "
            "over a period of several months with careful controls in place. "
            "In conclusion, the study provides valuable insights into the field. "
            "These findings have implications for future research directions."
        )
        docs.append({"text": text})
    return docs


def _make_bad_docs(n: int = 50) -> list[dict]:
    """Create mock 'mediocre' documents that should fail various filters."""
    docs = []
    for i in range(n):
        # Short, poorly structured text without proper punctuation
        text = f"q{i}: how to do thing\na: just do it lol\nok thanks"
        docs.append({"text": text})
    return docs


class TestRunBenchmark:
    def test_basic_benchmark_with_mock_data(self):
        """Run benchmark with mock good/bad datasets — good should pass more."""
        good_docs = _make_good_docs(30)
        bad_docs = _make_bad_docs(30)

        config = PipelineConfig(
            text_field="text",
            filters=[
                FilterConfig("length", params={"min_words": 20, "max_words": 100000}),
            ],
        )

        report = run_benchmark(
            datasets={"Good": good_docs, "Bad": bad_docs},
            n=30,
            no_dedup=True,
        )

        assert "Good" in report.datasets
        assert "Bad" in report.datasets
        assert report.datasets["Good"].overall_pass_rate > report.datasets["Bad"].overall_pass_rate

    def test_benchmark_with_custom_config(self, tmp_path):
        """Run benchmark with a simple custom config."""
        yaml_content = """
pipeline:
  text_field: text
  filters:
    - name: length
      params:
        min_words: 10
        max_words: 100000
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        good_docs = _make_good_docs(20)
        bad_docs = _make_bad_docs(20)

        report = run_benchmark(
            config_path=str(config_file),
            datasets={"Good": good_docs, "Bad": bad_docs},
            n=20,
            no_dedup=True,
        )

        assert len(report.datasets) == 2

    def test_benchmark_report_fields(self):
        """Verify all expected fields are present in the report."""
        report = run_benchmark(
            datasets={
                "A": _make_good_docs(10),
                "B": _make_bad_docs(10),
            },
            n=10,
            no_dedup=True,
        )

        for name in ["A", "B"]:
            dr = report.datasets[name]
            assert dr.name == name
            assert dr.num_docs == 10
            assert 0.0 <= dr.overall_pass_rate <= 1.0
            assert isinstance(dr.per_filter_pass_rate, dict)
            assert dr.stats is not None

    def test_discrimination_scores(self):
        """Discrimination scores should be non-negative."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(20),
                "Bad": _make_bad_docs(20),
            },
            n=20,
            no_dedup=True,
        )

        scores = report.discrimination_scores()
        for name, delta in scores.items():
            assert delta >= 0.0, f"Negative discrimination for {name}"

    def test_single_dataset_no_discrimination(self):
        """With only one dataset, discrimination scores should be empty."""
        report = run_benchmark(
            datasets={"Only": _make_good_docs(10)},
            n=10,
            no_dedup=True,
        )
        assert report.discrimination_scores() == {}


class TestBenchmarkReport:
    def _make_report(self) -> BenchmarkReport:
        """Helper to create a report for testing."""
        return run_benchmark(
            datasets={
                "Good": _make_good_docs(20),
                "Bad": _make_bad_docs(20),
            },
            n=20,
            no_dedup=True,
        )

    def test_json_output(self):
        report = self._make_report()
        json_str = benchmark_to_json(report)
        data = json.loads(json_str)

        assert "datasets" in data
        assert "Good" in data["datasets"]
        assert "Bad" in data["datasets"]
        assert "discrimination" in data
        assert "overall_pass_rate" in data["datasets"]["Good"]
        assert "per_filter_pass_rate" in data["datasets"]["Good"]

    def test_json_write_to_file(self, tmp_path):
        report = self._make_report()
        out_path = tmp_path / "bench.json"
        benchmark_to_json(report, path=out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "datasets" in data

    def test_markdown_output(self):
        report = self._make_report()
        md = benchmark_to_markdown(report)

        assert "# Data Quality Benchmark" in md
        assert "Good" in md
        assert "Bad" in md
        assert "Overall pipeline" in md

    def test_markdown_write_to_file(self, tmp_path):
        report = self._make_report()
        out_path = tmp_path / "bench.md"
        benchmark_to_markdown(report, path=out_path)
        assert out_path.exists()
        content = out_path.read_text()
        assert "Benchmark" in content

    def test_rich_print_no_crash(self, capsys):
        """Ensure rich console output doesn't crash."""
        from rich.console import Console

        report = self._make_report()
        test_console = Console(file=None, force_terminal=False, no_color=True, width=120)
        # Should not raise
        print_benchmark_report(report, console=test_console)

    def test_rich_print_single_dataset_warning(self, capsys):
        """With < 2 datasets, should print a warning, not crash."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        test_console = Console(file=buf, force_terminal=False, no_color=True, width=120)
        report = run_benchmark(
            datasets={"Only": _make_good_docs(10)},
            n=10,
            no_dedup=True,
        )
        print_benchmark_report(report, console=test_console)
        output = buf.getvalue()
        assert "at least 2 datasets" in output


class TestMergeAlpacaFields:
    def test_all_fields(self):
        item = {"instruction": "Do X", "input": "with Y", "output": "result Z"}
        text = _merge_alpaca_fields(item)
        assert "Do X" in text
        assert "with Y" in text
        assert "result Z" in text

    def test_empty_input(self):
        item = {"instruction": "Do X", "input": "", "output": "result Z"}
        text = _merge_alpaca_fields(item)
        assert "Do X" in text
        assert "result Z" in text
        assert "\n\n" not in text

    def test_missing_fields(self):
        item = {"instruction": "Do X"}
        text = _merge_alpaca_fields(item)
        assert text == "Do X"

    def test_none_fields(self):
        item = {"instruction": "Do X", "input": None, "output": None}
        text = _merge_alpaca_fields(item)
        assert text == "Do X"


class TestFilterResult:
    def test_defaults(self):
        fr = FilterResult()
        assert fr.total == 0
        assert fr.passed == 0
        assert fr.failed == 0
        assert fr.pass_rate == 0.0
        assert fr.sample_failed == []

    def test_with_values(self):
        fr = FilterResult(total=100, passed=80, failed=20, pass_rate=0.8)
        assert fr.total == 100
        assert fr.pass_rate == 0.8


class TestBenchmarkResultAlias:
    def test_alias_is_same_class(self):
        assert BenchmarkResult is BenchmarkReport


class TestPerFilterDetails:
    def test_per_filter_populated(self):
        """Verify per_filter dict has FilterResult objects."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(20),
                "Bad": _make_bad_docs(20),
            },
            no_dedup=True,
        )
        for ds_result in report.datasets.values():
            assert len(ds_result.per_filter) > 0
            for fname, fr in ds_result.per_filter.items():
                assert isinstance(fr, FilterResult)
                # Some filters may see 0 docs if prior filter dropped everything
                assert fr.passed + fr.failed == fr.total
                assert 0.0 <= fr.pass_rate <= 1.0

    def test_sample_failures_collected(self):
        """Bad docs should generate sample_failed entries."""
        report = run_benchmark(
            datasets={
                "Bad": _make_bad_docs(20),
                "Good": _make_good_docs(20),
            },
            no_dedup=True,
        )
        bad_result = report.datasets["Bad"]
        # At least one filter should have failures with samples
        has_samples = any(
            len(fr.sample_failed) > 0
            for fr in bad_result.per_filter.values()
            if fr.failed > 0
        )
        assert has_samples


class TestMarkdownVerdicts:
    def test_markdown_has_verdicts(self):
        report = run_benchmark(
            datasets={
                "Original": _make_bad_docs(30),
                "Cleaned": _make_good_docs(30),
            },
            no_dedup=True,
        )
        md = benchmark_to_markdown(report)
        # Should contain at least one verdict indicator
        assert any(v in md for v in ["✅", "⚠️", "—"])

    def test_json_has_verdicts(self):
        report = run_benchmark(
            datasets={
                "Original": _make_bad_docs(30),
                "Cleaned": _make_good_docs(30),
            },
            no_dedup=True,
        )
        json_str = benchmark_to_json(report)
        data = json.loads(json_str)
        # discrimination dict should have verdict strings
        for f_data in data["discrimination"].values():
            assert "verdict" in f_data
            assert "delta" in f_data


class TestSkipDedupAlias:
    def test_skip_dedup_works(self):
        report = run_benchmark(
            datasets={"A": _make_good_docs(5), "B": _make_bad_docs(5)},
            skip_dedup=True,
        )
        assert len(report.datasets) == 2
