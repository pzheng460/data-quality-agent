"""Tests for the benchmark module using mock datasets (no network downloads)."""

import json
from unittest.mock import MagicMock, patch

from dq.filters import ensure_registered; ensure_registered()
from dq.benchmark import (
    BenchmarkReport,
    BenchmarkResult,
    DatasetResult,
    FilterResult,
    SFTScores,
    _extract_sft_fields,
    _merge_alpaca_fields,
    detect_data_type,
    run_benchmark,
    run_llm_scoring,
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
                FilterConfig("gopher_quality", params={"min_words": 20, "max_words": 100000, "min_stopwords": 0, "min_lines_end_punct": 0.0}),
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
    - name: gopher_quality
      params:
        min_words: 10
        max_words: 100000
        min_stopwords: 0
        min_lines_end_punct: 0.0
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
            assert len(dr.per_filter) > 0
            # All per-filter totals should equal num_docs (independent evaluation)
            for fname, fr in dr.per_filter.items():
                assert fr.total == 10

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

    def test_rich_print_single_dataset_report(self, capsys):
        """With a single dataset, should print a quality report, not a comparison."""
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
        assert "Data Quality Report: Only" in output
        assert "Pass Rate" in output


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


def _make_sft_docs(n: int = 20) -> list[dict]:
    """Create mock SFT documents with instruction/output fields."""
    docs = []
    for i in range(n):
        docs.append({
            "instruction": f"Explain the concept of topic {i} in detail.",
            "input": "",
            "output": (
                f"Topic {i} is an important concept in science. "
                "It involves multiple interconnected processes that work together. "
                "The key aspects include theoretical foundations and practical applications."
            ),
            "text": f"Explain the concept of topic {i} in detail.\n"
                    f"Topic {i} is an important concept in science. "
                    "It involves multiple interconnected processes that work together. "
                    "The key aspects include theoretical foundations and practical applications.",
        })
    return docs


class TestDetectDataType:
    def test_sft_with_instruction_field(self):
        docs = [{"instruction": "Do X", "output": "Y"} for _ in range(5)]
        assert detect_data_type(docs) == "sft"

    def test_sft_with_conversations_field(self):
        docs = [{"conversations": [{"role": "user", "content": "hi"}]} for _ in range(5)]
        assert detect_data_type(docs) == "sft"

    def test_pretrain_with_text_only(self):
        docs = [{"text": "Some long text here"} for _ in range(5)]
        assert detect_data_type(docs) == "pretrain"

    def test_empty_list(self):
        assert detect_data_type([]) == "pretrain"

    def test_mixed_fields_majority_sft(self):
        docs = [
            {"instruction": "Do X", "output": "Y"},
            {"instruction": "Do Z", "output": "W"},
            {"instruction": "Do A", "output": "B"},
            {"text": "plain text"},
        ]
        assert detect_data_type(docs) == "sft"

    def test_mixed_fields_majority_pretrain(self):
        docs = [
            {"text": "plain text 1"},
            {"text": "plain text 2"},
            {"text": "plain text 3"},
            {"instruction": "Do X", "output": "Y"},
        ]
        assert detect_data_type(docs) == "pretrain"


class TestExtractSFTFields:
    def test_with_instruction_output(self):
        doc = {"instruction": "Do X", "output": "Result Y"}
        instr, out = _extract_sft_fields(doc)
        assert instr == "Do X"
        assert out == "Result Y"

    def test_with_text_only(self):
        doc = {"text": "Question here\nAnswer is this"}
        instr, out = _extract_sft_fields(doc)
        assert instr == "Question here"
        assert out == "Answer is this"

    def test_with_text_no_newline(self):
        doc = {"text": "Just a single line"}
        instr, out = _extract_sft_fields(doc)
        assert instr == "Just a single line"
        assert out == ""

    def test_empty_doc(self):
        doc = {}
        instr, out = _extract_sft_fields(doc)
        assert instr == ""
        assert out == ""


class TestSFTScores:
    def test_defaults(self):
        scores = SFTScores()
        assert scores.high_count == 0
        assert scores.low_count == 0
        assert scores.high_rate == 0.0
        assert scores.num_scored == 0
        assert scores.scoring_errors == 0

    def test_to_dict(self):
        scores = SFTScores(
            high_count=6,
            low_count=2,
            high_rate=0.75,
            rule_fail_counts={"completeness": 1, "factuality": 1},
            num_scored=8,
            scoring_errors=0,
        )
        d = scores.to_dict()
        assert d["type"] == "sft"
        assert d["high_count"] == 6
        assert d["low_count"] == 2
        assert d["high_rate"] == 0.75
        assert d["num_scored"] == 8
        assert d["rule_fail_counts"]["completeness"] == 1


class TestDatasetResultNewFields:
    def test_data_type_field(self):
        dr = DatasetResult(name="test", num_docs=10, data_type="sft")
        assert dr.data_type == "sft"

    def test_llm_scores_field(self):
        scores_dict = {"type": "sft", "high_rate": 0.75}
        dr = DatasetResult(name="test", num_docs=10, llm_scores=scores_dict)
        assert dr.llm_scores["high_rate"] == 0.75

    def test_default_data_type(self):
        dr = DatasetResult(name="test", num_docs=10)
        assert dr.data_type == "pretrain"

    def test_stats_optional(self):
        """Stats is None for SFT-only benchmark."""
        dr = DatasetResult(name="test", num_docs=10, data_type="sft")
        assert dr.stats is None


class TestBenchmarkReportNewFields:
    def test_llm_scoring_fields(self):
        report = BenchmarkReport(
            llm_scoring_enabled=True,
            llm_samples=50,
        )
        assert report.llm_scoring_enabled is True
        assert report.llm_samples == 50

    def test_defaults(self):
        report = BenchmarkReport()
        assert report.llm_scoring_enabled is False
        assert report.llm_samples == 0


class TestRunBenchmarkWithDataType:
    def test_pretrain_explicit(self):
        """Explicit pretrain data_type runs heuristic filters."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(10),
                "Bad": _make_bad_docs(10),
            },
            no_dedup=True,
            data_type="pretrain",
        )
        # Should have filter results
        for dr in report.datasets.values():
            assert len(dr.per_filter) > 0

    def test_auto_detect_pretrain(self):
        """Auto-detect should classify text-only docs as pretrain."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(10),
                "Bad": _make_bad_docs(10),
            },
            no_dedup=True,
            data_type="auto",
        )
        # Text-only docs should be detected as pretrain
        for dr in report.datasets.values():
            assert len(dr.per_filter) > 0

    def test_sft_samples_zero_skips_llm(self):
        """sft_samples=0 should skip LLM scoring even with data_type set."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(10),
                "Bad": _make_bad_docs(10),
            },
            no_dedup=True,
            sft_samples=0,
        )
        assert report.llm_scoring_enabled is False


class TestRunLLMScoring:
    def test_llm_scoring_sft_mocked(self):
        """Test LLM scoring with mocked API calls."""
        # Create a base report
        report = run_benchmark(
            datasets={
                "A": _make_sft_docs(5),
                "B": _make_sft_docs(5),
            },
            no_dedup=True,
            data_type="pretrain",  # Force pretrain for Layer 1
        )

        # Mock the scorer
        with patch("dq.benchmark._score_docs") as mock_score:
            mock_score.return_value = {
                "type": "sft",
                "avg_complexity": 3.0,
                "avg_quality": 4.0,
                "complexity_distribution": {3: 5},
                "quality_distribution": {4: 5},
                "empty_output_ratio": 0.0,
                "num_scored": 5,
                "scoring_errors": 0,
            }

            result = run_llm_scoring(
                report=report,
                datasets={
                    "A": _make_sft_docs(5),
                    "B": _make_sft_docs(5),
                },
                llm_samples=5,
                data_type_override="sft",
            )

            assert result.llm_scoring_enabled is True
            assert result.llm_samples == 5
            for name in ["A", "B"]:
                dr = result.datasets[name]
                assert dr.llm_scores is not None
                assert dr.llm_scores["type"] == "sft"
                assert dr.data_type == "sft"


class TestBenchmarkReportWithLLMScores:
    def _make_report_with_llm(self) -> BenchmarkReport:
        """Create a report with LLM scores for testing display."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(10),
                "Bad": _make_bad_docs(10),
            },
            no_dedup=True,
        )
        # Manually add LLM scores
        report.llm_scoring_enabled = True
        report.llm_samples = 5
        for name in report.datasets:
            report.datasets[name].data_type = "sft"
            report.datasets[name].llm_scores = {
                "type": "sft",
                "high_count": 3,
                "low_count": 2,
                "high_rate": 0.6,
                "rule_fail_counts": {"completeness": 1, "factuality": 1},
                "num_scored": 5,
                "scoring_errors": 0,
            }
        return report

    def test_json_includes_llm_scores(self):
        report = self._make_report_with_llm()
        json_str = benchmark_to_json(report)
        data = json.loads(json_str)

        assert data["llm_scoring_enabled"] is True
        assert data["llm_samples"] == 5
        for ds in data["datasets"].values():
            assert "llm_scores" in ds
            assert ds["llm_scores"]["type"] == "sft"
            assert ds["data_type"] == "sft"

    def test_markdown_includes_llm_section(self):
        report = self._make_report_with_llm()
        md = benchmark_to_markdown(report)
        assert "LLM Binary Judge" in md
        assert "HIGH quality rate" in md

    def test_rich_print_with_llm_no_crash(self):
        """Ensure rich console output with LLM scores doesn't crash."""
        from rich.console import Console

        report = self._make_report_with_llm()
        test_console = Console(file=None, force_terminal=False, no_color=True, width=120)
        print_benchmark_report(report, console=test_console)


class TestLoadHfDataset:
    """Tests for load_hf_dataset with mocked HF datasets library."""

    def test_load_hf_dataset_basic(self):
        """load_hf_dataset should stream N docs from a HF dataset."""
        from dq.benchmark.datasets import load_hf_dataset

        fake_items = [{"text": f"Document number {i} with enough content."} for i in range(100)]

        def fake_load_dataset(dataset_id, *, split, streaming, **kwargs):
            return iter(fake_items)

        with patch("dq.benchmark.datasets._ensure_datasets", return_value=fake_load_dataset):
            docs = load_hf_dataset("fake/dataset", n=10)
            assert len(docs) == 10
            assert all("text" in d for d in docs)

    def test_load_hf_dataset_custom_text_field(self):
        """load_hf_dataset should respect custom text_field."""
        from dq.benchmark.datasets import load_hf_dataset

        fake_items = [{"content": f"Doc {i} text here."} for i in range(50)]

        def fake_load_dataset(dataset_id, *, split, streaming, **kwargs):
            return iter(fake_items)

        with patch("dq.benchmark.datasets._ensure_datasets", return_value=fake_load_dataset):
            docs = load_hf_dataset("fake/dataset", n=5, text_field="content")
            assert len(docs) == 5
            assert all(d["text"].startswith("Doc") for d in docs)

    def test_load_hf_dataset_skips_empty(self):
        """load_hf_dataset should skip items with empty text."""
        from dq.benchmark.datasets import load_hf_dataset

        fake_items = [{"text": ""}, {"text": "Valid document."}, {"text": ""}]

        def fake_load_dataset(dataset_id, *, split, streaming, **kwargs):
            return iter(fake_items)

        with patch("dq.benchmark.datasets._ensure_datasets", return_value=fake_load_dataset):
            docs = load_hf_dataset("fake/dataset", n=10)
            assert len(docs) == 1
            assert docs[0]["text"] == "Valid document."


class TestSingleDatasetBenchmark:
    """Tests for running benchmark with a single dataset (like dq bench file.jsonl)."""

    def test_single_dataset_report(self):
        """Single dataset should produce valid report with per-rule stats."""
        report = run_benchmark(
            datasets={"TestData": _make_good_docs(20)},
            no_dedup=True,
        )
        assert "TestData" in report.datasets
        assert len(report.datasets) == 1
        result = report.datasets["TestData"]
        assert result.overall_pass_rate > 0
        assert len(result.per_filter) > 0

    def test_single_dataset_has_rule_stats(self):
        """Single dataset benchmark should still collect per-rule stats."""
        report = run_benchmark(
            datasets={"TestData": _make_good_docs(20)},
            no_dedup=True,
        )
        assert "TestData" in report.rule_stats
        rule_stats = report.rule_stats["TestData"]
        # Should have at least gopher_quality rules
        assert "gopher_quality" in rule_stats or len(rule_stats) > 0

    def test_single_dataset_json_output(self):
        """Single dataset report should produce valid JSON."""
        report = run_benchmark(
            datasets={"TestData": _make_good_docs(20)},
            no_dedup=True,
        )
        json_str = benchmark_to_json(report)
        data = json.loads(json_str)
        assert "datasets" in data
        assert "TestData" in data["datasets"]

    def test_single_dataset_markdown_output(self):
        """Single dataset report should produce valid Markdown."""
        report = run_benchmark(
            datasets={"TestData": _make_good_docs(20)},
            no_dedup=True,
        )
        md = benchmark_to_markdown(report)
        assert "TestData" in md
        assert "Overall pass rate" in md


class TestBenchCLI:
    """Tests for the dq bench CLI command with custom input."""

    def test_bench_with_local_file(self, tmp_path):
        """dq bench should accept a local file as input."""
        import json as json_mod
        from click.testing import CliRunner
        from dq.cli import main

        # Create test JSONL file
        data_file = tmp_path / "test.jsonl"
        docs = _make_good_docs(10)
        with open(data_file, "w") as f:
            for doc in docs:
                f.write(json_mod.dumps(doc) + "\n")

        output_dir = tmp_path / "output"
        runner = CliRunner()
        result = runner.invoke(main, [
            "bench", str(data_file), "-n", "10", "-o", str(output_dir),
        ])
        assert result.exit_code == 0
        assert (output_dir / "benchmark.json").exists()
        assert (output_dir / "benchmark.md").exists()

    def test_bench_no_input_shows_usage(self):
        """dq bench without input should show usage error."""
        from click.testing import CliRunner
        from dq.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["bench"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_bench_reports_saved_by_default(self, tmp_path):
        """Reports should always be saved to output dir."""
        import json as json_mod
        from click.testing import CliRunner
        from dq.cli import main

        data_file = tmp_path / "test.jsonl"
        docs = _make_good_docs(10)
        with open(data_file, "w") as f:
            for doc in docs:
                f.write(json_mod.dumps(doc) + "\n")

        output_dir = tmp_path / "reports"
        runner = CliRunner()
        result = runner.invoke(main, [
            "bench", str(data_file), "-o", str(output_dir),
        ])
        assert result.exit_code == 0
        assert (output_dir / "benchmark.json").exists()
        assert (output_dir / "benchmark.md").exists()

        # Verify JSON is valid
        data = json_mod.loads((output_dir / "benchmark.json").read_text())
        assert "datasets" in data
