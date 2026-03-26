"""Tests for the production data cleaning runner."""

import json
from pathlib import Path

import pytest

from dq.config import FilterConfig, PipelineConfig, DedupConfig
from dq.runner import run_cleaning


def _make_jsonl(path: Path, docs: list[dict]):
    with open(path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


def _make_good_docs(n: int = 20) -> list[dict]:
    docs = []
    for i in range(n):
        text = (
            f"This is a well-written article about topic number {i}. "
            "The researchers discovered several important findings in their study. "
            "According to the data, the results were statistically significant. "
            "Furthermore, the analysis showed that the proposed method outperforms "
            "existing approaches on multiple benchmarks. The experiment was conducted "
            "under controlled conditions and yielded reproducible results that "
            "advance the state of the art in this field of research."
        )
        docs.append({"text": text})
    return docs


def _make_bad_docs(n: int = 10) -> list[dict]:
    return [{"text": f"q{i}: how\na: ok"} for i in range(n)]


def _simple_config(**filter_kwargs) -> PipelineConfig:
    return PipelineConfig(
        text_field="text",
        filters=[
            FilterConfig("gopher_quality", params={
                "min_words": 20, "max_words": 100000,
                "min_stopwords": 0, "min_lines_end_punct": 0.0,
                **filter_kwargs,
            }),
        ],
        dedup=DedupConfig(exact=False),
    )


class TestRunCleaning:

    def test_basic_filtering(self, tmp_path):
        """Good docs pass, bad docs get filtered."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        _make_jsonl(input_file, _make_good_docs(10) + _make_bad_docs(10))

        stats = run_cleaning(
            input_path=str(input_file),
            output_path=str(output_file),
            config=_simple_config(),
            dedup=False,
        )

        assert stats["input_rows"] == 20
        assert 0 < stats["output_rows"] < 20
        assert stats["drop_rate"] > 0
        # Output file exists and has content
        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == stats["output_rows"]

    def test_all_pass(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        _make_jsonl(input_file, _make_good_docs(10))

        stats = run_cleaning(
            input_path=str(input_file),
            output_path=str(output_file),
            config=_simple_config(min_words=5),
            dedup=False,
        )

        assert stats["input_rows"] == 10
        assert stats["output_rows"] == 10

    def test_dedup(self, tmp_path):
        docs = _make_good_docs(5) + _make_good_docs(5)  # 5 unique + 5 duplicates
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        _make_jsonl(input_file, docs)

        config = PipelineConfig(
            text_field="text",
            filters=[],
            dedup=DedupConfig(exact=True),
        )

        stats = run_cleaning(
            input_path=str(input_file),
            output_path=str(output_file),
            config=config,
            dedup=True,
        )

        assert stats["input_rows"] == 10
        assert stats["output_rows"] == 5

    def test_no_filters_passthrough(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        _make_jsonl(input_file, _make_bad_docs(5))

        config = PipelineConfig(
            text_field="text",
            filters=[],
            dedup=DedupConfig(exact=False),
        )

        stats = run_cleaning(
            input_path=str(input_file),
            output_path=str(output_file),
            config=config,
            dedup=False,
        )

        assert stats["input_rows"] == 5
        assert stats["output_rows"] == 5

    def test_parquet_io(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        docs = _make_good_docs(10)
        table = pa.table({"text": [d["text"] for d in docs]})
        input_file = tmp_path / "input.parquet"
        pq.write_table(table, str(input_file))

        output_file = tmp_path / "output.parquet"

        stats = run_cleaning(
            input_path=str(input_file),
            output_path=str(output_file),
            config=_simple_config(min_words=5),
            dedup=False,
        )

        assert stats["output_rows"] == 10
        result = pq.read_table(str(output_file))
        assert result.num_rows == 10

    def test_multiple_filters(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        _make_jsonl(input_file, _make_good_docs(10) + _make_bad_docs(10))

        config = PipelineConfig(
            text_field="text",
            filters=[
                FilterConfig("gopher_quality", params={
                    "min_words": 20, "max_words": 100000,
                    "min_stopwords": 0, "min_lines_end_punct": 0.0,
                }),
                FilterConfig("c4", params={
                    "min_sentences": 2,
                    "remove_no_terminal_punct": True,
                    "remove_javascript_lines": True,
                    "remove_policy_lines": True,
                    "remove_lorem_ipsum": True,
                    "remove_curly_brace": False,
                }),
            ],
            dedup=DedupConfig(exact=False),
        )

        stats = run_cleaning(
            input_path=str(input_file),
            output_path=str(output_file),
            config=config,
            dedup=False,
        )

        assert stats["input_rows"] == 20
        assert stats["output_rows"] < 20

    def test_parallel_workers(self, tmp_path):
        """Explicit worker count should work."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        _make_jsonl(input_file, _make_good_docs(50))

        stats = run_cleaning(
            input_path=str(input_file),
            output_path=str(output_file),
            config=_simple_config(min_words=5),
            dedup=False,
            parallelism=4,
        )

        assert stats["output_rows"] == 50

    def test_missing_input(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_cleaning(
                input_path="/nonexistent/file.jsonl",
                output_path=str(tmp_path / "out.jsonl"),
                config=_simple_config(),
            )


class TestCLIRun:

    def test_cli_run_basic(self, tmp_path):
        from click.testing import CliRunner
        from dq.cli import main

        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"
        _make_jsonl(input_file, _make_good_docs(10))

        runner = CliRunner()
        result = runner.invoke(main, [
            "run", str(input_file), str(output_file), "--no-dedup",
        ])
        assert result.exit_code == 0
        assert "Done" in result.output

    def test_cli_run_missing_input(self):
        from click.testing import CliRunner
        from dq.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "run", "/nonexistent/file.jsonl", "/tmp/out.jsonl",
        ])
        assert result.exit_code == 1
