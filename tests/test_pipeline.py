"""Tests for pipeline, config, and IO."""

import json
import tempfile
from pathlib import Path

from dq.filters import ensure_registered; ensure_registered()
from dq.config import PipelineConfig, FilterConfig
from dq.pipeline import Pipeline, get_filter_class
from dq.utils.io import read_jsonl, write_jsonl, read_docs, write_docs


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestConfig:
    def test_default_config(self):
        config = PipelineConfig.default()
        assert config.text_field == "text"
        assert len(config.filters) > 0
        assert config.dedup.exact is True

    def test_from_dict(self):
        raw = {
            "pipeline": {
                "text_field": "content",
                "filters": [
                    {"name": "length", "params": {"min_words": 10}},
                ],
                "dedup": {"exact": True},
            }
        }
        config = PipelineConfig.from_dict(raw)
        assert config.text_field == "content"
        assert len(config.filters) == 1
        assert config.filters[0].name == "length"
        assert config.filters[0].params["min_words"] == 10

    def test_from_yaml(self, tmp_path):
        yaml_content = """
pipeline:
  text_field: text
  filters:
    - name: length
      params:
        min_words: 20
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)
        config = PipelineConfig.from_yaml(yaml_path)
        assert config.filters[0].params["min_words"] == 20


class TestFilterRegistry:
    def test_known_filters(self):
        for name in ["gopher_quality", "gopher_repetition", "c4", "fineweb", "pii", "length"]:
            cls = get_filter_class(name)
            assert cls is not None

    def test_unknown_filter_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown filter"):
            get_filter_class("nonexistent_filter")


class TestPipeline:
    def test_end_to_end_good_docs(self):
        config = PipelineConfig(
            text_field="text",
            filters=[
                FilterConfig("length", params={"min_words": 10, "max_words": 10000}),
            ],
        )
        pipeline = Pipeline(config)
        docs = [
            {"text": "This is a document with enough words to pass the length filter easily and more."},
        ]
        result = list(pipeline.run(iter(docs)))
        assert len(result) == 1
        assert pipeline.stats.total_in == 1
        assert pipeline.stats.total_out == 1

    def test_end_to_end_filters_bad(self):
        config = PipelineConfig(
            text_field="text",
            filters=[
                FilterConfig("length", params={"min_words": 50}),
            ],
        )
        pipeline = Pipeline(config)
        docs = [
            {"text": "Too short."},
            {"text": " ".join(["word"] * 60) + "."},
        ]
        result = list(pipeline.run(iter(docs)))
        assert len(result) == 1
        assert pipeline.stats.total_dropped == 1

    def test_dry_run(self):
        config = PipelineConfig(
            text_field="text",
            filters=[FilterConfig("length", params={"min_words": 50})],
        )
        pipeline = Pipeline(config)
        docs = [{"text": "short"}, {"text": " ".join(["word"] * 60) + "."}]
        stats = pipeline.dry_run(iter(docs))
        assert stats.total_in == 2
        assert stats.total_out == 1

    def test_with_fixture_files(self):
        """Run pipeline on good fixture file — all should pass length filter."""
        config = PipelineConfig(
            text_field="text",
            filters=[FilterConfig("length", params={"min_words": 10, "max_words": 100000})],
        )
        pipeline = Pipeline(config)
        docs = list(read_jsonl(FIXTURES_DIR / "sample_good.jsonl"))
        result = list(pipeline.run(iter(docs)))
        assert len(result) == len(docs)


class TestIO:
    def test_jsonl_roundtrip(self, tmp_path):
        docs = [{"text": "hello", "id": 1}, {"text": "world", "id": 2}]
        path = tmp_path / "test.jsonl"
        count = write_jsonl(iter(docs), path)
        assert count == 2

        loaded = list(read_jsonl(path))
        assert len(loaded) == 2
        assert loaded[0]["text"] == "hello"

    def test_csv_roundtrip(self, tmp_path):
        docs = [{"text": "hello", "id": "1"}, {"text": "world", "id": "2"}]
        path = tmp_path / "test.csv"
        count = write_docs(iter(docs), path)
        assert count == 2

        loaded = list(read_docs(path))
        assert len(loaded) == 2
        assert loaded[0]["text"] == "hello"

    def test_parquet_roundtrip(self, tmp_path):
        docs = [{"text": "hello", "id": 1}, {"text": "world", "id": 2}]
        path = tmp_path / "test.parquet"
        count = write_docs(iter(docs), path)
        assert count == 2

        loaded = list(read_docs(path))
        assert len(loaded) == 2
        assert loaded[0]["text"] == "hello"

    def test_auto_detect_jsonl(self, tmp_path):
        docs = [{"text": "test"}]
        path = tmp_path / "data.jsonl"
        write_docs(iter(docs), path)
        loaded = list(read_docs(path))
        assert len(loaded) == 1
