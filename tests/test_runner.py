"""Tests for the production runner (shard writer/reader, IO, engine)."""

import json
from pathlib import Path

import pytest

from dq.runner.shard import ShardWriter, read_shards
from dq.runner.stats import PhaseStats, PhaseTimer


class TestShardWriter:
    def test_write_and_read_back(self, tmp_path):
        """Write docs to shard, read them back, verify roundtrip."""
        docs = [{"text": f"Document {i}", "id": f"doc_{i}"} for i in range(10)]

        with ShardWriter(tmp_path / "shards") as w:
            for doc in docs:
                w.write(doc)

        # Read back
        result = list(read_shards(tmp_path / "shards"))
        assert len(result) == 10
        assert result[0]["id"] == "doc_0"
        assert result[9]["id"] == "doc_9"

    def test_shard_rotation(self, tmp_path):
        """Shards rotate when exceeding target_bytes."""
        docs = [{"text": "x" * 500, "id": f"doc_{i}"} for i in range(100)]

        # Very small target to force rotation
        with ShardWriter(tmp_path / "shards", target_bytes=2000) as w:
            for doc in docs:
                w.write(doc)

        # Should have created multiple shards
        shard_files = sorted((tmp_path / "shards").glob("*.jsonl.zst"))
        assert len(shard_files) > 1

        # All docs should be readable
        result = list(read_shards(tmp_path / "shards"))
        assert len(result) == 100

    def test_shard_info(self, tmp_path):
        """shard_info should contain metadata for each shard."""
        docs = [{"text": f"doc {i}", "id": f"doc_{i}"} for i in range(5)]

        w = ShardWriter(tmp_path / "out")
        for doc in docs:
            w.write(doc)
        w.close()

        assert len(w.shard_info) >= 1
        info = w.shard_info[0]
        assert "path" in info
        assert "num_documents" in info
        assert "sha256" in info
        assert info["num_documents"] > 0

    def test_empty_write(self, tmp_path):
        """Writing nothing should produce no shards."""
        with ShardWriter(tmp_path / "empty") as w:
            pass
        assert w.shard_info == []


class TestPhaseStats:
    def test_to_dict(self):
        stats = PhaseStats(
            phase="test",
            input_count=100,
            output_count=80,
            rejected_count=20,
            reject_reasons={"bad_quality": 15, "too_short": 5},
        )
        d = stats.to_dict()
        assert d["phase"] == "test"
        assert d["keep_rate"] == 0.8
        assert d["rejected_count"] == 20

    def test_save_and_load(self, tmp_path):
        stats = PhaseStats(phase="test", input_count=50, output_count=40, rejected_count=10)
        path = tmp_path / "stats.json"
        stats.save(path)

        import json
        with open(path) as f:
            data = json.load(f)
        assert data["phase"] == "test"
        assert data["input_count"] == 50

    def test_timer(self):
        stats = PhaseStats(phase="test")
        with PhaseTimer(stats):
            total = sum(range(1000))
        assert stats.duration_seconds > 0


class TestZstdIO:
    def test_read_write_roundtrip(self, tmp_path):
        """Test .jsonl.zst read/write through utils/io."""
        from dq.utils.io import read_jsonl_zst, write_jsonl_zst

        docs = [{"text": f"hello {i}", "id": i} for i in range(100)]
        path = tmp_path / "test.jsonl.zst" if False else str(tmp_path / "test.jsonl.zst")

        count = write_jsonl_zst(iter(docs), tmp_path / "test.jsonl.zst")
        assert count == 100

        result = list(read_jsonl_zst(tmp_path / "test.jsonl.zst"))
        assert len(result) == 100
        assert result[0]["text"] == "hello 0"

    def test_read_docs_dispatch(self, tmp_path):
        """read_docs should auto-detect .jsonl.zst files."""
        from dq.utils.io import read_docs, write_jsonl_zst

        docs = [{"text": "test", "id": "1"}]
        write_jsonl_zst(iter(docs), tmp_path / "data.jsonl.zst")

        result = list(read_docs(tmp_path / "data.jsonl.zst"))
        assert len(result) == 1
        assert result[0]["text"] == "test"


class TestPhaseEngine:
    def test_success_markers(self, tmp_path):
        """is_phase_done / mark_phase_done should track completion."""
        # Create a minimal config
        config_path = tmp_path / "config.yaml"
        config_path.write_text("pipeline:\n  text_field: text\n  filters: []\n")
        input_path = tmp_path / "input.jsonl"
        input_path.write_text("")

        from dq.runner.engine import PhaseEngine
        engine = PhaseEngine(
            config_path=str(config_path),
            input_path=str(input_path),
            output_dir=str(tmp_path / "output"),
        )

        assert not engine.is_phase_done("phase1_parse")
        engine.mark_phase_done("phase1_parse")
        assert engine.is_phase_done("phase1_parse")


# Import for test
from dq.runner.shard import ShardWriter
