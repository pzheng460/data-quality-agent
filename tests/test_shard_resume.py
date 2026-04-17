"""Tests for shard-level resume helpers in dq/shared/shard.py."""

from __future__ import annotations

import json
from pathlib import Path

from dq.shared.shard import (
    ShardWriter,
    SingleShardWriter,
    list_shards,
    read_shard,
    is_shard_done,
    mark_shard_done,
    shard_marker_path,
    clear_shard_markers,
)


def test_list_shards_empty_dir(tmp_path: Path):
    assert list_shards(tmp_path) == []


def test_single_shard_writer_roundtrip(tmp_path: Path):
    out = tmp_path / "shard-00000.jsonl.zst"
    with SingleShardWriter(out) as w:
        w.write({"id": 1, "text": "a"})
        w.write({"id": 2, "text": "b"})
    assert out.exists()
    docs = list(read_shard(out))
    assert [d["id"] for d in docs] == [1, 2]


def test_single_shard_writer_cleans_partial_on_exception(tmp_path: Path):
    out = tmp_path / "shard-00000.jsonl.zst"
    class Boom(RuntimeError): ...
    try:
        with SingleShardWriter(out) as w:
            w.write({"id": 1, "text": "a"})
            raise Boom("crash mid-shard")
    except Boom:
        pass
    assert not out.exists(), "partial shard should have been deleted on exception"


def test_shard_marker_lifecycle(tmp_path):
    stage_dir = tmp_path / "stage2"
    # Initially not done
    assert not is_shard_done(stage_dir, "shard-00000.jsonl.zst")
    # After marking, done
    mark_shard_done(stage_dir, "shard-00000.jsonl.zst", {"input": 100, "kept": 80, "rejected": 20})
    assert is_shard_done(stage_dir, "shard-00000.jsonl.zst")
    # Marker contents are recoverable
    marker = shard_marker_path(stage_dir, "shard-00000.jsonl.zst")
    with open(marker) as f:
        meta = json.load(f)
    assert meta == {"input": 100, "kept": 80, "rejected": 20}


def test_clear_shard_markers_removes_all(tmp_path: Path):
    stage_dir = tmp_path / "stage2"
    for n in ("a", "b", "c"):
        mark_shard_done(stage_dir, n)
    clear_shard_markers(stage_dir)
    assert not is_shard_done(stage_dir, "a")
    assert not is_shard_done(stage_dir, "b")
    assert not is_shard_done(stage_dir, "c")


def test_list_shards_picks_up_single_shard_writer_outputs(tmp_path: Path):
    with SingleShardWriter(tmp_path / "shard-00000.jsonl.zst") as w:
        w.write({"id": 1})
    with SingleShardWriter(tmp_path / "shard-00001.jsonl.zst") as w:
        w.write({"id": 2})
    paths = list_shards(tmp_path)
    assert [p.name for p in paths] == ["shard-00000.jsonl.zst", "shard-00001.jsonl.zst"]
