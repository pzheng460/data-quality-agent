"""Tests for the WebDataset tar packager."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

from dq.shared.webdataset import WebDatasetWriter


def _docs(n: int, with_figures: bool = True):
    for i in range(n):
        fig_list = [{"name": f"fig.png", "bytes": b"\x89PNG" + bytes([i])}] if with_figures else []
        yield {
            "id": f"arxiv_{i:03d}",
            "text": f"doc {i} text",
            "metadata": {"arxiv_id": f"2310.{i:05d}", "figures": fig_list},
        }


def test_basic_write_one_shard(tmp_path):
    with WebDatasetWriter(tmp_path, samples_per_shard=10) as w:
        for d in _docs(3):
            w.write(d)
    shards = sorted(tmp_path.glob("shard-*.tar"))
    assert len(shards) == 1
    with tarfile.open(shards[0]) as tf:
        names = sorted(m.name for m in tf.getmembers())
    # Each doc contributes .txt + .json + one figure
    assert "arxiv_000.txt" in names
    assert "arxiv_000.json" in names
    assert "arxiv_000.fig_00.png" in names


def test_rotation_when_shard_full(tmp_path):
    with WebDatasetWriter(tmp_path, samples_per_shard=2) as w:
        for d in _docs(5):
            w.write(d)
    shards = sorted(tmp_path.glob("shard-*.tar"))
    assert len(shards) == 3  # 2+2+1


def test_docs_without_figures_still_write(tmp_path):
    with WebDatasetWriter(tmp_path, samples_per_shard=10) as w:
        for d in _docs(2, with_figures=False):
            w.write(d)
    with tarfile.open(next(tmp_path.glob("shard-*.tar"))) as tf:
        names = sorted(m.name for m in tf.getmembers())
    # No figure files, just .txt + .json each
    assert names == ["arxiv_000.json", "arxiv_000.txt",
                     "arxiv_001.json", "arxiv_001.txt"]


def test_metadata_roundtrip(tmp_path):
    with WebDatasetWriter(tmp_path, samples_per_shard=10) as w:
        w.write({"id": "x", "text": "hi", "metadata": {"arxiv_id": "2310.00001"}})
    [shard] = list(tmp_path.glob("*.tar"))
    with tarfile.open(shard) as tf:
        m = tf.getmember("x.json")
        meta = json.loads(tf.extractfile(m).read())
    assert meta["metadata"]["arxiv_id"] == "2310.00001"


def test_figures_from_disk_path(tmp_path):
    img_path = tmp_path / "fig.png"
    img_path.write_bytes(b"\x89PNG disk bytes")
    doc = {
        "id": "p1",
        "text": "t",
        "metadata": {"figures": [{"name": "fig.png", "path": str(img_path)}]},
    }
    with WebDatasetWriter(tmp_path / "out", samples_per_shard=10) as w:
        w.write(doc)
    [shard] = list((tmp_path / "out").glob("*.tar"))
    with tarfile.open(shard) as tf:
        content = tf.extractfile("p1.fig_00.png").read()
    assert content == b"\x89PNG disk bytes"
