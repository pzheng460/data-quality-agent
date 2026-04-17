"""Shard writer/reader for zstd-compressed JSONL files."""

import hashlib
import json
from pathlib import Path
from typing import Iterator

import zstandard as zstd


class ShardWriter:
    """Write documents to numbered .jsonl.zst shards with auto-rotation by size.

    Usage:
        with ShardWriter(output_dir) as w:
            for doc in docs:
                w.write(doc)
        print(w.shard_info)  # per-shard metadata
    """

    def __init__(
        self,
        output_dir: Path,
        prefix: str = "shard",
        target_bytes: int = 1_073_741_824,
        zstd_level: int = 3,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.target_bytes = target_bytes
        self.zstd_level = zstd_level

        self._shard_idx = 0
        self._fh = None
        self._writer = None
        self._bytes = 0
        self._doc_count = 0
        self._hash = hashlib.sha256()
        self._current_path: Path | None = None

        self.total_docs = 0
        self.shard_info: list[dict] = []

    def _start_shard(self) -> None:
        """Open a new shard file for writing."""
        path = self.output_dir / f"{self.prefix}-{self._shard_idx:05d}.jsonl.zst"
        self._current_path = path
        self._fh = open(path, "wb")
        cctx = zstd.ZstdCompressor(level=self.zstd_level)
        self._writer = cctx.stream_writer(self._fh)
        self._bytes = 0
        self._doc_count = 0
        self._hash = hashlib.sha256()

    def _finish_shard(self) -> None:
        """Close the current shard and record its metadata."""
        if self._writer is None:
            return
        self._writer.close()
        self._fh.close()
        if self._doc_count > 0:
            self.shard_info.append({
                "path": self._current_path.name,
                "num_documents": self._doc_count,
                "compressed_bytes": self._current_path.stat().st_size,
                "sha256": self._hash.hexdigest(),
            })
        self._shard_idx += 1
        self._writer = None
        self._fh = None

    def write(self, doc: dict) -> None:
        """Write a single document. Opens/rotates shards as needed."""
        if self._writer is None:
            self._start_shard()

        line = json.dumps(doc, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        self._writer.write(encoded)
        self._hash.update(encoded)
        self._bytes += len(encoded)
        self._doc_count += 1
        self.total_docs += 1

        # Rotate when uncompressed bytes exceed target
        if self._bytes >= self.target_bytes:
            self._finish_shard()

    def close(self) -> None:
        """Finalize the last shard."""
        self._finish_shard()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # Internal state
    _shard_idx: int = 0


def read_shards(directory: Path) -> Iterator[dict]:
    """Read all .jsonl.zst (or .jsonl) shards from a directory in sorted order."""
    directory = Path(directory)
    if not directory.exists():
        return

    for path in list_shards(directory):
        yield from read_shard(path)


def list_shards(directory: Path) -> list[Path]:
    """Return sorted list of shard files in a directory (jsonl.zst preferred)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    shards = sorted(directory.glob("*.jsonl.zst"))
    if not shards:
        shards = sorted(directory.glob("*.jsonl"))
    return shards


def read_shard(path: Path) -> Iterator[dict]:
    """Read a single shard file (jsonl or jsonl.zst)."""
    path = Path(path)
    if path.name.endswith(".jsonl.zst"):
        from dq.utils.io import read_jsonl_zst
        yield from read_jsonl_zst(path)
    else:
        from dq.utils.io import read_jsonl
        yield from read_jsonl(path)


# ── Shard-level resume markers ──────────────────────────────────────────
#
# A "done" marker is written when a worker has finished processing an input
# shard and its corresponding output shard(s) have been flushed to disk. On
# restart, inputs with a done marker are skipped.

def _marker_dir(stage_output_dir: Path) -> Path:
    return Path(stage_output_dir) / ".done"


def shard_marker_path(stage_output_dir: Path, input_shard_name: str) -> Path:
    """Path of the done marker for a given input shard inside a stage's output."""
    return _marker_dir(stage_output_dir) / input_shard_name


def is_shard_done(stage_output_dir: Path, input_shard_name: str) -> bool:
    return shard_marker_path(stage_output_dir, input_shard_name).exists()


def mark_shard_done(stage_output_dir: Path, input_shard_name: str,
                    meta: dict | None = None) -> None:
    """Write a done marker. Optionally record per-shard stats."""
    marker = shard_marker_path(stage_output_dir, input_shard_name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(meta or {}, ensure_ascii=False))


def clear_shard_markers(stage_output_dir: Path) -> None:
    """Remove all done markers (used on --no-resume / --force)."""
    d = Path(stage_output_dir) / ".done"
    if d.exists():
        for f in d.iterdir():
            f.unlink()


class SingleShardWriter:
    """Write every doc to one named shard, no auto-rotation.

    Use this for shard-level pipelines: 1 input shard → 1 output shard.
    Unlike ShardWriter, this never rotates — the output path is fixed.
    Crashes leave a partial file that the resume logic treats as "not done"
    (the .done marker is only written after successful close).
    """

    def __init__(self, output_path: Path, zstd_level: int = 3):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.zstd_level = zstd_level
        self._fh = None
        self._writer = None
        self._bytes = 0
        self._docs = 0
        self._hash = hashlib.sha256()

    def __enter__(self):
        self._fh = open(self.output_path, "wb")
        self._writer = zstd.ZstdCompressor(level=self.zstd_level).stream_writer(self._fh)
        return self

    def write(self, doc: dict) -> None:
        line = json.dumps(doc, ensure_ascii=False) + "\n"
        encoded = line.encode("utf-8")
        self._writer.write(encoded)
        self._hash.update(encoded)
        self._bytes += len(encoded)
        self._docs += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __exit__(self, exc_type, *_):
        self.close()
        # If the caller raised, drop the partial file so resume re-processes it.
        if exc_type is not None and self.output_path.exists():
            try:
                self.output_path.unlink()
            except OSError:
                pass

    @property
    def num_docs(self) -> int:
        return self._docs

    @property
    def sha256(self) -> str:
        return self._hash.hexdigest()


def write_manifest(output_dir: Path, shard_info: list[dict], version: str = "") -> None:
    """Write manifest.json with per-shard checksums and document counts."""
    from datetime import datetime, timezone

    total_docs = sum(s["num_documents"] for s in shard_info)
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_shards": len(shard_info),
        "total_documents": total_docs,
        "shards": shard_info,
    }
    path = Path(output_dir) / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
