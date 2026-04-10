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

    shard_files = sorted(directory.glob("*.jsonl.zst"))
    if not shard_files:
        shard_files = sorted(directory.glob("*.jsonl"))

    for path in shard_files:
        if path.name.endswith(".jsonl.zst"):
            from dq.utils.io import read_jsonl_zst
            yield from read_jsonl_zst(path)
        else:
            from dq.utils.io import read_jsonl
            yield from read_jsonl(path)


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
