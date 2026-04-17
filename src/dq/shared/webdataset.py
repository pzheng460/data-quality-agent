"""WebDataset tar packager.

Bundles docs + figures into `shard-NNNNN.tar` files using the WebDataset
convention (one sample = a set of files sharing a key prefix). Matches NeMo
Curator / torchdata / HuggingFace datasets' "webdataset" loader.

Per-sample layout:
    <key>.txt             markdown text
    <key>.json            doc metadata (arxiv_id, figures, ...)
    <key>.fig_NN.<ext>    per-figure bytes

Usage:
    with WebDatasetWriter(Path("out/"), samples_per_shard=1000) as w:
        for doc in docs:
            w.write(doc)
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path


class WebDatasetWriter:
    def __init__(self, output_dir: Path | str, samples_per_shard: int = 1000,
                 prefix: str = "shard") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = int(samples_per_shard)
        self.prefix = prefix

        self._idx = 0
        self._count = 0
        self._tar: tarfile.TarFile | None = None

    def __enter__(self) -> "WebDatasetWriter":
        self._open_shard()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def write(self, doc: dict) -> None:
        if self._tar is None:
            self._open_shard()
        key = str(doc.get("id") or f"doc_{self._idx:05d}_{self._count:06d}").replace("/", "_")

        self._add(f"{key}.txt", (doc.get("text") or "").encode("utf-8"))
        meta = {k: v for k, v in doc.items() if k != "text"}
        self._add(f"{key}.json",
                  json.dumps(meta, ensure_ascii=False, default=str).encode("utf-8"))

        for i, fig in enumerate(_figures_of(doc)):
            blob = _figure_bytes(fig)
            if not blob:
                continue
            ext = Path(fig.get("name", "fig.png")).suffix or ".png"
            self._add(f"{key}.fig_{i:02d}{ext}", blob)

        self._count += 1
        if self._count >= self.samples_per_shard:
            self.close()
            self._idx += 1
            self._open_shard()

    def _open_shard(self) -> None:
        self.close()
        path = self.output_dir / f"{self.prefix}-{self._idx:05d}.tar"
        self._tar = tarfile.open(path, "w")
        self._count = 0

    def _add(self, name: str, payload: bytes) -> None:
        assert self._tar is not None
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        self._tar.addfile(info, io.BytesIO(payload))


def _figures_of(doc: dict) -> list[dict]:
    return (doc.get("metadata") or {}).get("figures") or []


def _figure_bytes(fig: dict) -> bytes | None:
    b = fig.get("bytes")
    if b:
        return bytes(b)
    p = fig.get("path")
    if not p:
        return None
    try:
        with open(p, "rb") as fp:
            return fp.read()
    except OSError:
        return None
