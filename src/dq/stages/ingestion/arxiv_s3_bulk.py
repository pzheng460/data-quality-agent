"""Bulk arxiv ingestion via the official S3 requester-pays bucket.

Same approach as NeMo Curator / Red-Pajama: list s3://arxiv/src/, download
monthly `arXiv_src_YYMM_NNN.tar` dumps, and extract every paper. No rate
limits; user pays AWS egress (~$0.09/GB, ~$100-150 for the entire corpus).

Data layout on S3:
    arXiv_src_2310_000.tar   (monthly bulk, ~1 GB each)
      └── 2310.00001  (per-paper blob: either a gzipped tarball or a
      └── 2310.00002   single gzipped .tex file, depending on the paper)
      └── ...

Downstream stages treat the yielded docs identically to `arxiv_latexml`.
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import re
import tarfile
from pathlib import Path
from typing import Iterator

from dq.stages.ingestion.base import IngestSource
from dq.stages.ingestion.registry import register_source

logger = logging.getLogger(__name__)

_BUCKET = "arxiv"
_PREFIX = "src/"


@register_source("arxiv_s3_bulk")
class ArxivS3BulkSource(IngestSource):
    """Bulk arxiv LaTeX source via S3 requester-pays — no rate limits.

    Requires `boto3` and AWS credentials (requester-pays). See README.
    """

    name = "arxiv_s3_bulk"
    domain = "arxiv"
    priority = 400
    output_format = "latex"

    @classmethod
    def params_schema(cls):
        return {
            "months": {"type": "list", "label": "Months (YYMM)", "required": False},
            "download_dir": {"type": "string", "label": "Staging dir",
                             "default": "/tmp/arxiv_s3_bulk"},
            "keep_tars": {"type": "bool", "label": "Keep tars after processing",
                          "default": False},
        }

    def __init__(
        self,
        months: list[str] | None = None,
        download_dir: str = "/tmp/arxiv_s3_bulk",
        keep_tars: bool = False,
        **_kwargs,
    ) -> None:
        self.months = set(months) if months else None
        self.download_dir = Path(download_dir)
        self.keep_tars = keep_tars

    def fetch(self, limit: int = 0) -> Iterator[dict]:
        try:
            import boto3  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "arxiv_s3_bulk requires boto3. Install with: pip install boto3"
            ) from e

        s3 = boto3.client("s3")
        self.download_dir.mkdir(parents=True, exist_ok=True)

        tars = self._list_tars(s3)
        logger.info("arxiv_s3_bulk: %d monthly tar(s) to process", len(tars))

        count = 0
        for key in tars:
            if limit and count >= limit:
                return
            local = self.download_dir / os.path.basename(key)
            if not local.exists():
                logger.info("Downloading %s (may take a few minutes)", key)
                s3.download_file(_BUCKET, key, str(local),
                                 ExtraArgs={"RequestPayer": "requester"})
            for doc in self._iter_papers(local, limit - count if limit else 0):
                yield doc
                count += 1
                if limit and count >= limit:
                    break
            if not self.keep_tars and local_is_disposable(local):
                try: local.unlink()
                except OSError: pass
        logger.info("arxiv_s3_bulk yielded %d papers", count)

    def _list_tars(self, s3) -> list[str]:
        paginator = s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=_BUCKET, Prefix=_PREFIX,
                                       RequestPayer="requester"):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if not key.endswith(".tar"):
                    continue
                if self.months:
                    m = re.search(r"arXiv_src_(\d{4})_\d+\.tar$", key)
                    if m and m.group(1) not in self.months:
                        continue
                keys.append(key)
        keys.sort()
        return keys

    def _iter_papers(self, local_tar: Path, remaining: int) -> Iterator[dict]:
        """Yield all papers inside a monthly bulk tar."""
        local_tar_name = str(local_tar.name)
        with tarfile.open(local_tar, "r") as outer:
            seen = 0
            for member in outer:
                if remaining and seen >= remaining:
                    return
                if not member.isfile():
                    continue
                arxiv_id = _parse_arxiv_id(member.name)
                if not arxiv_id:
                    continue
                fobj = outer.extractfile(member)
                if fobj is None:
                    continue
                tex = _extract_tex_blob(fobj.read())
                if not tex or len(tex) < 200:
                    continue
                yield {
                    "id": f"arxiv_{arxiv_id}",
                    "text": tex,
                    "source": "arxiv_s3_bulk",
                    "metadata": {
                        "arxiv_id": arxiv_id,
                        "bulk_tar": local_tar_name,
                    },
                }
                seen += 1

# ── helpers ──

def local_is_disposable(path: Path) -> bool:
    """Conservative guard: only delete files under our staging dir."""
    try:
        return "arxiv_s3_bulk" in str(path.resolve())
    except Exception:
        return False


def _parse_arxiv_id(member_name: str) -> str | None:
    """Handle both new (2310.12345) and old (hep-th/0001001) arxiv IDs."""
    s = member_name
    for ext in (".tar.gz", ".tgz", ".gz", ".tar", ".pdf"):
        if s.endswith(ext):
            s = s[:-len(ext)]
            break

    # Old-style with path separator: "hep-th/0001001", "math.AG/0610001"
    m = re.search(r"([a-z\-]+(?:\.[A-Z]{2,4})?)/(\d{7})$", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    stem = os.path.basename(s)
    # New-style: 2310.12345[vN]
    m = re.match(r"(\d{4}\.\d{4,6})(v\d+)?$", stem)
    if m:
        return m.group(1)

    # Old-style no slash: hep-th0001001
    m = re.match(r"([a-z\-]+)(\d{7})$", stem)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    return None


def _extract_tex_blob(data: bytes) -> str | None:
    """Single-paper blob → merged LaTeX string."""
    # Try gzipped tarball (typical for multi-file submissions)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tex_files: dict[str, str] = {}
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith(".tex"):
                    f = tar.extractfile(m)
                    if f:
                        tex_files[m.name] = f.read().decode("utf-8", errors="replace")
            if tex_files:
                return _merge_tex_files(tex_files)
    except (tarfile.TarError, OSError, gzip.BadGzipFile):
        pass

    # Try single-file gzip (single .tex submission)
    try:
        return gzip.decompress(data).decode("utf-8", errors="replace")
    except OSError:
        pass

    # Raw data (rare)
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def _merge_tex_files(tex_files: dict[str, str]) -> str:
    """Pick the main .tex (contains \\begin{document}) and inline \\input{}s."""
    main = None
    for name, content in tex_files.items():
        if r"\begin{document}" in content:
            main = name
            break
    if main is None:
        main = max(tex_files, key=lambda n: len(tex_files[n]))
    content = tex_files[main]
    for _ in range(3):
        def _resolve(m: re.Match) -> str:
            fname = m.group(1).strip()
            if not fname.endswith(".tex"):
                fname += ".tex"
            for p, body in tex_files.items():
                if p == fname or p.endswith("/" + fname):
                    return body
            return ""
        content = re.sub(r"\\input\{([^}]+)\}", _resolve, content)
    return content
