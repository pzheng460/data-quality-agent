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
            "ids": {
                "type": "list",
                "label": "Arxiv IDs (e.g. 2310.12345). Only these papers are yielded; "
                         "the monthly tars that contain them are downloaded.",
                "required": False,
            },
            "full_archive": {
                "type": "bool",
                "label": "Download entire arxiv corpus (~1.5 TB, ~$135 AWS egress)",
                "default": False,
            },
            "months": {
                "type": "list",
                "label": "Months (YYMM) — ignored if full_archive or ids are set",
                "required": False,
            },
            "download_dir": {
                "type": "string",
                "label": "Local staging dir",
                "default": "/tmp/arxiv_s3_bulk",
            },
            "keep_tars": {
                "type": "bool",
                "label": "Keep monthly tars after processing",
                "default": False,
            },
            "save_figures": {
                "type": "bool",
                "label": "Extract figures (png/jpg/pdf/eps) from paper tarballs",
                "default": False,
            },
            "image_dir": {
                "type": "string",
                "label": "Where to save figures (default: <download_dir>/../images)",
                "required": False,
            },
        }

    def __init__(
        self,
        ids: list[str] | None = None,
        months: list[str] | None = None,
        full_archive: bool = False,
        download_dir: str = "/tmp/arxiv_s3_bulk",
        keep_tars: bool = False,
        save_figures: bool = False,
        image_dir: str | None = None,
        **_kwargs,
    ) -> None:
        self.ids = set(ids) if ids else None
        self.full_archive = bool(full_archive)
        # Precedence: full_archive > ids > months. Populate months automatically
        # when only ids are given so we scan the right tars.
        if self.full_archive:
            self.months = None
        elif self.ids:
            self.months = _months_from_ids(self.ids)
        else:
            self.months = set(months) if months else None
        self.download_dir = Path(download_dir)
        self.keep_tars = keep_tars
        self.save_figures = bool(save_figures)
        self.image_dir = Path(image_dir) if image_dir else self.download_dir.parent / "images"

    def fetch(self, limit: int = 0) -> Iterator[dict]:
        try:
            import boto3  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "arxiv_s3_bulk requires boto3. Install with: pip install boto3"
            ) from e

        s3 = boto3.client("s3")
        self.download_dir.mkdir(parents=True, exist_ok=True)

        entries = self._list_tars_with_size(s3)
        total_bytes = sum(sz for _, sz in entries)
        est_cost = total_bytes / 1e9 * 0.09  # AWS egress ≈ $0.09/GB
        if self.full_archive:
            scope = "FULL arxiv archive"
        elif self.months:
            scope = f"months={sorted(self.months)}"
        else:
            scope = "all months (implicit)"
        logger.warning(
            "arxiv_s3_bulk: %s — %d tar(s), %.1f GB, est. $%.2f egress",
            scope, len(entries), total_bytes / 1e9, est_cost,
        )

        try:
            from tqdm import tqdm
            bar = tqdm(entries, desc="arxiv tars", unit="tar", leave=True)
        except ImportError:
            bar = entries

        count = 0
        remaining_ids = set(self.ids) if self.ids else None
        for key, size in bar:
            if limit and count >= limit:
                return
            # If all requested ids already found, no point downloading more tars.
            if remaining_ids is not None and not remaining_ids:
                break
            local = self.download_dir / os.path.basename(key)
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(f"{os.path.basename(key)} {size/1e6:.0f} MB")
            if not local.exists():
                s3.download_file(
                    _BUCKET, key, str(local),
                    ExtraArgs={"RequestPayer": "requester"},
                )
            for doc in self._iter_papers(local, limit - count if limit else 0):
                yield doc
                count += 1
                # Track which requested ids have been yielded so we can stop early.
                if remaining_ids is not None:
                    aid = doc.get("metadata", {}).get("arxiv_id")
                    if aid in remaining_ids:
                        remaining_ids.discard(aid)
                if limit and count >= limit:
                    break
            if not self.keep_tars and local_is_disposable(local):
                try: local.unlink()
                except OSError: pass

        if remaining_ids:
            logger.info("arxiv_s3_bulk: %d requested ids not found: %s",
                        len(remaining_ids), sorted(remaining_ids)[:5])
        logger.info("arxiv_s3_bulk yielded %d papers", count)

    def _list_tars(self, s3) -> list[str]:
        return [k for k, _ in self._list_tars_with_size(s3)]

    def _list_tars_with_size(self, s3) -> list[tuple[str, int]]:
        """List (key, bytes) for monthly tars, optionally filtered by self.months."""
        paginator = s3.get_paginator("list_objects_v2")
        out: list[tuple[str, int]] = []
        for page in paginator.paginate(
            Bucket=_BUCKET, Prefix=_PREFIX, RequestPayer="requester",
        ):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if not key.endswith(".tar"):
                    continue
                if self.months:
                    m = re.search(r"arXiv_src_(\d{4})_\d+\.tar$", key)
                    if m and m.group(1) not in self.months:
                        continue
                out.append((key, int(obj.get("Size", 0))))
        out.sort()
        return out

    def _iter_papers(self, local_tar: Path, remaining: int) -> Iterator[dict]:
        """Yield all papers inside a monthly bulk tar.

        If self.save_figures is True, also extract image files (.png/.jpg/.pdf/.eps)
        to {self.image_dir}/{arxiv_id}/ and attach their relative paths to the doc.
        """
        local_tar_name = local_tar.name
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
                # Filter: if specific IDs were requested, skip the rest.
                if self.ids and arxiv_id not in self.ids:
                    continue
                fobj = outer.extractfile(member)
                if fobj is None:
                    continue
                raw = fobj.read()

                tex, figures = _extract_tex_and_figures(
                    raw,
                    save_figures=self.save_figures,
                    out_dir=(self.image_dir / arxiv_id.replace("/", "_")) if self.save_figures else None,
                )
                if not tex or len(tex) < 200:
                    continue

                meta = {
                    "arxiv_id": arxiv_id,
                    "bulk_tar": local_tar_name,
                }
                if figures:
                    meta["figures"] = figures
                yield {
                    "id": f"arxiv_{arxiv_id}",
                    "text": tex,
                    "source": "arxiv_s3_bulk",
                    "metadata": meta,
                }
                seen += 1

# ── helpers ──

def local_is_disposable(path: Path) -> bool:
    """Conservative guard: only delete files under our staging dir."""
    try:
        return "arxiv_s3_bulk" in str(path.resolve())
    except Exception:
        return False


def _months_from_ids(ids: set[str]) -> set[str]:
    """Infer which YYMM S3 buckets to scan based on arxiv IDs.

    New-style IDs (YYMM.NNNNN) map directly.
    Old-style IDs (hep-th/0001001) map to YYMM = first 4 digits of the number.
    Unknown IDs are silently ignored — they'll simply not match anything.
    """
    months: set[str] = set()
    for aid in ids:
        m = re.match(r"(\d{4})\.\d{4,6}(?:v\d+)?$", aid)
        if m:
            months.add(m.group(1))
            continue
        m = re.match(r"[a-z\-]+/(\d{4})\d{3}$", aid)
        if m:
            months.add(m.group(1))
    return months


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


# File extensions treated as figure assets inside an arxiv paper tarball.
_FIGURE_EXTS = (".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg", ".gif")


def _extract_tex_and_figures(
    data: bytes,
    *,
    save_figures: bool = False,
    out_dir: Path | None = None,
) -> tuple[str | None, list[dict]]:
    """Parse a single-paper blob → (merged LaTeX, list of figure metadata).

    Figure metadata entries: {"name", "path", "bytes"}, with bytes only retained
    in-memory; when save_figures=True, bytes are written to disk under out_dir
    and the dict has {"name", "path"}.
    """
    figures: list[dict] = []

    # Paper is typically a gzipped tarball containing .tex + figures.
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tex_files: dict[str, str] = {}
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                if m.name.endswith(".tex"):
                    f = tar.extractfile(m)
                    if f:
                        tex_files[m.name] = f.read().decode("utf-8", errors="replace")
                elif m.name.lower().endswith(_FIGURE_EXTS):
                    f = tar.extractfile(m)
                    if not f:
                        continue
                    img_bytes = f.read()
                    if not img_bytes:
                        continue
                    fig_name = os.path.basename(m.name)
                    if save_figures and out_dir is not None:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        dest = out_dir / fig_name
                        with open(dest, "wb") as fp:
                            fp.write(img_bytes)
                        figures.append({"name": fig_name, "path": str(dest)})
                    else:
                        figures.append({"name": fig_name, "bytes": img_bytes, "size": len(img_bytes)})
            if tex_files:
                return _merge_tex_files(tex_files), figures
    except (tarfile.TarError, OSError, gzip.BadGzipFile):
        pass

    # Single-file gzip fallback (no figures possible).
    try:
        return gzip.decompress(data).decode("utf-8", errors="replace"), []
    except OSError:
        pass

    # Raw text fallback
    try:
        return data.decode("utf-8", errors="replace"), []
    except Exception:
        return None, []


def _extract_tex_blob(data: bytes) -> str | None:
    """Back-compat shim: returns only the LaTeX text (drop figures)."""
    tex, _ = _extract_tex_and_figures(data)
    return tex


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
