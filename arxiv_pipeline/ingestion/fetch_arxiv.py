#!/usr/bin/env python3
"""Stage 1: Download raw LaTeX source from arxiv.

Downloads LaTeX .tar.gz from arxiv e-print, merges \\input{} files,
and writes raw LaTeX text to JSONL. No conversion — that's extraction's job.

Output: raw/{batch_name}.jsonl  (one doc per line, raw LaTeX in 'text' field)

Usage:
    python -m arxiv_pipeline.ingestion.fetch_arxiv \
        --ids 2310.06825,2307.09288 \
        --output raw/batch1.jsonl

    python -m arxiv_pipeline.ingestion.fetch_arxiv \
        --from-date 2025-04-01 --to-date 2025-04-07 \
        --categories cs.CL \
        --output raw/april_week1.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import re
import tarfile
import time
import urllib.request
from html import unescape
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

_EPRINT_URL = "https://arxiv.org/e-print"
_API_URL = "http://export.arxiv.org/api/query"
_OAI_URL = "http://export.arxiv.org/oai2"


# ── Download ──

def download_latex(arxiv_id: str) -> str | None:
    """Download LaTeX source, find main .tex, merge \\input{} files."""
    url = f"{_EPRINT_URL}/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "dq-pipeline/1.0"})
    try:
        data = urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        return None

    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tex_files: dict[str, str] = {}
            for m in tar.getmembers():
                if m.name.endswith(".tex"):
                    f = tar.extractfile(m)
                    if f:
                        tex_files[m.name] = f.read().decode("utf-8", errors="replace")
            if not tex_files:
                return None
            # Find main file
            main = None
            for name, content in tex_files.items():
                if r"\begin{document}" in content:
                    main = name
                    break
            if not main:
                main = max(tex_files, key=lambda k: len(tex_files[k]))
            # Merge \input{}
            content = tex_files[main]
            for _ in range(3):
                def _resolve(m: re.Match) -> str:
                    fname = m.group(1).strip()
                    if not fname.endswith(".tex"):
                        fname += ".tex"
                    for path in tex_files:
                        if path == fname or path.endswith("/" + fname):
                            return tex_files[path]
                    return ""
                content = re.sub(r"\\input\{([^}]+)\}", _resolve, content)
            return content
    except tarfile.TarError:
        pass
    try:
        return gzip.decompress(data).decode("utf-8", errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")


import re


def fetch_metadata(arxiv_ids: list[str]) -> dict[str, dict]:
    """Batch fetch metadata from arxiv API."""
    import html as html_mod
    result: dict[str, dict] = {}
    for i in range(0, len(arxiv_ids), 50):
        batch = arxiv_ids[i:i + 50]
        url = f"{_API_URL}?id_list={','.join(batch)}&max_results={len(batch)}"
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml = resp.read().decode()
            for entry in re.finditer(r"<entry>(.*?)</entry>", xml, re.DOTALL):
                block = entry.group(1)
                aid_m = re.search(r"<id>.*?/abs/([^<]+)</id>", block)
                title_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL)
                abs_m = re.search(r"<summary>(.*?)</summary>", block, re.DOTALL)
                cats = re.findall(r'<category[^>]*term="([^"]*)"', block)
                if aid_m:
                    aid = re.sub(r"v\d+$", "", aid_m.group(1).strip())
                    result[aid] = {
                        "arxiv_id": aid,
                        "title": unescape(title_m.group(1).strip()) if (title_m := re.search(r"<title>(.*?)</title>", block, re.DOTALL)) else "",
                        "abstract": unescape(abs_m.group(1).strip()) if abs_m else "",
                        "categories": cats,
                        "primary_category": cats[0] if cats else "",
                        "version": "v1",
                    }
        except Exception as e:
            logger.warning("Metadata fetch failed: %s", e)
        time.sleep(0.5)
    return result


def unescape(s: str) -> str:
    from html import unescape as _u
    return _u(s)


def fetch_by_ids(ids: list[str], output: Path, delay: float = 3.0) -> int:
    """Download papers by arxiv IDs and write raw LaTeX to JSONL."""
    meta = fetch_metadata(ids)
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for aid in ids:
            logger.info("Downloading %s...", aid)
            tex = download_latex(aid)
            if not tex or len(tex) < 200:
                logger.warning("Skip %s: no source", aid)
                continue
            m = meta.get(aid, {})
            doc = {
                "id": f"arxiv_{aid}",
                "text": tex,                    # raw LaTeX — NOT converted
                "source": "arxiv",
                "source_format": "latex",
                "metadata": m or {"arxiv_id": aid},
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
            logger.info("Downloaded %s: %s (%d chars)", aid, m.get("title", "")[:40], len(tex))
            time.sleep(delay)
    logger.info("Ingestion done: %d papers → %s", count, output)
    return count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Download raw arxiv LaTeX")
    parser.add_argument("--ids", help="Comma-separated arxiv IDs")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    fetch_by_ids(ids, Path(args.output), args.delay)


if __name__ == "__main__":
    main()
