#!/usr/bin/env python3
"""Stage 1a: Load historical arxiv papers from HuggingFace bulk dataset.

Uses marin-community/ar5iv-no-problem-markdown — ar5iv papers already converted
to markdown with math preserved. Skips LaTeX parsing entirely.

Output: raw/bulk.jsonl.zst  (already markdown, ready for curation)

Usage:
    python -m arxiv_pipeline.ingestion.hf_bulk \
        --output raw/bulk.jsonl.zst \
        --limit 10000

    python -m arxiv_pipeline.ingestion.hf_bulk \
        --output raw/bulk.jsonl.zst \
        --limit 0  # all papers
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_ID = "marin-community/ar5iv-no-problem-markdown"


def _extract_arxiv_id(doc_id: str) -> str:
    """Extract arxiv ID from ar5iv document ID.

    Example: 'no-problem/0001/cond-mat0001234.html' → 'cond-mat/0001234'
              'no-problem/2310/2310.06825.html' → '2310.06825'
    """
    # Remove prefix and .html suffix
    name = doc_id.split("/")[-1].replace(".html", "")
    # New-style ID: 2310.06825
    if re.match(r"\d{4}\.\d{4,5}", name):
        return name
    # Old-style: cond-mat0001234 → cond-mat/0001234
    m = re.match(r"([a-z-]+)(\d+)", name)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return name


def load_bulk(
    output_path: str | Path,
    limit: int = 0,
    categories: list[str] | None = None,
) -> int:
    """Stream ar5iv markdown dataset from HuggingFace and write to JSONL.

    Args:
        output_path: Output JSONL path (.jsonl or .jsonl.zst)
        limit: Max documents (0 = all)
        categories: Filter by arxiv category prefix (e.g. ["cs.", "math."])

    Returns:
        Number of documents written.
    """
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from datasets import load_dataset
    from dq.utils.io import write_jsonl_zst, write_jsonl

    logger.info("Loading %s from HuggingFace (streaming)...", DATASET_ID)
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    def doc_iter():
        count = 0
        for sample in ds:
            arxiv_id = _extract_arxiv_id(sample.get("id", ""))
            text = sample.get("text", "")

            if not text or len(text) < 200:
                continue

            doc = {
                "id": f"arxiv_{arxiv_id}",
                "text": text,
                "source": "ar5iv",
                "source_format": "markdown",
                "extraction_method": "ar5iv_latexml",
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "title": text.split("\n")[0].lstrip("# ").strip() if text.startswith("#") else "",
                    "categories": [],
                    "primary_category": "",
                },
            }

            yield doc
            count += 1
            if count % 10000 == 0:
                logger.info("Processed %d papers...", count)
            if limit > 0 and count >= limit:
                break

    if str(output).endswith(".jsonl.zst"):
        from dq.utils.io import write_jsonl_zst
        count = write_jsonl_zst(doc_iter(), output)
    else:
        from dq.utils.io import write_jsonl
        count = write_jsonl(doc_iter(), output)

    logger.info("Wrote %d docs → %s", count, output)
    return count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load arxiv papers from HF bulk dataset")
    parser.add_argument("--output", required=True, help="Output JSONL(.zst) path")
    parser.add_argument("--limit", type=int, default=0, help="Max papers (0=all)")
    parser.add_argument("--categories", default="", help="Filter categories, e.g. 'cs.,stat.'")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] or None
    n = load_bulk(args.output, limit=args.limit, categories=cats)
    print(f"Done: {n} papers")


if __name__ == "__main__":
    main()
