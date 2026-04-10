#!/usr/bin/env python3
"""Stage 1b: Fetch recent papers from ar5iv HTML (incremental updates).

For new papers not yet in the HF bulk dataset, fetches HTML from
ar5iv.labs.arxiv.org and converts to markdown.

Usage:
    # By IDs
    python -m arxiv_pipeline.ingestion.fetch_ar5iv \
        --ids 2310.06825,2307.09288 \
        --output raw/new_papers.jsonl

    # By date range (discovers via OAI-PMH, then fetches from ar5iv)
    python -m arxiv_pipeline.ingestion.fetch_ar5iv \
        --from-date 2025-04-01 --to-date 2025-04-07 \
        --output raw/april_week1.jsonl
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

AR5IV_BASE = "https://ar5iv.labs.arxiv.org/html"


def fetch_ar5iv_html(arxiv_id: str) -> str | None:
    """Fetch HTML from ar5iv for a single paper."""
    url = f"{AR5IV_BASE}/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "dq-pipeline/1.0"})
    try:
        return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", arxiv_id, e)
        return None


def html_to_markdown(html: str) -> str:
    """Convert ar5iv HTML to clean markdown."""
    # Try pandoc first (best quality)
    try:
        import pypandoc
        md = pypandoc.convert_text(html, "markdown", format="html",
                                    extra_args=["--wrap=none"])
        md = _clean_markdown(md)
        return md
    except Exception:
        pass

    # Fallback: basic regex extraction
    import re
    # Remove scripts/styles
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    # Headers
    text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", text, flags=re.DOTALL)
    text = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", text, flags=re.DOTALL)
    # Paragraphs
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"</p>", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    from html import unescape
    return unescape(text.strip())


def _clean_markdown(md: str) -> str:
    """Clean pandoc/ar5iv markdown output."""
    md = re.sub(r"\{[^}]*reference-type[^}]*\}", "", md)
    md = re.sub(r"\[@[^\]]*\]", "", md)
    md = re.sub(r"^:::.*$", "", md, flags=re.MULTILINE)
    md = re.sub(r"!\[.*?\]\(.*?\)(?:\{[^}]*\})?", "", md)  # images
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


_clean_markdown_fn = _clean_markdown


def fetch_papers(arxiv_ids: list[str], output_path: str | Path, delay: float = 3.0) -> int:
    """Fetch papers from ar5iv and write to JSONL."""
    from dq.ingest.arxiv_source import _batch_metadata

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Get metadata
    meta = _batch_metadata(arxiv_ids)

    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for aid in arxiv_ids:
            logger.info("Fetching %s from ar5iv...", aid)
            html = fetch_ar5iv_html(aid)
            if not html:
                continue

            md = html_to_markdown(html)
            if len(md) < 200:
                logger.warning("Skip %s: too short after conversion", aid)
                continue

            m = meta.get(aid, {})
            doc = {
                "id": f"arxiv_{aid}",
                "text": md,
                "source": "ar5iv",
                "source_format": "markdown",
                "extraction_method": "ar5iv_html",
                "metadata": {
                    "arxiv_id": aid,
                    "title": m.get("title", ""),
                    "abstract": m.get("abstract", ""),
                    "categories": m.get("categories", []),
                    "primary_category": m.get("primary_category", ""),
                    "version": m.get("version", "v1"),
                },
            }

            import json
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
            logger.info("Fetched %s: %s (%d chars)", aid, m.get("title", "")[:40], len(md))
            time.sleep(1)  # be nice to ar5iv

    logger.info("Fetched %d papers → %s", count, output)
    return count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch papers from ar5iv (incremental)")
    parser.add_argument("--ids", help="Comma-separated arxiv IDs")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    fetch_papers(ids, Path(args.output), delay=args.delay)


if __name__ == "__main__":
    main()
