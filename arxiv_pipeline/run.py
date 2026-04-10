#!/usr/bin/env python3
"""Orchestrate the arxiv data pipeline.

Two ingestion modes:
  --bulk    Load from HF dataset (marin-community/ar5iv-no-problem-markdown)
            For historical bulk data. Already markdown, skips extraction.
  --ids     Fetch from ar5iv by arxiv IDs (incremental)
            For new papers. HTML → markdown conversion.

Then runs dq curation (filter → quality score → dedup → contamination → package).

Usage:
    # Bulk: load 10K papers from HF, then curate
    python -m arxiv_pipeline.run --bulk --limit 10000 --workdir /data/arxiv

    # Incremental: fetch specific papers, then curate
    python -m arxiv_pipeline.run --ids 2310.06825,2307.09288 --workdir /data/arxiv

    # Each stage independently
    python -m arxiv_pipeline.ingestion.hf_bulk --output raw/bulk.jsonl.zst --limit 1000
    python -m arxiv_pipeline.ingestion.fetch_ar5iv --ids 2310.06825 --output raw/new.jsonl
    dq run raw/bulk.jsonl.zst -o cleaned/ -c configs/arxiv.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Arxiv pipeline: ingest → curate")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bulk", action="store_true", help="Load from HF bulk dataset")
    group.add_argument("--ids", help="Comma-separated arxiv IDs to fetch from ar5iv")
    parser.add_argument("--workdir", required=True, help="Working directory")
    parser.add_argument("--config", default="configs/arxiv.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Max papers for --bulk (0=all)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    work = Path(args.workdir)

    if args.bulk:
        # ── Bulk: HF dataset → curate ──
        raw_path = work / "raw" / "bulk.jsonl.zst"
        logger.info("=== Stage 1: Bulk ingestion from HuggingFace ===")
        from arxiv_pipeline.ingestion.hf_bulk import load_bulk
        count = load_bulk(raw_path, limit=args.limit)
        logger.info("Loaded %d papers", count)

        logger.info("=== Stage 2: Skipped (ar5iv data is already markdown) ===")

        logger.info("=== Stage 3: Curation ===")
        from arxiv_pipeline.curation.run import run_curation
        run_curation(str(raw_path), str(work / "cleaned"), args.config, workers=args.workers)

    else:
        # ── Incremental: ar5iv HTML → curate ──
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
        raw_path = work / "raw" / "incremental.jsonl"

        logger.info("=== Stage 1: Fetch %d papers from ar5iv ===", len(ids))
        from arxiv_pipeline.ingestion.fetch_ar5iv import fetch_papers
        count = fetch_papers(ids, raw_path)
        logger.info("Fetched %d papers", count)

        logger.info("=== Stage 2: Skipped (ar5iv HTML already converted) ===")

        logger.info("=== Stage 3: Curation ===")
        from arxiv_pipeline.curation.run import run_curation
        run_curation(str(raw_path), str(work / "cleaned"), args.config, workers=args.workers)

    logger.info("=== Pipeline complete: %s ===", work)


if __name__ == "__main__":
    main()
