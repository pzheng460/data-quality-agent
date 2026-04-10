#!/usr/bin/env python3
"""Orchestrate the full arxiv pipeline: ingestion → extraction → curation.

Each stage can also be run independently.

Usage:
    # Full pipeline
    python -m arxiv_pipeline.run \
        --ids 1706.03762,2310.06825,2303.08774 \
        --workdir /data/arxiv_pipeline \
        --config configs/arxiv.yaml

    # Single stage
    python -m arxiv_pipeline.ingestion.fetch_arxiv \
        --ids 2310.06825 --output raw/batch1.jsonl

    python -m arxiv_pipeline.extraction.latex_to_markdown \
        --input raw/batch1.jsonl --output extracted/batch1.jsonl.zst

    python -m arxiv_pipeline.curation.run \
        --input extracted/batch1.jsonl.zst --output cleaned/ --config configs/arxiv.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_full_pipeline(
    ids: list[str],
    workdir: str,
    config: str = "configs/arxiv.yaml",
    workers: int = 1,
    delay: float = 3.0,
):
    """Run all 3 stages sequentially."""
    work = Path(workdir)
    raw_path = work / "raw" / "input.jsonl"
    extracted_path = work / "extracted" / "input.jsonl.zst"
    cleaned_dir = work / "cleaned"

    # ── Stage 1: Ingestion ──
    logger.info("=== Stage 1: Ingestion ===")
    from arxiv_pipeline.ingestion.fetch_arxiv import fetch_by_ids
    count = fetch_by_ids(ids, raw_path, delay=delay)
    logger.info("Downloaded %d papers → %s", count, raw_path)

    # ── Stage 2: Extraction ──
    logger.info("=== Stage 2: Extraction ===")
    from arxiv_pipeline.extraction.latex_to_markdown import process_file
    count = process_file(raw_path, extracted_path)
    logger.info("Extracted %d papers → %s", count, extracted_path)

    # ── Stage 3: Curation ──
    logger.info("=== Stage 3: Curation ===")
    from arxiv_pipeline.curation.run import run_curation
    run_curation(str(extracted_path), str(cleaned_dir), config, workers=workers)
    logger.info("Curation done → %s", cleaned_dir)


def main():
    parser = argparse.ArgumentParser(description="Arxiv pipeline: ingest → extract → curate")
    parser.add_argument("--ids", required=True, help="Comma-separated arxiv IDs")
    parser.add_argument("--workdir", required=True, help="Working directory for all stages")
    parser.add_argument("--config", default="configs/arxiv.yaml")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    workdir = Path(args.workdir)
    ids = [s.strip() for s in args.ids.split(",") if s.strip()]

    # Stage 1: Ingestion
    raw_path = workdir / "raw" / "input.jsonl"
    logger.info("=== Stage 1: Ingestion (%d papers) ===", len(ids))
    from arxiv_pipeline.ingestion.fetch_arxiv import fetch_by_ids
    fetch_by_ids(ids, raw_path, delay=args.delay)

    # Stage 2: Extraction
    extracted_path = workdir / "extracted" / "input.jsonl.zst"
    logger.info("=== Stage 2: Extraction ===")
    from arxiv_pipeline.extraction.latex_to_markdown import process_file
    process_file(raw_path, extracted_path)

    # Stage 3: Curation (dq run)
    cleaned_dir = workdir / "cleaned"
    logger.info("=== Stage 3: Curation ===")
    from arxiv_pipeline.curation.run import run_curation
    run_curation(str(extracted_path), str(cleaned_dir), args.config, args.workers)

    logger.info("=== Done: %s ===", workdir)


if __name__ == "__main__":
    main()
