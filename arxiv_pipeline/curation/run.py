#!/usr/bin/env python3
"""Stage 3: Quality curation — filter, dedup, contamination check.

Reads extracted markdown JSONL, runs dq pipeline filters,
writes kept/rejected shards.

Input:  extracted/*.jsonl.zst  (text is clean markdown)
Output: cleaned/{kept,rejected}/shard-*.jsonl.zst

This is a thin wrapper around `dq run` that reads from the
extraction output instead of raw input.

Usage:
    python -m arxiv_pipeline.curation.run \
        --input extracted/batch1.jsonl.zst \
        --output cleaned/ \
        --config configs/arxiv.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_curation(input_path: str, output_dir: str, config_path: str,
                 workers: int = 1, resume: bool = True):
    """Run the dq pipeline on extracted data."""
    from dq.runner.engine import PhaseEngine

    # For extracted input, skip phase1 (parse) — data is already clean markdown.
    # Start directly from phase2 (filter).
    engine = PhaseEngine(
        config_path=config_path,
        input_path=input_path,
        output_dir=output_dir,
        workers=workers,
    )
    engine.run_all(resume=resume)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Quality curation (dq pipeline)")
    parser.add_argument("--input", required=True, help="Extracted JSONL.zst")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/arxiv.yaml", help="Pipeline config")
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_curation(args.input, args.output, args.config, args.workers)


if __name__ == "__main__":
    main()
