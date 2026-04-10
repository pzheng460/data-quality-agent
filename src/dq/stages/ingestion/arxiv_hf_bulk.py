"""Bulk ingest arxiv papers from HuggingFace ar5iv dataset."""

from __future__ import annotations

import logging
import re
from typing import Iterator

from dq.stages.ingestion.base import IngestSource
from dq.stages.ingestion.registry import register_source

logger = logging.getLogger(__name__)

DATASET_ID = "marin-community/ar5iv-no-problem-markdown"


@register_source("arxiv_hf_bulk")
class HfBulkSource(IngestSource):
    """Stream ar5iv-converted papers from HuggingFace dataset.

    Supports two modes:
    - By IDs: pass `ids=["2310.06825", ...]` to fetch specific papers
    - Bulk: omit `ids` to stream all papers (with optional limit)
    """

    domain = "arxiv"
    priority = 100
    output_format = "text"

    @classmethod
    def params_schema(cls):
        return {
            "ids": {"type": "list", "label": "Arxiv IDs (empty = all)", "required": False},
            "dataset_id": {"type": "string", "label": "HF dataset ID", "default": DATASET_ID},
            "categories": {"type": "list", "label": "Category filter (e.g. cs., math.)", "required": False},
        }

    def __init__(
        self,
        ids: list[str] | None = None,
        dataset_id: str = DATASET_ID,
        categories: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.ids = set(ids) if ids else None
        self.dataset_id = dataset_id
        self.categories = categories

    def fetch(self, limit: int = 0) -> Iterator[dict]:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        from datasets import load_dataset

        if self.ids:
            logger.info("Fetching %d specific papers from %s...", len(self.ids), self.dataset_id)
        else:
            logger.info("Streaming %s from HuggingFace...", self.dataset_id)

        ds = load_dataset(self.dataset_id, split="train", streaming=True)

        remaining = set(self.ids) if self.ids else None
        count = 0

        for sample in ds:
            arxiv_id = _extract_id(sample.get("id", ""))
            text = sample.get("text", "")

            # If filtering by IDs, check match
            if remaining is not None:
                if arxiv_id not in remaining:
                    continue
                remaining.discard(arxiv_id)

            if not text or len(text) < 200:
                continue

            yield {
                "id": f"arxiv_{arxiv_id}",
                "text": text,
                "source": "ar5iv",
                "metadata": {
                    "arxiv_id": arxiv_id,
                    "title": text.split("\n")[0].lstrip("# ").strip() if text.startswith("#") else "",
                    "categories": [],
                    "primary_category": "",
                },
            }
            count += 1
            if count % 10000 == 0:
                logger.info("Streamed %d papers...", count)
            if limit > 0 and count >= limit:
                break
            # If we found all requested IDs, stop early
            if remaining is not None and len(remaining) == 0:
                break

        if remaining:
            logger.warning("IDs not found in dataset: %s", remaining)
        logger.info("Done: %d papers from %s", count, self.dataset_id)


def _extract_id(doc_id: str) -> str:
    """Extract arxiv ID from HF dataset doc ID."""
    name = doc_id.split("/")[-1].replace(".html", "")
    if re.match(r"\d{4}\.\d+", name):
        return name
    m = re.match(r"([a-z-]+)(\d+)", name)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return name
