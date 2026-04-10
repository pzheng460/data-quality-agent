"""Bulk ingest arxiv papers from HuggingFace ar5iv dataset.

Supports:
- By IDs: stream dataset, filter for specific papers
- By date: discover IDs via OAI-PMH, then filter from dataset
- Bulk: stream all papers (with optional limit)
"""

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
    """Stream ar5iv-converted papers from HuggingFace dataset."""

    domain = "arxiv"
    priority = 100
    output_format = "text"

    @classmethod
    def params_schema(cls):
        return {
            "ids": {"type": "list", "label": "Arxiv IDs", "required": False},
            "from_date": {"type": "string", "label": "From date", "required": False},
            "to_date": {"type": "string", "label": "To date", "required": False},
            "categories": {"type": "list", "label": "Categories", "required": False},
            "dataset_id": {"type": "string", "label": "HF dataset ID", "default": DATASET_ID},
        }

    def __init__(
        self,
        ids: list[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        categories: list[str] | None = None,
        dataset_id: str = DATASET_ID,
        **kwargs,
    ) -> None:
        self.ids = ids
        self.from_date = from_date
        self.to_date = to_date
        self.categories = set(categories) if categories else None
        self.dataset_id = dataset_id

    def fetch(self, limit: int = 0) -> Iterator[dict]:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset

        # Resolve IDs from date range if needed
        target_ids = None
        if self.ids:
            target_ids = set(self.ids)
        elif self.from_date:
            from dq.stages.ingestion.arxiv_source import _oai_list_ids
            id_list = _oai_list_ids(self.from_date, self.to_date, self.categories, max_results=limit or 1000)
            logger.info("OAI-PMH returned %d IDs for date range", len(id_list))
            target_ids = set(id_list)

        if target_ids is not None:
            logger.info("Filtering HF dataset for %d specific IDs...", len(target_ids))
        else:
            logger.info("Streaming all from %s...", self.dataset_id)

        ds = load_dataset(self.dataset_id, split="train", streaming=True)
        remaining = set(target_ids) if target_ids is not None else None
        count = 0

        for sample in ds:
            arxiv_id = _extract_id(sample.get("id", ""))
            text = sample.get("text", "")

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
