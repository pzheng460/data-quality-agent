"""Bulk ingest arxiv papers from HuggingFace ar5iv dataset."""

from __future__ import annotations

import logging
import re
from typing import Iterator

from dq.ingest.base import IngestSource
from dq.ingest.registry import register_source

logger = logging.getLogger(__name__)

DATASET_ID = "marin-community/ar5iv-no-problem-markdown"


@register_source("arxiv_hf_bulk")
class HfBulkSource(IngestSource):
    """Stream ar5iv-converted papers from HuggingFace dataset."""

    domain = "arxiv"
    priority = 100

    @classmethod
    def params_schema(cls):
        return {
            "dataset_id": {"type": "string", "label": "HF dataset ID", "default": DATASET_ID},
            "categories": {"type": "list", "label": "Category filter (e.g. cs., math.)", "required": False},
        }

    def __init__(
        self,
        dataset_id: str = DATASET_ID,
        categories: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.dataset_id = dataset_id
        self.categories = categories

    def fetch(self, limit: int = 0) -> Iterator[dict]:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        from datasets import load_dataset

        logger.info("Streaming %s from HuggingFace...", self.dataset_id)
        ds = load_dataset(self.dataset_id, split="train", streaming=True)

        count = 0
        for sample in ds:
            arxiv_id = _extract_id(sample.get("id", ""))
            text = sample.get("text", "")
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

        logger.info("Done: %d papers from %s", count, self.dataset_id)


def _extract_id(doc_id: str) -> str:
    name = doc_id.split("/")[-1].replace(".html", "")
    if re.match(r"\d{4}\.\d+", name):
        return name
    m = re.match(r"([a-z-]+)(\d+)", name)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return name
