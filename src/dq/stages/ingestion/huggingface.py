"""Generic HuggingFace dataset source.

Streams any HF dataset (FineWeb, C4, RedPajama, etc.) via `datasets.load_dataset`.
Great for quickly benchmarking filters on public web-text corpora.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

from dq.stages.ingestion.base import IngestSource
from dq.stages.ingestion.registry import register_source

logger = logging.getLogger(__name__)


@register_source("huggingface")
class HuggingFaceSource(IngestSource):
    """Stream any HuggingFace dataset by id / subset / split.

    Example params:
      dataset:  "HuggingFaceFW/fineweb"
      subset:   "sample-10BT"     (optional — dataset-specific)
      split:    "train"
      text_field: "text"          (column containing the doc text)
    """

    name = "huggingface"
    domain = "general"
    priority = 100
    output_format = "text"

    @classmethod
    def params_schema(cls):
        return {
            "dataset": {"type": "string", "label": "HF dataset id", "required": True},
            "subset": {"type": "string", "label": "Config/subset", "required": False},
            "split": {"type": "string", "label": "Split", "default": "train"},
            "text_field": {"type": "string", "label": "Text column", "default": "text"},
        }

    def __init__(self, dataset: str, subset: str | None = None,
                 split: str = "train", text_field: str = "text", **_kwargs):
        if not dataset:
            raise ValueError("`dataset` is required for huggingface source")
        self.dataset = dataset
        self.subset = subset or None
        self.split = split
        self.text_field = text_field

    @classmethod
    def params_schema_description(cls) -> str:
        return (
            "Generic HuggingFace dataset source. For FineWeb use "
            "dataset=HuggingFaceFW/fineweb subset=sample-10BT."
        )

    def fetch(self, limit: int = 0) -> Iterator[dict]:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset

        logger.info("Streaming HF dataset %s [%s] split=%s",
                    self.dataset, self.subset or "default", self.split)
        load_kwargs = {"split": self.split, "streaming": True}
        if self.subset:
            ds = load_dataset(self.dataset, self.subset, **load_kwargs)
        else:
            ds = load_dataset(self.dataset, **load_kwargs)

        for i, row in enumerate(ds):
            if limit and i >= limit:
                break
            text = row.get(self.text_field, "")
            if not text:
                continue
            doc_id = str(row.get("id") or row.get("url") or f"hf_{i}")
            yield {
                "id": f"hf_{i}",
                "text": text,
                "source": self.dataset,
                "metadata": {
                    "dataset": self.dataset,
                    "subset": self.subset,
                    "split": self.split,
                    "original_id": doc_id,
                    **{k: v for k, v in row.items()
                       if k not in ("text", "id", "url") and isinstance(v, (str, int, float, bool))},
                },
            }

        logger.info("Streamed from HF dataset %s", self.dataset)
