"""Ingest from local JSONL/Parquet/CSV files."""

from __future__ import annotations

import logging
from typing import Iterator

from dq.stages.ingestion.base import IngestSource
from dq.stages.ingestion.registry import register_source

logger = logging.getLogger(__name__)


@register_source("local_file")
class LocalFileSource(IngestSource):
    """Read documents from a local file (JSONL, Parquet, CSV)."""

    domain = "local"
    priority = 100
    output_format = "text"

    @classmethod
    def params_schema(cls):
        return {
            "path": {"type": "string", "label": "File path", "required": True},
            "text_field": {"type": "string", "label": "Text field name", "default": "text"},
        }

    def __init__(self, path: str, text_field: str = "text", **kwargs) -> None:
        self.path = path
        self.text_field = text_field

    def fetch(self, limit: int = 0) -> Iterator[dict]:
        from dq.utils.io import read_docs

        count = 0
        for doc in read_docs(self.path):
            if self.text_field != "text" and self.text_field in doc:
                doc["text"] = doc[self.text_field]
            if "text" not in doc or not doc["text"]:
                continue
            if "id" not in doc:
                doc["id"] = f"local_{count}"
            doc.setdefault("source", "local")
            doc.setdefault("metadata", {})
            yield doc
            count += 1
            if limit and count >= limit:
                break

        logger.info("Read %d docs from %s", count, self.path)
