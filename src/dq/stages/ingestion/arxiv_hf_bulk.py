"""Bulk ingest arxiv papers from HuggingFace ar5iv dataset.

Optimization: dataset files are named by YYMM (e.g. 2310.jsonl.gz).
For ID-based lookup, we download only the relevant shard file instead
of streaming the entire dataset.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path
from typing import Iterator

from dq.stages.ingestion.base import IngestSource
from dq.stages.ingestion.registry import register_source

logger = logging.getLogger(__name__)

DATASET_ID = "marin-community/ar5iv-no-problem-markdown"


@register_source("arxiv_hf_bulk")
class HfBulkSource(IngestSource):
    """Stream ar5iv-converted papers from HuggingFace dataset.

    Optimized: when searching by IDs, downloads only the relevant YYMM
    shard files instead of scanning the entire dataset.
    """

    domain = "arxiv"
    priority = 300
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

    def __init__(self, ids=None, from_date=None, to_date=None, categories=None,
                 dataset_id=DATASET_ID, **kwargs):
        self.ids = ids
        self.from_date = from_date
        self.to_date = to_date
        self.categories = set(categories) if categories else None
        self.dataset_id = dataset_id

    def fetch(self, limit: int = 0) -> Iterator[dict]:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        # Resolve IDs from date range if needed
        target_ids = None
        if self.ids:
            target_ids = set(self.ids)
        elif self.from_date:
            from dq.stages.ingestion.arxiv_source import _oai_list_ids
            id_list = _oai_list_ids(self.from_date, self.to_date, self.categories, max_results=limit or 1000)
            logger.info("OAI-PMH returned %d IDs", len(id_list))
            target_ids = set(id_list)

        if not target_ids:
            raise ValueError("Provide either ids or from_date")

        # Fast path: download only relevant shard files
        yield from self._fetch_by_shard(target_ids, limit)

    def _fetch_by_shard(self, target_ids: set[str], limit: int) -> Iterator[dict]:
        """Fast ID lookup: group IDs by YYMM prefix, download shards in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        from huggingface_hub import hf_hub_download

        # Group IDs by YYMM shard
        shards: dict[str, set[str]] = {}
        for aid in target_ids:
            # New-style IDs: 2310.06825 → shard "2310"
            m = re.match(r"(\d{4})\.\d+", aid)
            if m:
                shards.setdefault(m.group(1), set()).add(aid)
            else:
                # Old-style IDs: hep-ph/0001001 → shard "0001" (approximate)
                parts = aid.split("/")
                if len(parts) == 2:
                    shards.setdefault(parts[1][:4], set()).add(aid)
                else:
                    shards.setdefault("unknown", set()).add(aid)

        logger.info("Looking up %d IDs across %d shard(s): %s",
                     len(self.ids) if hasattr(self, 'ids') and self.ids else len(target_ids),
                     len(shards), list(shards.keys()))

        count = 0
        remaining = set(target_ids)

        # Download shards in parallel (HF CDN handles concurrency fine)
        shard_names = [n for n in shards if n != "unknown"]
        if "unknown" in shards:
            logger.warning("Cannot determine shard for IDs: %s", shards["unknown"])

        def _dl(name: str) -> tuple[str, str | None]:
            try:
                p = hf_hub_download(self.dataset_id, f"{name}.jsonl.gz", repo_type="dataset")
                return name, p
            except Exception as e:
                logger.warning("Shard %s.jsonl.gz not found: %s", name, e)
                return name, None

        shard_paths: dict[str, str] = {}
        if shard_names:
            max_workers = min(8, len(shard_names))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for name, path in pool.map(_dl, shard_names):
                    if path:
                        shard_paths[name] = path
            logger.info("Downloaded %d/%d shards (concurrency=%d)",
                        len(shard_paths), len(shard_names), max_workers)

        for shard_name, path in shard_paths.items():
            if not remaining:
                break

            # Scan shard for matching IDs
            import gzip, json
            with gzip.open(path, 'rt') as f:
                for line in f:
                    doc = json.loads(line)
                    arxiv_id = _extract_id(doc.get("id", ""))
                    if arxiv_id in remaining:
                        remaining.discard(arxiv_id)
                        text = doc.get("text", "")
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
                        if limit and count >= limit:
                            return
                        if not remaining:
                            return

        if remaining:
            msg = f"IDs not found in ar5iv dataset (may have conversion issues): {remaining}"
            logger.warning(msg)
            if count == 0:
                raise ValueError(msg)

    def _fetch_stream(self, limit: int) -> Iterator[dict]:
        """Bulk streaming mode — scan entire dataset."""
        from datasets import load_dataset

        logger.info("Streaming all from %s...", self.dataset_id)
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
            if limit and count >= limit:
                break
        logger.info("Done: %d papers", count)


def _extract_id(doc_id: str) -> str:
    name = doc_id.split("/")[-1].replace(".html", "")
    if re.match(r"\d{4}\.\d+", name):
        return name
    m = re.match(r"([a-z-]+)(\d+)", name)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return name
