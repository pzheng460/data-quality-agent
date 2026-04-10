"""Fetch raw HTML from ar5iv (pre-rendered by LaTeXML upstream).

Yields raw HTML — extraction stage converts to clean text.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from typing import Iterator

from dq.ingest.base import IngestSource
from dq.ingest.registry import register_source

logger = logging.getLogger(__name__)

AR5IV_BASE = "https://ar5iv.labs.arxiv.org/html"


@register_source("arxiv_ar5iv")
class Ar5ivSource(IngestSource):
    """Fetch raw HTML from ar5iv. Extraction done in extraction stage."""

    domain = "arxiv"
    priority = 200
    output_format = "html"

    @classmethod
    def params_schema(cls):
        return {
            "ids": {"type": "list", "label": "Arxiv IDs", "required": True},
            "delay": {"type": "float", "label": "Delay (s)", "default": 1.0},
        }

    def __init__(self, ids: list[str], delay: float = 1.0, **kwargs) -> None:
        self.ids = ids
        self.delay = delay

    def fetch(self, limit: int = 0) -> Iterator[dict]:
        from dq.ingest.arxiv_source import _batch_metadata

        meta = _batch_metadata(self.ids)
        count = 0
        for aid in self.ids:
            if limit and count >= limit:
                break
            try:
                html = _fetch_html(aid)
                if not html:
                    continue
                yield {
                    "id": f"arxiv_{aid}",
                    "text": html,  # raw HTML — extraction stage converts to text
                    "source": "ar5iv",
                    "metadata": {
                        "arxiv_id": aid,
                        "title": meta.get(aid, {}).get("title", ""),
                        "abstract": (meta.get(aid, {}).get("abstract", "") or "")[:200],
                        "categories": meta.get(aid, {}).get("categories", []),
                        "primary_category": meta.get(aid, {}).get("primary_category", ""),
                    },
                }
                count += 1
                logger.info("Fetched ar5iv %s", aid)
            except Exception as e:
                logger.warning("ar5iv failed %s: %s", aid, e)
            time.sleep(self.delay)


def _fetch_html(arxiv_id: str) -> str | None:
    url = f"{AR5IV_BASE}/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "dq-pipeline/1.0"})
    try:
        return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to fetch ar5iv HTML for %s: %s", arxiv_id, e)
        return None
