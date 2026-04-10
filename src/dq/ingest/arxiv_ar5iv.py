"""Fetch papers from ar5iv HTML (pre-rendered by LaTeXML upstream).

Uses the same _html_to_text extraction as the local LaTeXML path
to ensure consistent output format.
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
    """Fetch papers from ar5iv.labs.arxiv.org, extract text with same
    pipeline as local LaTeXML (BeautifulSoup + math preservation)."""

    domain = "arxiv"
    priority = 200

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
        from dq.ingest.arxiv_source import _batch_metadata, _html_to_text

        meta = _batch_metadata(self.ids)
        count = 0
        for aid in self.ids:
            if limit and count >= limit:
                break
            try:
                html = _fetch_html(aid)
                if not html:
                    continue
                # Same extraction as local LaTeXML path
                text = _html_to_text(html)
                if len(text) < 200:
                    logger.warning("Skip %s: too short after conversion", aid)
                    continue
                m = meta.get(aid, {})
                title = m.get("title", "")
                yield {
                    "id": f"arxiv_{aid}",
                    "text": f"# {title}\n\n{text}" if title else text,
                    "source": "ar5iv",
                    "metadata": {
                        "arxiv_id": aid,
                        "title": title,
                        "abstract": (m.get("abstract", "") or "")[:200],
                        "categories": m.get("categories", []),
                        "primary_category": m.get("primary_category", ""),
                    },
                }
                count += 1
                logger.info("Fetched ar5iv %s (%d chars)", aid, len(text))
            except Exception as e:
                logger.warning("ar5iv failed %s: %s", aid, e)
            time.sleep(self.delay)


def _fetch_html(arxiv_id: str) -> str | None:
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "dq-pipeline/1.0"})
    try:
        return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("Failed to fetch ar5iv HTML for %s: %s", arxiv_id, e)
        return None
