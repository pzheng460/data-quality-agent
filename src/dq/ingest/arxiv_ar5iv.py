"""Fetch papers from ar5iv HTML (pre-rendered by LaTeXML upstream)."""

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
    """Fetch papers from ar5iv.labs.arxiv.org and convert HTML to markdown."""

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
                md = _html_to_markdown(html)
                if len(md) < 200:
                    logger.warning("Skip %s: too short after conversion", aid)
                    continue
                m = meta.get(aid, {})
                yield {
                    "id": f"arxiv_{aid}",
                    "text": md,
                    "source": "ar5iv",
                    "metadata": {
                        "arxiv_id": aid,
                        "title": m.get("title", ""),
                        "abstract": (m.get("abstract", "") or "")[:200],
                        "categories": m.get("categories", []),
                        "primary_category": m.get("primary_category", ""),
                    },
                }
                count += 1
                logger.info("Fetched ar5iv %s (%d chars)", aid, len(md))
            except Exception as e:
                logger.warning("ar5iv failed %s: %s", aid, e)
            time.sleep(self.delay)


def _fetch_html(arxiv_id: str) -> str | None:
    url = f"{AR5IV_BASE}/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "dq-pipeline/1.0"})
    try:
        return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _html_to_markdown(html: str) -> str:
    """Convert ar5iv HTML to markdown. Try pandoc, fallback to regex."""
    try:
        import pypandoc
        md = pypandoc.convert_text(html, "markdown", format="html",
                                    extra_args=["--wrap=none"])
        return _clean_markdown(md)
    except Exception:
        pass
    # Regex fallback
    import re
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    text = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", lambda m: f"\n{'#' * int(m.group(1))} {m.group(2)}\n", text, flags=re.DOTALL)
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"</p>", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    from html import unescape
    return unescape(text.strip())


def _clean_markdown(md: str) -> str:
    import re
    md = re.sub(r"\{[^}]*reference-type[^}]*\}", "", md)
    md = re.sub(r"\[@[^\]]*\]", "", md)
    md = re.sub(r"^:::.*$", "", md, flags=re.MULTILINE)
    md = re.sub(r"!\[.*?\]\(.*?\)(?:\{[^}]*\})?", "", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


