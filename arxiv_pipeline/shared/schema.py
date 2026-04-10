"""Unified document schema for the arxiv pipeline.

Every stage reads and writes documents in this format.
Fields are added progressively — ingestion fills metadata,
extraction fills text, curation fills quality_scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass
class ArxivDocument:
    """Standard document flowing through all pipeline stages."""

    # ── Identity ──
    id: str = ""                        # "arxiv_2310.06825"
    arxiv_id: str = ""                  # "2310.06825"

    # ── Content (filled by extraction) ──
    text: str = ""                      # Clean markdown text
    raw_text: str = ""                  # Original pre-extraction text (for comparison)

    # ── Metadata (filled by ingestion) ──
    title: str = ""
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    primary_category: str = ""
    version: str = ""
    source: str = ""                    # "arxiv_eprint", "redpajama", "dolma", ...
    source_format: str = ""             # "latex", "html", "pdf"

    # ── Extraction metadata ──
    extraction_method: str = ""         # "pandoc", "pylatexenc", "w3m+llm"
    text_length: int = 0

    # ── Quality scores (filled by curation) ──
    quality_scores: dict[str, Any] = field(default_factory=dict)
    rejections: list[dict] = field(default_factory=list)
    is_rejected: bool = False

    # ── Pipeline trace ──
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict for JSONL output."""
        d = asdict(self)
        # Flatten to match dq pipeline expected format
        d["metadata"] = {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "categories": self.categories,
            "primary_category": self.primary_category,
            "version": self.version,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ArxivDocument":
        """Deserialize from dict (JSONL input)."""
        meta = d.get("metadata", {})
        return cls(
            id=d.get("id", ""),
            arxiv_id=meta.get("arxiv_id", "") if (meta := d.get("metadata", {})) else "",
            text=d.get("text", ""),
            raw_text=d.get("raw_text", ""),
            title=meta.get("title", ""),
            source=d.get("source", ""),
            source_format=d.get("source_format", ""),
            categories=meta.get("categories", []),
            primary_category=meta.get("primary_category", ""),
            version=meta.get("version", ""),
            abstract=meta.get("abstract", ""),
            extraction_method=d.get("extraction_method", ""),
            text_length=len(d.get("text", "")),
            quality_scores=d.get("quality_scores", {}),
            rejections=d.get("__dq_rejections", []),
            is_rejected=bool(d.get("__dq_rejections")),
            trace=d.get("trace", {}),
        )

    def to_jsonl_line(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)
