"""FineWeb custom quality filters.

Detects list-heavy documents, duplicate-line documents, and bad line breaks.
"""

import re
from collections import Counter
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

_BULLET_RE = re.compile(r"^(\s*[-•*▪▸►]|\s*\d+[.)]\s|\s*[a-zA-Z][.)]\s)")


@register_filter("fineweb")
class FineWebFilter(BaseFilter):
    """FineWeb custom filters.

    - List document detection (>threshold lines start with bullet/number)
    - Duplicate line document (>threshold identical lines)
    - Bad line break detection (short avg line length with many lines)
    """

    def __init__(
        self,
        text_field: str = "text",
        max_list_line_ratio: float = 0.90,
        max_dup_line_ratio: float = 0.30,
        min_avg_line_len: float = 30.0,
        max_short_line_count: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.max_list_line_ratio = max_list_line_ratio
        self.max_dup_line_ratio = max_dup_line_ratio
        self.min_avg_line_len = min_avg_line_len
        self.max_short_line_count = max_short_line_count

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        lines = [l for l in text.split("\n") if l.strip()]

        if not lines:
            return False, {"filter": self.name, "reason": "empty_doc"}

        # List document detection
        bullet_count = sum(1 for l in lines if _BULLET_RE.match(l))
        list_ratio = bullet_count / len(lines)
        if list_ratio > self.max_list_line_ratio:
            return False, {"filter": self.name, "reason": "list_document", "value": list_ratio}

        # Duplicate line detection
        counts = Counter(l.strip() for l in lines)
        dup_count = sum(c for c in counts.values() if c > 1)
        dup_ratio = dup_count / len(lines)
        if dup_ratio > self.max_dup_line_ratio:
            return False, {"filter": self.name, "reason": "dup_line_document", "value": dup_ratio}

        # Bad line break detection
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len < self.min_avg_line_len and len(lines) > self.max_short_line_count:
            return False, {"filter": self.name, "reason": "bad_line_breaks", "value": avg_len}

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        text = self.get_text(doc)
        lines = [l for l in text.split("\n") if l.strip()]
        failures: list[dict] = []

        if not lines:
            failures.append({"filter": self.name, "rule": "empty_doc", "value": 0, "threshold": 1})
            return False, failures

        bullet_count = sum(1 for l in lines if _BULLET_RE.match(l))
        list_ratio = bullet_count / len(lines)
        if list_ratio > self.max_list_line_ratio:
            failures.append({"filter": self.name, "rule": "list_line_ratio", "value": list_ratio, "threshold": self.max_list_line_ratio})

        counts = Counter(l.strip() for l in lines)
        dup_count = sum(c for c in counts.values() if c > 1)
        dup_ratio = dup_count / len(lines)
        if dup_ratio > self.max_dup_line_ratio:
            failures.append({"filter": self.name, "rule": "dup_line_ratio", "value": dup_ratio, "threshold": self.max_dup_line_ratio})

        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len < self.min_avg_line_len and len(lines) > self.max_short_line_count:
            failures.append({"filter": self.name, "rule": "bad_line_breaks", "value": avg_len, "threshold": self.min_avg_line_len})

        return len(failures) == 0, failures
