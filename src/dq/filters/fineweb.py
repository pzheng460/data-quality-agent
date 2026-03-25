"""FineWeb quality filters.

Aligned with datatrove's FineWebQualityFilter reference implementation.
Checks: line_punct_ratio, short_line_ratio, char_dup_ratio, list_ratio (newline/word).
"""

from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter
from dq.utils.stats import get_words

# Stop chars for terminal punctuation — loaded lazily from datatrove
_STOP_CHARS: tuple[str, ...] | None = None


def _get_stop_chars() -> tuple[str, ...]:
    global _STOP_CHARS
    if _STOP_CHARS is None:
        from datatrove.pipeline.filters.fineweb_quality_filter import FineWebQualityFilter as _DT
        _STOP_CHARS = _DT().stop_chars
    return _STOP_CHARS


def _find_duplicate_chars(lines: list[str]) -> int:
    """Count total characters in duplicate lines (matching datatrove's find_duplicates)."""
    seen: set[str] = set()
    dup_chars = 0
    for line in lines:
        if line in seen:
            dup_chars += len(line)
        else:
            seen.add(line)
    return dup_chars


@register_filter("fineweb")
class FineWebFilter(BaseFilter):
    """FineWeb quality filters (aligned with datatrove).

    - Line punctuation ratio: fraction of lines ending with terminal punct
    - Short line ratio: fraction of lines ≤ short_line_length chars
    - Character duplicate ratio: duplicate line char coverage
    - List ratio: newline count / word count (list-like detection)
    """

    def __init__(
        self,
        text_field: str = "text",
        line_punct_thr: float = 0.12,
        line_punct_exclude_zero: bool = False,
        short_line_thr: float = 0.67,
        short_line_length: int = 30,
        char_duplicates_ratio: float = 0.01,
        new_line_ratio: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.line_punct_thr = line_punct_thr
        self.line_punct_exclude_zero = line_punct_exclude_zero
        self.short_line_thr = short_line_thr
        self.short_line_length = short_line_length
        self.char_duplicates_ratio = char_duplicates_ratio
        self.new_line_ratio = new_line_ratio

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        lines = [line for line in text.split("\n") if line.strip() != ""]

        if not lines:
            return False, {"filter": self.name, "reason": "empty_doc"}

        stop_chars = _get_stop_chars()

        # Line punctuation ratio
        punct_ratio = sum(1 for line in lines if line.endswith(stop_chars)) / len(lines)
        if punct_ratio < self.line_punct_thr and not (punct_ratio == 0 and self.line_punct_exclude_zero):
            return False, {"filter": self.name, "reason": "line_punct_ratio", "value": punct_ratio}

        # Short line ratio
        short_ratio = sum(1 for line in lines if len(line) <= self.short_line_length) / len(lines)
        if short_ratio > self.short_line_thr:
            return False, {"filter": self.name, "reason": "short_line_ratio", "value": short_ratio}

        # Character duplicate ratio
        text_no_newlines = text.replace("\n", "")
        if text_no_newlines:
            dup_chars = _find_duplicate_chars(lines)
            char_dup_ratio = dup_chars / len(text_no_newlines)
            if char_dup_ratio > self.char_duplicates_ratio:
                return False, {"filter": self.name, "reason": "char_dup_ratio", "value": char_dup_ratio}

        # List ratio (newlines / words)
        words = get_words(text)
        if words:
            newline_count = text.count("\n")
            list_ratio = newline_count / len(words)
            if list_ratio > self.new_line_ratio:
                return False, {"filter": self.name, "reason": "list_ratio", "value": list_ratio}

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        text = self.get_text(doc)
        lines = [line for line in text.split("\n") if line.strip() != ""]
        failures: list[dict] = []

        if not lines:
            failures.append({"filter": self.name, "rule": "empty_doc", "value": 0, "threshold": 1})
            return False, failures

        stop_chars = _get_stop_chars()

        punct_ratio = sum(1 for line in lines if line.endswith(stop_chars)) / len(lines)
        if punct_ratio < self.line_punct_thr and not (punct_ratio == 0 and self.line_punct_exclude_zero):
            failures.append({"filter": self.name, "rule": "line_punct_ratio", "value": punct_ratio, "threshold": self.line_punct_thr})

        short_ratio = sum(1 for line in lines if len(line) <= self.short_line_length) / len(lines)
        if short_ratio > self.short_line_thr:
            failures.append({"filter": self.name, "rule": "short_line_ratio", "value": short_ratio, "threshold": self.short_line_thr})

        text_no_newlines = text.replace("\n", "")
        if text_no_newlines:
            dup_chars = _find_duplicate_chars(lines)
            char_dup_ratio = dup_chars / len(text_no_newlines)
            if char_dup_ratio > self.char_duplicates_ratio:
                failures.append({"filter": self.name, "rule": "char_dup_ratio", "value": char_dup_ratio, "threshold": self.char_duplicates_ratio})

        words = get_words(text)
        if words:
            newline_count = text.count("\n")
            list_ratio = newline_count / len(words)
            if list_ratio > self.new_line_ratio:
                failures.append({"filter": self.name, "rule": "list_ratio", "value": list_ratio, "threshold": self.new_line_ratio})

        return len(failures) == 0, failures
