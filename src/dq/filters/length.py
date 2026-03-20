"""Simple length-based filters."""

from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter
from dq.utils.stats import word_count


@register_filter("length")
class LengthFilter(BaseFilter):
    """Filter documents by word count."""

    def __init__(
        self,
        text_field: str = "text",
        min_words: int = 50,
        max_words: int = 100_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.min_words = min_words
        self.max_words = max_words

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        wc = word_count(text)

        if wc < self.min_words:
            return False, {"filter": self.name, "reason": "too_short", "value": wc}
        if wc > self.max_words:
            return False, {"filter": self.name, "reason": "too_long", "value": wc}

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        text = self.get_text(doc)
        wc = word_count(text)
        failures: list[dict] = []

        if wc < self.min_words:
            failures.append({"filter": self.name, "rule": "min_words", "value": wc, "threshold": self.min_words})
        if wc > self.max_words:
            failures.append({"filter": self.name, "rule": "max_words", "value": wc, "threshold": self.max_words})

        return len(failures) == 0, failures
