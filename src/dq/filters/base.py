"""Base filter abstract class for data quality pipeline."""

from abc import ABC, abstractmethod
from typing import Any


class BaseFilter(ABC):
    """Abstract base class for all data quality filters.

    Each filter receives a document dict and returns a tuple of
    (keep: bool, info: dict) where info contains filter-specific metadata
    such as the reason for dropping.
    """

    name: str = "base"

    def __init__(self, text_field: str = "text", **kwargs: Any) -> None:
        self.text_field = text_field
        self.params = kwargs

    def get_text(self, doc: dict) -> str:
        """Extract text from document using configured field name."""
        return doc.get(self.text_field, "")

    @abstractmethod
    def filter(self, doc: dict) -> tuple[bool, dict]:
        """Filter a single document.

        Args:
            doc: Document dict, must contain the text field.

        Returns:
            Tuple of (keep, info) where keep is True to keep the doc,
            and info is a dict with metadata (e.g. reason for dropping).
        """
        ...

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        """Check all rules and return all failures, not just the first.

        Used by benchmark mode to collect per-rule statistics.
        Default implementation wraps filter() for backward compat.

        Returns:
            Tuple of (all_passed, [list of failure dicts]).
            Each failure dict has: filter, rule, value, threshold.
        """
        keep, info = self.filter(doc)
        return keep, [info] if info else []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
