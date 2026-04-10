"""Base class for data ingestion sources."""

from abc import ABC, abstractmethod
from typing import Any, Iterator


class IngestSource(ABC):
    """Base class for a data source that yields documents.

    Subclasses must set `name` and `domain`, and implement `fetch()`.
    Use `@register_source("name")` decorator from `dq.ingest.registry`
    to auto-register.
    """

    name: str = "base"
    domain: str = "generic"
    priority: int = 100  # lower = higher priority within same domain

    @classmethod
    def params_schema(cls) -> dict[str, dict[str, Any]]:
        """Declare accepted parameters for UI rendering and validation.

        Returns dict of {param_name: {type, label, default?, required?}}.
        Example:
            {"ids": {"type": "list", "label": "Paper IDs", "required": True}}
        """
        return {}

    @abstractmethod
    def fetch(self, limit: int = 0) -> "Iterator[dict]":
        """Yield documents in dq standard format.

        Each doc must have at minimum:
            - id: str
            - text: str
            - source: str
            - metadata: dict
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, domain={self.domain!r})"
