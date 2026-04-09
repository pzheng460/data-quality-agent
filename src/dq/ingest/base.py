"""Base class for data ingestion sources."""

from abc import ABC, abstractmethod
from typing import Iterator


class IngestSource(ABC):
    """Base class for a data source that yields documents."""

    name: str = "base"

    @abstractmethod
    def fetch(self, limit: int = 0) -> Iterator[dict]:
        """Yield documents in dq standard format.

        Each document must have at minimum:
            - id: str
            - text: str
            - metadata: dict with arxiv_id, title, categories, etc.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
