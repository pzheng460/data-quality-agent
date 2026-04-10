"""Base class for format extractors."""

from abc import ABC, abstractmethod
from typing import Any


class Extractor(ABC):
    """Converts raw documents (LaTeX, HTML, etc.) to clean text.

    Subclasses register via @register_extractor and declare which
    output_format they handle.
    """

    name: str = "base"
    input_format: str = "text"  # what format this extractor handles

    @abstractmethod
    def extract(self, doc: dict) -> dict | None:
        """Convert a raw doc to clean text.

        Modifies doc["text"] in place (or sets it from raw content).
        Returns the doc, or None to skip it.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
