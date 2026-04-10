"""Extractor registry — mirrors the filter and source registry patterns."""

from __future__ import annotations

from typing import Any

from dq.stages.extraction.base import Extractor

_EXTRACTOR_REGISTRY: dict[str, type[Extractor]] = {}


def register_extractor(name: str):
    """Decorator to register an extractor class."""
    def wrapper(cls: type[Extractor]):
        cls.name = name
        _EXTRACTOR_REGISTRY[name] = cls
        return cls
    return wrapper


def get_extractor_class(name: str) -> type[Extractor]:
    if name not in _EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Unknown extractor: {name!r}. Available: {list(_EXTRACTOR_REGISTRY.keys())}"
        )
    return _EXTRACTOR_REGISTRY[name]


def get_extractor_for_format(fmt: str) -> type[Extractor]:
    """Find an extractor that handles a given input format."""
    for cls in _EXTRACTOR_REGISTRY.values():
        if cls.input_format == fmt:
            return cls
    raise ValueError(f"No extractor for format: {fmt!r}")


def list_extractors() -> list[dict[str, Any]]:
    return [
        {"name": name, "input_format": cls.input_format}
        for name, cls in _EXTRACTOR_REGISTRY.items()
    ]
