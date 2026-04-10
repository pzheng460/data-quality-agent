"""Source registry — mirrors the filter registry pattern in pipeline.py."""

from __future__ import annotations

from typing import Any

from dq.ingest.base import IngestSource

_SOURCE_REGISTRY: dict[str, type[IngestSource]] = {}


def register_source(name: str):
    """Decorator to register a source class under a given name."""
    def wrapper(cls: type[IngestSource]):
        _SOURCE_REGISTRY[name] = cls
        cls.name = name
        return cls
    return wrapper


def get_source_class(name: str) -> type[IngestSource]:
    """Look up a source class by name."""
    if name not in _SOURCE_REGISTRY:
        raise ValueError(
            f"Unknown source: {name!r}. Available: {list(_SOURCE_REGISTRY.keys())}"
        )
    return _SOURCE_REGISTRY[name]


def list_sources() -> dict[str, list[dict[str, Any]]]:
    """Return sources grouped by domain, sorted by priority.

    Returns:
        {"arxiv": [{"name": "arxiv_hf_bulk", "priority": 100, "params": {...}}, ...], ...}
    """
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for name, cls in _SOURCE_REGISTRY.items():
        entry = {
            "name": name,
            "domain": cls.domain,
            "priority": cls.priority,
            "params": cls.params_schema(),
        }
        by_domain.setdefault(cls.domain, []).append(entry)

    # Sort each domain's sources by priority
    for domain in by_domain:
        by_domain[domain].sort(key=lambda x: x["priority"])

    return by_domain
