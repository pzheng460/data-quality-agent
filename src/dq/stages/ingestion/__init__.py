"""Data ingestion sources.

Call ensure_sources_registered() to auto-import all source modules and
trigger @register_source decorators. Mirrors the filter pattern.
"""

import importlib
import pkgutil
from pathlib import Path

_registered = False


def ensure_sources_registered() -> None:
    """Auto-import all source modules in this package to trigger registration."""
    global _registered
    if _registered:
        return
    _registered = True

    pkg_dir = str(Path(__file__).parent)
    for _finder, name, _ispkg in pkgutil.iter_modules([pkg_dir]):
        if name not in ("base", "registry"):
            importlib.import_module(f"{__package__}.{name}")


from dq.stages.ingestion.base import IngestSource
from dq.stages.ingestion.registry import get_source_class, list_sources, register_source

__all__ = [
    "IngestSource",
    "register_source",
    "get_source_class",
    "list_sources",
    "ensure_sources_registered",
]
