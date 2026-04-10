"""Format extractors (LaTeX→text, HTML→text, etc.).

Call ensure_extractors_registered() to auto-import all extractor modules.
"""

import importlib
import pkgutil
from pathlib import Path

_registered = False


def ensure_extractors_registered() -> None:
    """Auto-import all extractor modules to trigger @register_extractor."""
    global _registered
    if _registered:
        return
    _registered = True

    pkg_dir = str(Path(__file__).parent)
    for _finder, name, _ispkg in pkgutil.iter_modules([pkg_dir]):
        if name not in ("base", "registry"):
            importlib.import_module(f"{__package__}.{name}")


from dq.stages.extraction.base import Extractor
from dq.stages.extraction.registry import register_extractor, get_extractor_class, get_extractor_for_format, list_extractors

__all__ = [
    "Extractor",
    "register_extractor",
    "get_extractor_class",
    "get_extractor_for_format",
    "list_extractors",
    "ensure_extractors_registered",
]
