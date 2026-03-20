"""Data quality filters.

Call ensure_registered() to auto-import all filter modules and trigger
@register_filter decorators. This replaces manual imports — new filters
just need @register_filter in their .py file, no __init__.py edit needed.
"""

import importlib
import pkgutil
from pathlib import Path

_registered = False


def ensure_registered() -> None:
    """Auto-import all filter modules in this package to trigger registration.

    Safe to call multiple times — only runs once.
    """
    global _registered
    if _registered:
        return
    _registered = True

    pkg_dir = Path(__file__).parent
    for _finder, name, _ispkg in pkgutil.iter_modules([str(pkg_dir)]):
        if name != "base":
            importlib.import_module(f"{__package__}.{name}")


# Re-export base class (no circular import — base.py doesn't import pipeline)
from dq.filters.base import BaseFilter

__all__ = [
    "BaseFilter",
    "ensure_registered",
]
