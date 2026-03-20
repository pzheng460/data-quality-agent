"""Model-based quality filters (Phase 2).

These filters require optional dependencies (transformers, fasttext, torch).
They gracefully skip if dependencies are missing.
Import triggers filter registration via @register_filter decorators.
"""

from dq.model_filters.fasttext_quality import FastTextQualityFilter
from dq.model_filters.perplexity import PerplexityFilter

__all__ = [
    "FastTextQualityFilter",
    "PerplexityFilter",
]
