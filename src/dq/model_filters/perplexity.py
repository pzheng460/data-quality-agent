"""Perplexity filter using a small language model.

Phase 2 stub — requires transformers and a small LM.
"""


class PerplexityFilter:
    """Filter documents by perplexity score from a small language model.

    High perplexity indicates unusual/low-quality text.

    Requires: transformers, a small causal LM (e.g., GPT-2)
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "PerplexityFilter is a Phase 2 feature. "
            "Install optional deps with: uv sync --extra model"
        )
