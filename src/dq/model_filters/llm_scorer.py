"""LLM scoring and distillation (FineWeb-Edu style).

Phase 2 stub — requires an LLM API or local model.
"""


class LLMScorer:
    """Score documents using an LLM (e.g., for educational content quality).

    FineWeb-Edu approach: use a large LLM to score documents on
    educational quality, then distill into a smaller classifier.

    Requires: transformers or API access to an LLM
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "LLMScorer is a Phase 2 feature. "
            "Install optional deps with: uv sync --extra model"
        )
