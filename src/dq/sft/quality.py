"""DEITA Evol-Quality scorer for SFT data.

Phase 2 stub.
"""


class QualityScorer:
    """Score response quality using the DEITA Evol-Quality method.

    Evaluates the quality of model responses in instruction-following data.

    Requires: transformers, a scorer model
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "QualityScorer is a Phase 2 feature. "
            "Install optional deps with: uv sync --extra model"
        )
