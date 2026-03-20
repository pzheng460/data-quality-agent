"""DEITA Evol-Complexity scorer for SFT data.

Phase 2 stub.
"""


class ComplexityScorer:
    """Score instruction complexity using the DEITA Evol-Complexity method.

    Evaluates how complex/challenging an instruction is, preferring
    instructions that require deeper reasoning.

    Requires: transformers, a scorer model
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "ComplexityScorer is a Phase 2 feature. "
            "Install optional deps with: uv sync --extra model"
        )
