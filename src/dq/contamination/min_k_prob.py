"""Min-K% Prob black-box contamination detection.

Phase 2 stub.
"""


class MinKProbDetector:
    """Detect contamination using Min-K% Prob method.

    Black-box approach: uses a language model to compute token
    probabilities and flags documents where the minimum K%
    of token probs are suspiciously high (memorized).

    Requires: transformers, a causal LM
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "MinKProbDetector is a Phase 2 feature. "
            "Install optional deps with: uv sync --extra model"
        )
