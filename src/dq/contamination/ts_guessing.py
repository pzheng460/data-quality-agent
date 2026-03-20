"""TS-Guessing for MCQ benchmark contamination detection.

Phase 2 stub.
"""


class TSGuessingDetector:
    """Detect MCQ benchmark contamination using TS-Guessing.

    Tests whether a model can guess the correct answer option
    for multiple-choice questions at above-chance rates,
    indicating it has seen the benchmark during training.

    Requires: transformers or API access to a model
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "TSGuessingDetector is a Phase 2 feature. "
            "Install optional deps with: uv sync --extra model"
        )
