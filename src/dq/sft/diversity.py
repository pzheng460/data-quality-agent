"""DEITA Repr Filter — embedding-based diversity selection for SFT data.

Phase 2 stub.
"""


class DiversityFilter:
    """Select diverse training samples using embedding clustering.

    DEITA Repr Filter approach: embed all samples, cluster them,
    and select representative samples to maximize diversity.

    Requires: sentence-transformers
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "DiversityFilter is a Phase 2 feature. "
            "Install optional deps with: uv sync --extra model"
        )
