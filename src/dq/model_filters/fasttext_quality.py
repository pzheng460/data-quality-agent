"""fastText quality classifier (DCLM style).

Phase 2 stub — requires trained fastText model.
"""


class FastTextQualityFilter:
    """Quality classifier using a trained fastText model.

    DCLM-style approach: train a fastText classifier on high-quality
    vs low-quality web text, then use it to score/filter documents.

    Requires: fasttext-wheel, a trained .bin model file
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "FastTextQualityFilter is a Phase 2 feature. "
            "Install optional deps with: uv sync --extra model"
        )
