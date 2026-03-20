"""KL divergence filter (Llama 3 style).

Phase 2 stub — requires tokenizer and reference distribution.
"""

from dq.filters.base import BaseFilter


class KLDivFilter(BaseFilter):
    """Filter documents by KL divergence from a reference distribution.

    Compares the token distribution of a document against a reference
    high-quality corpus distribution, filtering outliers.
    """

    name = "kl_div"

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "KLDivFilter is a Phase 2 feature. "
            "Requires a reference token distribution."
        )

    def filter(self, doc: dict) -> tuple[bool, dict]:
        raise NotImplementedError
