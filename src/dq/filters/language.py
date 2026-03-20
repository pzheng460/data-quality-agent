"""Language detection filter using fastText lid.176.bin.

Phase 2 stub — requires fasttext model file.
"""

from dq.filters.base import BaseFilter


class LanguageFilter(BaseFilter):
    """Filter documents by detected language using fastText lid.176.bin.

    Requires the fastText language identification model (lid.176.bin)
    and the fasttext-wheel package.
    """

    name = "language"

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "LanguageFilter is a Phase 2 feature. "
            "Install optional deps with: uv sync --extra model "
            "and download lid.176.bin from fastText."
        )

    def filter(self, doc: dict) -> tuple[bool, dict]:
        raise NotImplementedError
