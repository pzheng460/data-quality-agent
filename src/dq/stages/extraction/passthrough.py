"""Passthrough extractor for sources that already yield clean text."""

from dq.stages.extraction.base import Extractor
from dq.stages.extraction.registry import register_extractor


@register_extractor("passthrough")
class PassthroughExtractor(Extractor):
    """No-op extractor for data that is already plain text."""

    input_format = "text"

    def extract(self, doc: dict) -> dict:
        return doc
