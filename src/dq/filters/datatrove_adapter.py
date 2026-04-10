"""Thin adapters wrapping datatrove filters into the dq BaseFilter interface.

Usage in YAML configs:
    - name: dt_gopher_quality     # use datatrove's Gopher filter
      params:
        min_doc_words: 200

All datatrove constructor kwargs are passed through directly.
Filter names are prefixed with "dt_" to distinguish from our own implementations.
"""

from __future__ import annotations

import logging
from typing import Any

from datatrove.data import Document

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

logger = logging.getLogger(__name__)


class _DatatroveAdapter(BaseFilter):
    """Base adapter: wraps any datatrove filter, adds filter_detailed()."""

    _dt_class: type  # subclasses set this

    def __init__(self, text_field: str = "text", **kwargs: Any) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self._dt = self._dt_class(**kwargs)

    def _make_doc(self, doc: dict) -> Document:
        return Document(text=doc.get(self.text_field, ""), id=doc.get("id", ""))

    def filter(self, doc: dict) -> tuple[bool, dict]:
        result = self._dt.filter(self._make_doc(doc))
        if result is True:
            return True, {}
        reason = result[1] if isinstance(result, tuple) else str(result)
        return False, {"filter": self.name, "rule": reason}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        keep, info = self.filter(doc)
        return keep, [info] if info else []


# ── Concrete filters ──

@register_filter("dt_gopher_quality")
class DTGopherQuality(_DatatroveAdapter):
    @property
    def _dt_class(self):
        from datatrove.pipeline.filters.gopher_quality_filter import GopherQualityFilter
        return GopherQualityFilter


@register_filter("dt_gopher_repetition")
class DTGopherRepetition(_DatatroveAdapter):
    @property
    def _dt_class(self):
        from datatrove.pipeline.filters.gopher_repetition_filter import GopherRepetitionFilter
        return GopherRepetitionFilter


@register_filter("dt_c4")
class DTC4Filter(_DatatroveAdapter):
    @property
    def _dt_class(self):
        from datatrove.pipeline.filters.c4_filters import C4QualityFilter
        return C4QualityFilter


@register_filter("dt_fineweb")
class DTFineWeb(_DatatroveAdapter):
    @property
    def _dt_class(self):
        from datatrove.pipeline.filters.fineweb_quality_filter import FineWebQualityFilter
        return FineWebQualityFilter


@register_filter("dt_language")
class DTLanguage(_DatatroveAdapter):
    @property
    def _dt_class(self):
        from datatrove.pipeline.filters.language_filter import LanguageFilter
        return LanguageFilter


@register_filter("dt_badwords")
class DTBadWords(_DatatroveAdapter):
    @property
    def _dt_class(self):
        from datatrove.pipeline.filters.c4_filters import C4BadWordsFilter
        return C4BadWordsFilter
