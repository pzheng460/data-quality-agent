"""Data quality filters."""

# Import all filter modules to trigger registration
from dq.filters.c4 import C4Filter
from dq.filters.fineweb import FineWebFilter
from dq.filters.gopher import GopherQualityFilter, GopherRepetitionFilter
from dq.filters.length import LengthFilter
from dq.filters.pii import PIIFilter
from dq.filters.sft_rules import SFTRulesFilter

__all__ = [
    "C4Filter",
    "FineWebFilter",
    "GopherQualityFilter",
    "GopherRepetitionFilter",
    "LengthFilter",
    "PIIFilter",
    "SFTRulesFilter",
]
