"""Benchmark module: validate quality filters against real-world datasets.

Two-layer benchmark:
- Layer 1: Rule-based filters (pre-training: Gopher/C4/FineWeb; SFT: empty_output/ai_refusal/etc.) — fast, no API cost
- Layer 2: LLM binary judge (opt-in) — data-type-aware binary classification (HIGH/LOW)
"""

# Re-export all public names for backward compatibility

# Constants
from .datasets import (
    FINEWEB_DATASET,
    FINEWEB_CONFIG,
    ALPACA_DATASET,
    ALPACA_ORIGINAL,
    ALPACA_CLEANED,
)

# Dataset loading functions
from .datasets import (
    load_fineweb_sample,
    load_alpaca_sample,
    load_alpaca_original,
    load_alpaca_cleaned,
    _merge_alpaca_fields,  # Private function used by tests
)

# Utility functions
from .utils import (
    detect_data_type,
    SFT_FIELDS,  # Re-export from utils
    _extract_sft_fields,  # Private function used by tests
)

# Dataclasses and types
from .types import (
    SFTScores,
    PretrainScores,
    RuleStats,
    FilterResult,
    DatasetResult,
    BenchmarkReport,
    BenchmarkResult,  # Backward compat alias
)

# Benchmark runner functions
from .runner import (
    run_benchmark,
    run_llm_scoring,
    _score_docs,  # Private function used by tests
)

__all__ = [
    # Constants
    "FINEWEB_DATASET",
    "FINEWEB_CONFIG",
    "ALPACA_DATASET",
    "ALPACA_ORIGINAL",
    "ALPACA_CLEANED",
    # Dataset functions
    "load_fineweb_sample",
    "load_alpaca_sample",
    "load_alpaca_original",
    "load_alpaca_cleaned",
    "_merge_alpaca_fields",
    # Utility functions
    "detect_data_type",
    "SFT_FIELDS",
    "_extract_sft_fields",
    # Types
    "SFTScores",
    "PretrainScores",
    "RuleStats",
    "FilterResult",
    "DatasetResult",
    "BenchmarkReport",
    "BenchmarkResult",
    # Runner functions
    "run_benchmark",
    "run_llm_scoring",
    "_score_docs",
]