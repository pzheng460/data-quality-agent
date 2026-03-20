"""Contamination detection for training data quality.

**STANDALONE ANALYSIS TOOL** — Not part of the main quality pipeline filter chain.
Use via `dq contamination` CLI command for post-hoc analysis of potential benchmark leakage.

Provides three detection methods:
- N-gram overlap detection (fast, no model needed)
- Min-K% Prob detection (needs model)
- TS-Guessing for MCQ benchmarks (needs API)

Note: This module operates independently from the quality filters in dq.filters.
"""

from dq.contamination.ngram import NgramContaminationDetector, load_benchmark
from dq.contamination.min_k_prob import MinKProbDetector
from dq.contamination.ts_guessing import TSGuessingDetector
from dq.contamination.report import (
    ContaminationReport,
    ContaminationResult,
    BenchmarkContamination,
)

__all__ = [
    "NgramContaminationDetector",
    "MinKProbDetector",
    "TSGuessingDetector",
    "ContaminationReport",
    "ContaminationResult",
    "BenchmarkContamination",
    "load_benchmark",
]
