"""Contamination detection for training data quality.

Provides three methods:
- N-gram overlap detection (fast, no model needed)
- Min-K% Prob detection (needs model)
- TS-Guessing for MCQ benchmarks (needs API)
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
