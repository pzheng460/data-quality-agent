"""Contamination detection for training data quality.

Provides n-gram overlap detection for benchmark leakage analysis.
Accessible via `dq bench --check-contamination`.
"""

from dq.contamination.ngram import NgramContaminationDetector, load_benchmark
from dq.contamination.report import (
    ContaminationReport,
    ContaminationResult,
    BenchmarkContamination,
)

__all__ = [
    "NgramContaminationDetector",
    "ContaminationReport",
    "ContaminationResult",
    "BenchmarkContamination",
    "load_benchmark",
]
