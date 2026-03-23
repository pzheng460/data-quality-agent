"""Dataclasses for benchmark results and statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dq.pipeline import PipelineStats


@dataclass
class SFTScores:
    """SFT quality scores from LLM Binary Judge."""

    high_count: int = 0
    low_count: int = 0
    high_rate: float = 0.0
    rule_fail_counts: dict[str, int] = field(default_factory=dict)
    num_scored: int = 0
    scoring_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sft",
            "high_count": self.high_count,
            "low_count": self.low_count,
            "high_rate": round(self.high_rate, 3),
            "rule_fail_counts": self.rule_fail_counts,
            "num_scored": self.num_scored,
            "scoring_errors": self.scoring_errors,
        }


@dataclass
class PretrainScores:
    """Pre-training text quality scores from LLM Binary Judge."""

    high_count: int = 0
    low_count: int = 0
    high_rate: float = 0.0
    rule_fail_counts: dict[str, int] = field(default_factory=dict)
    num_scored: int = 0
    scoring_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "pretrain",
            "high_count": self.high_count,
            "low_count": self.low_count,
            "high_rate": round(self.high_rate, 3),
            "rule_fail_counts": self.rule_fail_counts,
            "num_scored": self.num_scored,
            "scoring_errors": self.scoring_errors,
        }


@dataclass
class RuleStats:
    """Per-rule statistics within a filter."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0


@dataclass
class FilterResult:
    """Per-filter results for a single dataset."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    sample_failed: list[dict] = field(default_factory=list)


@dataclass
class DedupStats:
    """Deduplication statistics."""

    exact_duplicates: int = 0
    duplicate_rate: float = 0.0


@dataclass
class DatasetStats:
    """Basic dataset statistics."""

    avg_word_count: float = 0.0
    min_word_count: int = 0
    max_word_count: int = 0
    avg_word_length: float = 0.0
    fields: list[str] = field(default_factory=list)
    dedup: DedupStats | None = None


@dataclass
class DatasetResult:
    """Benchmark results for a single dataset."""

    name: str
    num_docs: int
    stats: PipelineStats | None = None
    per_filter: dict[str, FilterResult] = field(default_factory=dict)
    per_filter_pass_rate: dict[str, float] = field(default_factory=dict)
    overall_pass_rate: float = 0.0
    data_type: str = "pretrain"  # 'sft' or 'pretrain'
    llm_scores: dict[str, Any] | None = None  # SFTScores.to_dict() or PretrainScores.to_dict()
    dataset_stats: DatasetStats | None = None  # Basic word count / field stats


@dataclass
class BenchmarkReport:
    """Full benchmark comparison report."""

    datasets: dict[str, DatasetResult] = field(default_factory=dict)
    config_path: str | None = None
    num_samples: int = 0
    # Per-rule breakdown: {dataset_name: {filter_name: {rule_name: RuleStats}}}
    rule_stats: dict[str, dict[str, dict[str, RuleStats]]] = field(default_factory=dict)
    llm_scoring_enabled: bool = False
    llm_samples: int = 0

    def discrimination_scores(self) -> dict[str, float]:
        """Compute per-filter discrimination: max pass rate - min pass rate."""
        if len(self.datasets) < 2:
            return {}
        all_filters: set[str] = set()
        for dr in self.datasets.values():
            all_filters.update(dr.per_filter_pass_rate.keys())

        scores: dict[str, float] = {}
        for f in sorted(all_filters):
            rates = [dr.per_filter_pass_rate.get(f, 0.0) for dr in self.datasets.values()]
            scores[f] = max(rates) - min(rates)
        return scores


# Backward compat alias
BenchmarkResult = BenchmarkReport