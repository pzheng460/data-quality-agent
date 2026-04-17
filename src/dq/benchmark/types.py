"""Dataclasses for benchmark results and statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dq.pipeline import PipelineStats


@dataclass
class _ScoresBase:
    """Shared aggregation for LLM judge results (binary + score rules)."""

    high_count: int = 0
    low_count: int = 0
    high_rate: float = 0.0
    rule_fail_counts: dict[str, int] = field(default_factory=dict)
    rule_modes: dict[str, str] = field(default_factory=dict)
    rule_score_sums: dict[str, float] = field(default_factory=dict)
    rule_score_counts: dict[str, int] = field(default_factory=dict)
    rule_thresholds: dict[str, float] = field(default_factory=dict)
    rule_max_scores: dict[str, float] = field(default_factory=dict)
    num_scored: int = 0
    scoring_errors: int = 0

    def _common_dict(self) -> dict[str, Any]:
        rule_score_avg: dict[str, float] = {}
        for name, total in self.rule_score_sums.items():
            cnt = self.rule_score_counts.get(name, 0)
            if cnt:
                rule_score_avg[name] = round(total / cnt, 3)
        return {
            "high_count": self.high_count,
            "low_count": self.low_count,
            "high_rate": round(self.high_rate, 3),
            "rule_fail_counts": self.rule_fail_counts,
            "rule_modes": self.rule_modes,
            "rule_score_avg": rule_score_avg,
            "rule_thresholds": self.rule_thresholds,
            "rule_max_scores": self.rule_max_scores,
            "num_scored": self.num_scored,
            "scoring_errors": self.scoring_errors,
        }


@dataclass
class SFTScores(_ScoresBase):
    def to_dict(self) -> dict[str, Any]:
        return {"type": "sft", **self._common_dict()}


@dataclass
class PretrainScores(_ScoresBase):
    def to_dict(self) -> dict[str, Any]:
        return {"type": "pretrain", **self._common_dict()}


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
    minhash_duplicates: int = 0
    minhash_rate: float = 0.0


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
    # All rejected docs per dataset: {dataset_name: [doc_with_rejections, ...]}
    rejected_docs: dict[str, list[dict]] = field(default_factory=dict)

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