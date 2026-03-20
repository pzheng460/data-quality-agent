"""Benchmark module: validate quality filters against real-world datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dq.pipeline import PipelineStats

logger = logging.getLogger(__name__)

FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_CONFIG = "sample-10BT"
ALPACA_DATASET = "tatsu-lab/alpaca"
ALPACA_ORIGINAL = "tatsu-lab/stanford_alpaca"
ALPACA_CLEANED = "yahma/alpaca-cleaned"


def _ensure_datasets():
    """Import datasets library or raise helpful error."""
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` library is required for benchmarks. "
            "Install with: uv pip install 'dq[bench]'"
        )


def _merge_alpaca_fields(item: dict) -> str:
    """Concatenate instruction + input + output into a single text field."""
    parts = [
        item.get("instruction", "") or "",
        item.get("input", "") or "",
        item.get("output", "") or "",
    ]
    return "\n".join(p for p in parts if p.strip())


@dataclass
class FilterResult:
    """Per-filter results for a single dataset."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    sample_failed: list[dict] = field(default_factory=list)


@dataclass
class DatasetResult:
    """Benchmark results for a single dataset."""

    name: str
    num_docs: int
    stats: PipelineStats
    per_filter: dict[str, FilterResult] = field(default_factory=dict)
    per_filter_pass_rate: dict[str, float] = field(default_factory=dict)
    overall_pass_rate: float = 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark comparison report."""

    datasets: dict[str, DatasetResult] = field(default_factory=dict)
    config_path: str | None = None
    num_samples: int = 0

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


def load_fineweb_sample(n: int = 1000, seed: int = 42) -> list[dict]:
    """Load n random samples from FineWeb sample-10BT (high-quality pre-training data)."""
    load_dataset = _ensure_datasets()

    logger.info("Loading FineWeb sample (%d docs)...", n)
    try:
        ds = load_dataset(
            FINEWEB_DATASET,
            name=FINEWEB_CONFIG,
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    except Exception as e:
        raise RuntimeError(f"Failed to load FineWeb dataset: {e}") from e

    samples: list[dict] = []
    for item in ds:
        samples.append({"text": item["text"]})
        if len(samples) >= n:
            break

    logger.info("Loaded %d FineWeb samples.", len(samples))
    return samples


def load_alpaca_sample(n: int = 1000, seed: int = 42) -> list[dict]:
    """Load n random samples from Stanford Alpaca 52K (mediocre SFT data).

    Concatenates instruction + input + output into a single 'text' field.
    """
    load_dataset = _ensure_datasets()

    logger.info("Loading Alpaca sample (%d docs)...", n)
    try:
        ds = load_dataset(ALPACA_DATASET, split="train")
        ds = ds.shuffle(seed=seed)
    except Exception as e:
        raise RuntimeError(f"Failed to load Alpaca dataset: {e}") from e

    samples: list[dict] = []
    for item in ds:
        text = _merge_alpaca_fields(item)
        samples.append({"text": text})
        if len(samples) >= n:
            break

    logger.info("Loaded %d Alpaca samples.", len(samples))
    return samples


def load_alpaca_original(n: int | None = None) -> list[dict]:
    """Load samples from tatsu-lab/stanford_alpaca (original, known quality issues).

    Args:
        n: Number of samples. None or 0 means all samples.
    """
    load_dataset = _ensure_datasets()
    logger.info("Loading Alpaca original (tatsu-lab/stanford_alpaca)...")
    try:
        ds = load_dataset(ALPACA_ORIGINAL, split="train", trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load Alpaca original dataset: {e}") from e

    samples: list[dict] = []
    limit = n if n and n > 0 else len(ds)
    for i, item in enumerate(ds):
        if i >= limit:
            break
        samples.append({"text": _merge_alpaca_fields(item)})
    logger.info("Loaded %d Alpaca original samples.", len(samples))
    return samples


def load_alpaca_cleaned(n: int | None = None) -> list[dict]:
    """Load samples from yahma/alpaca-cleaned (community-cleaned version).

    Args:
        n: Number of samples. None or 0 means all samples.
    """
    load_dataset = _ensure_datasets()
    logger.info("Loading Alpaca cleaned (yahma/alpaca-cleaned)...")
    try:
        ds = load_dataset(ALPACA_CLEANED, split="train", trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load Alpaca cleaned dataset: {e}") from e

    samples: list[dict] = []
    limit = n if n and n > 0 else len(ds)
    for i, item in enumerate(ds):
        if i >= limit:
            break
        samples.append({"text": _merge_alpaca_fields(item)})
    logger.info("Loaded %d Alpaca cleaned samples.", len(samples))
    return samples


def run_benchmark(
    config_path: str | None = None,
    datasets: dict[str, list[dict]] | None = None,
    n: int | None = None,
    no_dedup: bool = True,
    seed: int = 42,
    skip_dedup: bool | None = None,
) -> BenchmarkReport:
    """Run the quality pipeline on each dataset and collect comparison stats.

    Args:
        config_path: Path to pipeline config YAML. None uses default config.
        datasets: Dict of {name: docs}. If None, loads Alpaca original + cleaned.
        n: Number of samples per dataset. 0 or None means all samples.
        no_dedup: If True, skip dedup filters (faster for benchmarks).
        seed: Random seed for reproducibility.
        skip_dedup: Alias for no_dedup (for API compatibility).

    Returns:
        BenchmarkReport with per-dataset, per-filter pass rates.
    """
    import dq.filters  # noqa: F401 — trigger filter registration
    from dq.config import DedupConfig, PipelineConfig
    from dq.pipeline import Pipeline

    if skip_dedup is not None:
        no_dedup = skip_dedup

    if datasets is None:
        datasets = {
            "Alpaca Original": load_alpaca_original(n=n),
            "Alpaca Cleaned": load_alpaca_cleaned(n=n),
        }

    # Load config
    if config_path:
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig.default()

    # Disable dedup for benchmarks if requested
    if no_dedup:
        config.dedup = DedupConfig(exact=False, minhash={"enabled": False})

    report = BenchmarkReport(
        config_path=config_path,
        num_samples=n or 0,
    )

    for ds_name, docs in datasets.items():
        logger.info("Running pipeline on '%s' (%d docs)...", ds_name, len(docs))
        pipeline = Pipeline(config)
        list(pipeline.run(iter(docs)))

        stats = pipeline.stats

        # Compute per-filter results with details
        per_filter: dict[str, FilterResult] = {}
        per_filter_pass_rate: dict[str, float] = {}
        for fs in stats.filter_stats:
            fr = FilterResult(
                total=fs.docs_in,
                passed=fs.docs_out,
                failed=fs.docs_dropped,
                pass_rate=fs.docs_out / fs.docs_in if fs.docs_in > 0 else 1.0,
                sample_failed=fs.sample_drops[:5],
            )
            per_filter[fs.name] = fr
            per_filter_pass_rate[fs.name] = fr.pass_rate

        overall = stats.total_out / stats.total_in if stats.total_in > 0 else 0.0

        result = DatasetResult(
            name=ds_name,
            num_docs=len(docs),
            stats=stats,
            per_filter=per_filter,
            per_filter_pass_rate=per_filter_pass_rate,
            overall_pass_rate=overall,
        )
        report.datasets[ds_name] = result

    return report
