"""Benchmark module: validate quality filters against real-world datasets.

Two-layer benchmark:
- Layer 1: Rule-based filters (pre-training: Gopher/C4/FineWeb; SFT: empty_output/ai_refusal/etc.) — fast, no API cost
- Layer 2: LLM binary judge (opt-in) — data-type-aware binary classification (HIGH/LOW)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dq.pipeline import PipelineStats

logger = logging.getLogger(__name__)

FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_CONFIG = "sample-10BT"
ALPACA_DATASET = "tatsu-lab/alpaca"
ALPACA_ORIGINAL = "tatsu-lab/alpaca"
ALPACA_CLEANED = "yahma/alpaca-cleaned"

# Import canonical SFT field set from sft_rules — single source of truth
from dq.filters.sft_rules import SFT_DETECT_FIELDS as SFT_FIELDS


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


def detect_data_type(docs: list[dict]) -> str:
    """Detect whether docs are SFT (instruction) or pre-training data.

    Returns 'sft' if docs have instruction/conversations fields, 'pretrain' otherwise.
    """
    if not docs:
        return "pretrain"

    # Check first few docs for SFT-specific fields
    sample = docs[:min(10, len(docs))]
    sft_count = 0
    for doc in sample:
        doc_fields = set(doc.keys())
        if doc_fields & SFT_FIELDS:
            sft_count += 1

    # If majority of sampled docs have SFT fields, classify as SFT
    if sft_count > len(sample) / 2:
        return "sft"
    return "pretrain"


def _extract_sft_fields(doc: dict) -> tuple[str, str]:
    """Extract instruction and output from a doc.

    Handles both structured SFT docs (with 'instruction'/'output' fields)
    and merged text docs (split on first newline).
    """
    instruction = doc.get("instruction", "") or ""
    output = doc.get("output", "") or ""

    if instruction and output:
        return instruction, output

    # For merged text, try splitting on newlines
    text = doc.get("text", "") or ""
    if text:
        parts = text.split("\n", 1)
        instruction = parts[0].strip()
        output = parts[1].strip() if len(parts) > 1 else ""

    return instruction, output


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


def load_alpaca_original(n: int | None = None, keep_fields: bool = False) -> list[dict]:
    """Load samples from tatsu-lab/stanford_alpaca (original, known quality issues).

    Downloads directly from GitHub since HuggingFace Hub removed the dataset.

    Args:
        n: Number of samples. None or 0 means all samples.
        keep_fields: If True, preserve instruction/input/output fields alongside text.
    """
    import json
    import random
    from pathlib import Path
    from urllib.request import urlretrieve

    cache_path = Path.home() / ".cache" / "dq" / "alpaca_original.json"
    if not cache_path.exists():
        logger.info("Downloading Alpaca original from GitHub...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        try:
            urlretrieve(url, cache_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download Alpaca original: {e}") from e

    logger.info("Loading Alpaca original from cache...")
    with open(cache_path) as f:
        raw = json.load(f)

    # Shuffle with seed for reproducibility
    rng = random.Random(42)
    rng.shuffle(raw)

    limit = n if n and n > 0 else len(raw)
    samples = []
    for item in raw[:limit]:
        doc: dict[str, Any] = {"text": _merge_alpaca_fields(item)}
        if keep_fields:
            doc["instruction"] = item.get("instruction", "") or ""
            doc["input"] = item.get("input", "") or ""
            doc["output"] = item.get("output", "") or ""
        samples.append(doc)
    logger.info("Loaded %d Alpaca original samples.", len(samples))
    return samples


def load_alpaca_cleaned(n: int | None = None, keep_fields: bool = False) -> list[dict]:
    """Load samples from yahma/alpaca-cleaned (community-cleaned version).

    Args:
        n: Number of samples. None or 0 means all samples.
        keep_fields: If True, preserve instruction/input/output fields alongside text.
    """
    load_dataset = _ensure_datasets()
    logger.info("Loading Alpaca cleaned (yahma/alpaca-cleaned)...")
    try:
        ds = load_dataset(ALPACA_CLEANED, split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load Alpaca cleaned dataset: {e}") from e

    samples: list[dict] = []
    limit = n if n and n > 0 else len(ds)
    for i, item in enumerate(ds):
        if i >= limit:
            break
        doc: dict[str, Any] = {"text": _merge_alpaca_fields(item)}
        if keep_fields:
            doc["instruction"] = item.get("instruction", "") or ""
            doc["input"] = item.get("input", "") or ""
            doc["output"] = item.get("output", "") or ""
        samples.append(doc)
    logger.info("Loaded %d Alpaca cleaned samples.", len(samples))
    return samples


def run_benchmark(
    config_path: str | None = None,
    datasets: dict[str, list[dict]] | None = None,
    n: int | None = None,
    no_dedup: bool = True,
    seed: int = 42,
    skip_dedup: bool | None = None,
    data_type: str = "auto",
    sft_samples: int = 0,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> BenchmarkReport:
    """Run the quality pipeline on each dataset and collect comparison stats.

    Two-layer approach:
    - Layer 1: Rule-based filters (always runs)
    - Layer 2: LLM binary judge (opt-in, data-type-aware)

    For SFT data (auto-detected or specified), Layer 2 uses DEITA complexity
    and quality scoring. For pre-training data, it uses educational value and
    writing quality scoring.

    Args:
        config_path: Path to pipeline config YAML. None uses default config.
        datasets: Dict of {name: docs}. If None, loads Alpaca original + cleaned.
        n: Number of samples per dataset. 0 or None means all samples.
        no_dedup: If True, skip dedup filters (faster for benchmarks).
        seed: Random seed for reproducibility.
        skip_dedup: Alias for no_dedup (for API compatibility).
        data_type: 'sft', 'pretrain', or 'auto' (detect from data).
        sft_samples: Number of docs to score with LLM (0 = skip LLM scoring).
        api_url: LLM API base URL for scoring.
        api_key: LLM API key for scoring.
        model: LLM model name for scoring.

    Returns:
        BenchmarkReport with per-dataset, per-filter pass rates and optional LLM scores.
    """
    from dq.filters import ensure_registered; ensure_registered()
    from dq.config import DedupConfig, PipelineConfig
    from dq.pipeline import Pipeline

    if skip_dedup is not None:
        no_dedup = skip_dedup

    keep_fields = data_type in ("sft", "auto")

    if datasets is None:
        datasets = {
            "Alpaca Original": load_alpaca_original(n=n, keep_fields=keep_fields),
            "Alpaca Cleaned": load_alpaca_cleaned(n=n, keep_fields=keep_fields),
        }

    # Load configs: separate for pretrain vs SFT
    from pathlib import Path

    sft_config_path = Path(__file__).parent.parent.parent / "configs" / "sft.yaml"

    if config_path:
        pretrain_config = PipelineConfig.from_yaml(config_path)
        sft_config = PipelineConfig.from_yaml(config_path)  # user override applies to both
    else:
        pretrain_config = PipelineConfig.default()
        if sft_config_path.exists():
            sft_config = PipelineConfig.from_yaml(str(sft_config_path))
        else:
            sft_config = PipelineConfig.default()

    # Disable dedup for benchmarks if requested
    if no_dedup:
        pretrain_config.dedup = DedupConfig(exact=False, minhash={"enabled": False})
        sft_config.dedup = DedupConfig(exact=False, minhash={"enabled": False})

    report = BenchmarkReport(
        config_path=config_path,
        num_samples=n or 0,
    )

    for ds_name, docs in datasets.items():
        # Auto-detect data type per dataset
        if data_type == "auto":
            ds_type = detect_data_type(docs)
        else:
            ds_type = data_type

        # Use appropriate config
        config = sft_config if ds_type == "sft" else pretrain_config

        logger.info("Running pipeline on '%s' (%d docs, type=%s)...", ds_name, len(docs), ds_type)
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
            data_type=ds_type,
        )
        report.datasets[ds_name] = result

        # Collect per-rule stats via filter_detailed() (independent evaluation)
        ds_rule_stats: dict[str, dict[str, RuleStats]] = {}
        for f in pipeline.filters:
            ds_rule_stats[f.name] = {}

        for doc in docs:
            for f in pipeline.filters:
                _keep, failures = f.filter_detailed(doc)
                failed_rules = {fail["rule"] for fail in failures}
                # Collect all rule names we've seen for this filter
                for fail in failures:
                    rule = fail["rule"]
                    if rule not in ds_rule_stats[f.name]:
                        ds_rule_stats[f.name][rule] = RuleStats()
                    ds_rule_stats[f.name][rule].total += 1
                    ds_rule_stats[f.name][rule].failed += 1

                # Count passes for rules we've seen before but didn't fail this time
                for rule, rs in ds_rule_stats[f.name].items():
                    if rule not in failed_rules:
                        rs.total += 1
                        rs.passed += 1

        # Fix totals: rules discovered mid-way need total = len(docs)
        for filter_name, rules in ds_rule_stats.items():
            for rule_name, rs in rules.items():
                # Every doc was checked against every rule
                rs.total = len(docs)
                rs.passed = rs.total - rs.failed
                rs.pass_rate = rs.passed / rs.total if rs.total > 0 else 1.0

        report.rule_stats[ds_name] = ds_rule_stats

    # Layer 2: LLM binary judge (opt-in)
    if sft_samples > 0:
        # Auto-detect data type from first dataset
        detected_type = data_type
        if data_type == "auto":
            first_docs = next(iter(datasets.values()))
            detected_type = detect_data_type(first_docs)
            logger.info("Auto-detected data type: %s", detected_type)

        report = run_llm_scoring(
            report=report,
            datasets=datasets,
            llm_samples=sft_samples,
            data_type_override=detected_type if detected_type != "auto" else None,
            seed=seed,
            api_url=api_url,
            api_key=api_key,
            model=model,
        )

    return report


def run_llm_scoring(
    report: BenchmarkReport,
    datasets: dict[str, list[dict]],
    llm_samples: int = 50,
    data_type_override: str | None = None,
    seed: int = 42,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    progress: bool = True,
) -> BenchmarkReport:
    """Run Layer 2 LLM scoring on datasets and attach results to the report.

    Args:
        report: Existing BenchmarkReport from Layer 1.
        datasets: Dict of {name: docs} — same datasets used for Layer 1.
        llm_samples: Number of docs to score per dataset (randomly sampled).
        data_type_override: Force 'sft' or 'pretrain'. Auto-detects if None.
        seed: Random seed for sampling.
        api_url: Override LLM API URL.
        api_key: Override LLM API key.
        model: Override LLM model name.
        progress: Show tqdm progress bar.

    Returns:
        The same BenchmarkReport with llm_scores attached to each DatasetResult.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):  # type: ignore[misc]
            return iterable

    report.llm_scoring_enabled = True
    report.llm_samples = llm_samples

    for ds_name, docs in datasets.items():
        if ds_name not in report.datasets:
            continue

        # Detect data type
        if data_type_override:
            data_type = data_type_override
        else:
            data_type = detect_data_type(docs)
        report.datasets[ds_name].data_type = data_type

        # Sample docs for scoring
        rng = random.Random(seed)
        sample_docs = docs[:] if len(docs) <= llm_samples else rng.sample(docs, llm_samples)

        logger.info("LLM scoring '%s' (%d samples, type=%s)...", ds_name, len(sample_docs), data_type)

        scores = _score_docs(sample_docs, data_type, api_url, api_key, model, progress)

        report.datasets[ds_name].llm_scores = scores

    return report


def _score_docs(
    docs: list[dict],
    data_type: str,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Score docs using unified LLM Binary Judge.

    Args:
        docs: List of documents to score.
        data_type: Either "sft" or "pretrain".
        api_url: OpenAI-compatible API URL.
        api_key: API key.
        model: Model name.
        progress: Show progress bar.

    Returns:
        Dictionary with scoring results.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):  # type: ignore[misc]
            return iterable

    from dq.judge import LLMJudge

    judge = LLMJudge(api_url=api_url, api_key=api_key, model=model)

    # Create appropriate scores object based on data type
    if data_type == "sft":
        scores = SFTScores()
        desc = "  SFT judging"
    else:
        scores = PretrainScores()
        desc = "  Pretrain judging"

    for doc in tqdm(docs, desc=desc, disable=not progress):
        # Use appropriate judge method based on data type
        if data_type == "sft":
            instruction, output = _extract_sft_fields(doc)
            result = judge.judge_sft(instruction, output)
        else:
            text = doc.get("text", "")
            result = judge.judge_text(text)

        if "error" in result:
            scores.scoring_errors += 1
        elif result["quality"] == "high":
            scores.high_count += 1
        else:
            scores.low_count += 1
            for rule in result.get("failed_rules", []):
                scores.rule_fail_counts[rule] = scores.rule_fail_counts.get(rule, 0) + 1

    scores.num_scored = len(docs)
    total_judged = scores.high_count + scores.low_count
    scores.high_rate = scores.high_count / total_judged if total_judged > 0 else 0.0

    return scores.to_dict()
