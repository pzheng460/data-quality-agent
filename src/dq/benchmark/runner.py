"""Benchmark runner functions."""

import logging
import random
from pathlib import Path
from typing import Any

from .datasets import load_alpaca_original, load_alpaca_cleaned
from .types import BenchmarkReport, DatasetResult, DatasetStats, DedupStats, FilterResult, RuleStats, SFTScores, PretrainScores
from .utils import detect_data_type, _extract_sft_fields

logger = logging.getLogger(__name__)


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
    sft_config_path = Path(__file__).parent.parent.parent.parent / "configs" / "sft.yaml"

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

        # Compute basic dataset stats
        from dq.utils.stats import word_count as wc_fn, avg_word_length as awl_fn
        from dq.dedup.exact import ExactDedup
        word_counts = [wc_fn(d.get("text", "")) for d in docs]

        # Exact dedup detection (lightweight, no filtering)
        exact = ExactDedup(text_field="text")
        list(exact.dedup(docs))  # consume to compute stats
        dedup_info = DedupStats(
            exact_duplicates=exact.duplicate_docs,
            duplicate_rate=exact.duplicate_docs / exact.total_docs if exact.total_docs > 0 else 0.0,
        )

        ds_stats = DatasetStats(
            avg_word_count=sum(word_counts) / len(word_counts) if word_counts else 0.0,
            min_word_count=min(word_counts) if word_counts else 0,
            max_word_count=max(word_counts) if word_counts else 0,
            avg_word_length=sum(awl_fn(d.get("text", "")) for d in docs) / len(docs) if docs else 0.0,
            fields=list(docs[0].keys()) if docs else [],
            dedup=dedup_info,
        )

        logger.info("Running pipeline on '%s' (%d docs, type=%s)...", ds_name, len(docs), ds_type)
        pipeline = Pipeline(config)

        # Independent evaluation: run filter_detailed() on ALL docs for every filter
        # This gives consistent per-filter and per-rule stats on the same total
        num_docs = len(docs)
        per_filter: dict[str, FilterResult] = {}
        per_filter_pass_rate: dict[str, float] = {}
        ds_rule_stats: dict[str, dict[str, RuleStats]] = {}
        # Track per-filter failures (doc fails ANY rule in that filter)
        filter_fail_counts: dict[str, int] = {}
        filter_sample_drops: dict[str, list[dict]] = {}

        for f in pipeline.filters:
            ds_rule_stats[f.name] = {}
            filter_fail_counts[f.name] = 0
            filter_sample_drops[f.name] = []

        for doc in docs:
            for f in pipeline.filters:
                _keep, failures = f.filter_detailed(doc)
                failed_rules = {fail["rule"] for fail in failures}

                # Per-filter: count docs that fail this filter (any rule)
                if not _keep:
                    filter_fail_counts[f.name] += 1
                    if len(filter_sample_drops[f.name]) < 5:
                        text = doc.get("text", "")
                        # Use first failure reason for sample drop
                        reason = failures[0] if failures else {}
                        filter_sample_drops[f.name].append({
                            "text_preview": text[:200],
                            "reason": {
                                "filter": f.name,
                                "reason": reason.get("rule", "unknown"),
                                "value": reason.get("value", ""),
                            },
                        })

                # Per-rule: track each rule independently
                for fail in failures:
                    rule = fail["rule"]
                    if rule not in ds_rule_stats[f.name]:
                        ds_rule_stats[f.name][rule] = RuleStats()
                    ds_rule_stats[f.name][rule].failed += 1

                for rule in ds_rule_stats[f.name]:
                    if rule not in failed_rules:
                        ds_rule_stats[f.name][rule].passed += 1

        # Fix totals for rules discovered mid-way
        for filter_name, rules in ds_rule_stats.items():
            for rule_name, rs in rules.items():
                rs.total = num_docs
                rs.passed = rs.total - rs.failed
                rs.pass_rate = rs.passed / rs.total if rs.total > 0 else 1.0

        # Build per-filter results (independent, all docs)
        overall_failed = 0
        for f in pipeline.filters:
            failed = filter_fail_counts[f.name]
            passed = num_docs - failed
            rate = passed / num_docs if num_docs > 0 else 1.0
            fr = FilterResult(
                total=num_docs,
                passed=passed,
                failed=failed,
                pass_rate=rate,
                sample_failed=filter_sample_drops[f.name],
            )
            per_filter[f.name] = fr
            per_filter_pass_rate[f.name] = rate

        # Overall: doc passes only if it passes ALL filters
        docs_passing_all = 0
        for doc in docs:
            pass_all = True
            for f in pipeline.filters:
                keep, _ = f.filter_detailed(doc)
                if not keep:
                    pass_all = False
                    break
            if pass_all:
                docs_passing_all += 1
        overall = docs_passing_all / num_docs if num_docs > 0 else 0.0

        result = DatasetResult(
            name=ds_name,
            num_docs=num_docs,
            per_filter=per_filter,
            per_filter_pass_rate=per_filter_pass_rate,
            overall_pass_rate=overall,
            data_type=ds_type,
            dataset_stats=ds_stats,
        )
        report.datasets[ds_name] = result
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