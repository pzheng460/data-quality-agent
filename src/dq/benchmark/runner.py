"""Benchmark runner functions."""

import logging
import os
import random
from functools import partial
from pathlib import Path
from typing import Any

from .datasets import load_alpaca_original, load_alpaca_cleaned
from .types import BenchmarkReport, DatasetResult, DatasetStats, DedupStats, FilterResult, RuleStats, SFTScores, PretrainScores
from .utils import detect_data_type, _extract_sft_fields

logger = logging.getLogger(__name__)


# ── Worker function for multiprocessing ────────────────────────────

def _eval_chunk(
    chunk: list[dict],
    filter_configs: list[dict],
    text_field: str,
    collect_rejected: bool = False,
) -> dict:
    """Process a chunk of docs through all filters. Runs in a worker process.

    Each worker loads its own filter instances and spacy tokenizer.

    Args:
        chunk: List of documents to process.
        filter_configs: List of dicts with 'name' and 'params' for each filter.
        text_field: Text field name.
        collect_rejected: If True, collect all rejected docs with full text and reasons.

    Returns:
        Dict with per-filter fail counts, per-rule stats, and overall pass count.
    """
    # Limit internal threading to avoid CPU contention with other workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    from dq.stages.curation.filters import ensure_registered; ensure_registered()
    from dq.pipeline import get_filter_class

    # Build filters in this process
    filters = []
    for fc in filter_configs:
        cls = get_filter_class(fc["name"])
        filters.append(cls(text_field=text_field, **fc["params"]))

    # Per-filter stats
    filter_fail_counts: dict[str, int] = {f.name: 0 for f in filters}
    filter_sample_drops: dict[str, list[dict]] = {f.name: [] for f in filters}
    rule_fail_counts: dict[str, dict[str, int]] = {f.name: {} for f in filters}
    docs_passing_all = 0

    # Collect all rejected docs when requested
    rejected_docs: list[dict] = []

    # Dataset stats (computed alongside filtering to avoid extra tokenization pass)
    from dq.utils.stats import word_count as wc_fn, avg_word_length as awl_fn
    word_counts: list[int] = []
    total_awl: float = 0.0

    for doc in chunk:
        text = doc.get("text", "")
        word_counts.append(wc_fn(text))
        total_awl += awl_fn(text)

        pass_all = True
        doc_rejections: list[dict] = []
        for f in filters:
            _keep, failures = f.filter_detailed(doc)

            if not _keep:
                filter_fail_counts[f.name] += 1
                pass_all = False
                if len(filter_sample_drops[f.name]) < 5:
                    text = doc.get("text", "")
                    reason = failures[0] if failures else {}
                    filter_sample_drops[f.name].append({
                        "text_preview": text[:200],
                        "reason": {
                            "filter": f.name,
                            "reason": reason.get("rule", "unknown"),
                            "value": reason.get("value", ""),
                        },
                    })

                if collect_rejected:
                    for fail in failures:
                        doc_rejections.append({
                            "filter": f.name,
                            "rule": fail.get("rule", "unknown"),
                            "value": fail.get("value", ""),
                            "threshold": fail.get("threshold", ""),
                        })

            for fail in failures:
                rule = fail["rule"]
                if rule not in rule_fail_counts[f.name]:
                    rule_fail_counts[f.name][rule] = 0
                rule_fail_counts[f.name][rule] += 1

        if pass_all:
            docs_passing_all += 1
        elif collect_rejected and doc_rejections:
            rejected_doc = dict(doc)
            rejected_doc["__dq_rejections"] = doc_rejections
            rejected_docs.append(rejected_doc)

    result = {
        "num_docs": len(chunk),
        "filter_fail_counts": filter_fail_counts,
        "filter_sample_drops": filter_sample_drops,
        "rule_fail_counts": rule_fail_counts,
        "docs_passing_all": docs_passing_all,
        "word_counts": word_counts,
        "total_awl": total_awl,
    }
    if collect_rejected:
        result["rejected_docs"] = rejected_docs
    return result


def _merge_chunk_results(
    results: list[dict],
    filter_names: list[str],
    total_docs: int,
) -> tuple[dict[str, int], dict[str, list[dict]], dict[str, dict[str, RuleStats]], int]:
    """Merge results from multiple worker chunks."""
    merged_fail_counts: dict[str, int] = {name: 0 for name in filter_names}
    merged_sample_drops: dict[str, list[dict]] = {name: [] for name in filter_names}
    merged_rule_fails: dict[str, dict[str, int]] = {name: {} for name in filter_names}
    total_passing_all = 0

    for r in results:
        total_passing_all += r["docs_passing_all"]
        for name in filter_names:
            merged_fail_counts[name] += r["filter_fail_counts"][name]
            # Keep up to 5 sample drops total
            if len(merged_sample_drops[name]) < 5:
                remaining = 5 - len(merged_sample_drops[name])
                merged_sample_drops[name].extend(r["filter_sample_drops"][name][:remaining])
            for rule, count in r["rule_fail_counts"][name].items():
                merged_rule_fails[name][rule] = merged_rule_fails[name].get(rule, 0) + count

    # Convert to RuleStats
    ds_rule_stats: dict[str, dict[str, RuleStats]] = {}
    for name in filter_names:
        ds_rule_stats[name] = {}
        for rule, failed in merged_rule_fails[name].items():
            passed = total_docs - failed
            ds_rule_stats[name][rule] = RuleStats(
                total=total_docs,
                passed=passed,
                failed=failed,
                pass_rate=passed / total_docs if total_docs > 0 else 1.0,
            )

    return merged_fail_counts, merged_sample_drops, ds_rule_stats, total_passing_all


def _get_default_workers() -> int:
    """Default worker count: min(cpu_count / 4, 16), at least 1."""
    cpus = os.cpu_count() or 1
    return max(1, min(cpus // 4, 16))


# ── Main benchmark function ───────────────────────────────────────

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
    workers: int | None = None,
    save_rejected: bool = False,
) -> BenchmarkReport:
    """Run the quality pipeline on each dataset and collect comparison stats.

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
        workers: Number of parallel workers. None = auto-detect.
        save_rejected: If True, collect all rejected docs in report.rejected_docs.

    Returns:
        BenchmarkReport with per-dataset, per-filter pass rates.
    """
    from dq.stages.curation.filters import ensure_registered; ensure_registered()
    from dq.config import DedupConfig, PipelineConfig
    from dq.pipeline import Pipeline

    if skip_dedup is not None:
        no_dedup = skip_dedup

    if workers is None:
        workers = _get_default_workers()

    keep_fields = data_type in ("sft", "auto")

    if datasets is None:
        datasets = {
            "Alpaca Original": load_alpaca_original(n=n, keep_fields=keep_fields),
            "Alpaca Cleaned": load_alpaca_cleaned(n=n, keep_fields=keep_fields),
        }

    # Load configs
    sft_config_path = Path(__file__).parent.parent.parent.parent / "configs" / "sft.yaml"

    if config_path:
        pretrain_config = PipelineConfig.from_yaml(config_path)
        sft_config = PipelineConfig.from_yaml(config_path)
    else:
        pretrain_config = PipelineConfig.default()
        if sft_config_path.exists():
            sft_config = PipelineConfig.from_yaml(str(sft_config_path))
        else:
            sft_config = PipelineConfig.default()

    if no_dedup:
        pretrain_config.dedup = DedupConfig(exact=False, minhash={"enabled": False})
        sft_config.dedup = DedupConfig(exact=False, minhash={"enabled": False})

    report = BenchmarkReport(
        config_path=config_path,
        num_samples=n or 0,
    )

    for ds_name, docs in datasets.items():
        if data_type == "auto":
            ds_type = detect_data_type(docs)
        else:
            ds_type = data_type

        config = sft_config if ds_type == "sft" else pretrain_config

        logger.info("Running pipeline on '%s' (%d docs, type=%s, workers=%d)...",
                     ds_name, len(docs), ds_type, workers)
        pipeline = Pipeline(config)

        # Build serializable filter configs for workers
        filter_configs = []
        for fc in config.filters:
            if fc.enabled:
                filter_configs.append({"name": fc.name, "params": fc.params})
        filter_names = [fc["name"] for fc in filter_configs]

        num_docs = len(docs)

        # Run evaluation (parallel or single-process)
        # Workers compute both filter results AND dataset stats in one pass
        if workers > 1 and num_docs >= workers * 10:
            chunk_results = _run_parallel(docs, filter_configs, config.text_field, workers,
                                          collect_rejected=save_rejected)
        else:
            chunk_results = [_eval_chunk(docs, filter_configs, config.text_field,
                                         collect_rejected=save_rejected)]

        # Merge results
        filter_fail_counts, filter_sample_drops, ds_rule_stats, docs_passing_all = \
            _merge_chunk_results(chunk_results, filter_names, num_docs)

        # Merge dataset stats from workers (avoids separate tokenization pass)
        all_word_counts: list[int] = []
        total_awl = 0.0
        for r in chunk_results:
            all_word_counts.extend(r["word_counts"])
            total_awl += r["total_awl"]

        # Dedup detection (reports stats only, does not remove docs)
        from dq.stages.curation.dedup.exact import ExactDedup
        exact = ExactDedup(text_field=config.text_field)
        list(exact.dedup(docs))
        exact_dups = exact.duplicate_docs
        exact_rate = exact.duplicate_docs / exact.total_docs if exact.total_docs > 0 else 0.0

        minhash_dups = 0
        minhash_rate = 0.0
        minhash_cfg = config.dedup.minhash
        if minhash_cfg.get("enabled", False) and not no_dedup:
            from dq.stages.curation.dedup.minhash import MinHashDedup
            mh = MinHashDedup(
                text_field=config.text_field,
                num_perm=minhash_cfg.get("num_perm", 112),
                bands=minhash_cfg.get("bands", 14),
                rows=minhash_cfg.get("rows", 8),
                ngram_size=minhash_cfg.get("ngram_size", 5),
            )
            list(mh.dedup(docs))
            minhash_dups = mh.duplicate_docs
            minhash_rate = mh.duplicate_docs / mh.total_docs if mh.total_docs > 0 else 0.0
            logger.info("MinHash dedup: %d near-duplicates (%.1f%%)", minhash_dups, minhash_rate * 100)

        dedup_info = DedupStats(
            exact_duplicates=exact_dups,
            duplicate_rate=exact_rate,
            minhash_duplicates=minhash_dups,
            minhash_rate=minhash_rate,
        )

        ds_stats = DatasetStats(
            avg_word_count=sum(all_word_counts) / len(all_word_counts) if all_word_counts else 0.0,
            min_word_count=min(all_word_counts) if all_word_counts else 0,
            max_word_count=max(all_word_counts) if all_word_counts else 0,
            avg_word_length=total_awl / num_docs if num_docs > 0 else 0.0,
            fields=list(docs[0].keys()) if docs else [],
            dedup=dedup_info,
        )

        # Build per-filter results
        per_filter: dict[str, FilterResult] = {}
        per_filter_pass_rate: dict[str, float] = {}
        for name in filter_names:
            failed = filter_fail_counts[name]
            passed = num_docs - failed
            rate = passed / num_docs if num_docs > 0 else 1.0
            per_filter[name] = FilterResult(
                total=num_docs,
                passed=passed,
                failed=failed,
                pass_rate=rate,
                sample_failed=filter_sample_drops[name],
            )
            per_filter_pass_rate[name] = rate

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

        # Merge rejected docs from all chunks
        if save_rejected:
            ds_rejected: list[dict] = []
            for r in chunk_results:
                ds_rejected.extend(r.get("rejected_docs", []))
            report.rejected_docs[ds_name] = ds_rejected

    # Layer 2: LLM binary judge (opt-in)
    if sft_samples > 0:
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


def _run_parallel(
    docs: list[dict],
    filter_configs: list[dict],
    text_field: str,
    workers: int,
    collect_rejected: bool = False,
) -> list[dict]:
    """Split docs into chunks and process in parallel using multiprocessing."""
    from multiprocessing import get_context

    # Limit internal threading per worker to avoid CPU contention.
    # Must be set before spawning so child processes inherit it.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    chunk_size = (len(docs) + workers - 1) // workers
    chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]

    worker_fn = partial(_eval_chunk, filter_configs=filter_configs, text_field=text_field,
                        collect_rejected=collect_rejected)

    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.map(worker_fn, chunks)

    return results


# ── LLM scoring (unchanged) ───────────────────────────────────────

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
    """Run Layer 2 LLM scoring on datasets and attach results to the report."""
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

        if data_type_override:
            dt = data_type_override
        else:
            dt = detect_data_type(docs)
        report.datasets[ds_name].data_type = dt

        rng = random.Random(seed)
        sample_docs = docs[:] if len(docs) <= llm_samples else rng.sample(docs, llm_samples)

        logger.info("LLM scoring '%s' (%d samples, type=%s)...", ds_name, len(sample_docs), dt)

        scores = _score_docs(sample_docs, dt, api_url, api_key, model, progress)
        report.datasets[ds_name].llm_scores = scores

    return report


def _score_docs(
    docs: list[dict],
    data_type: str,
    api_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    progress: bool = True,
    workers: int = 8,
) -> dict[str, Any]:
    """Score docs using unified LLM Binary Judge."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):  # type: ignore[misc]
            return iterable

    from dq.judge import LLMJudge

    judge = LLMJudge(api_url=api_url, api_key=api_key, model=model)

    if data_type == "sft":
        scores = SFTScores()
        desc = "  SFT judging"
    else:
        scores = PretrainScores()
        desc = "  Pretrain judging"

    def _judge_one(doc):
        if data_type == "sft":
            instruction, output = _extract_sft_fields(doc)
            return judge.judge_sft(instruction, output)
        return judge.judge_text(doc.get("text", ""))

    from concurrent.futures import ThreadPoolExecutor
    from dq.judge import RULES_BY_NAME
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results_iter = ex.map(_judge_one, docs)
        for result in tqdm(results_iter, total=len(docs), desc=desc, disable=not progress):
            if "error" in result:
                scores.scoring_errors += 1
                continue
            if result.get("quality") == "high":
                scores.high_count += 1
            else:
                scores.low_count += 1

            # Aggregate per-rule pass/fail + per-rule score
            for name, info in (result.get("rules") or {}).items():
                if not info.get("pass", True):
                    scores.rule_fail_counts[name] = scores.rule_fail_counts.get(name, 0) + 1
                if "score" in info:
                    scores.rule_score_sums[name] = scores.rule_score_sums.get(name, 0.0) + float(info["score"])
                    scores.rule_score_counts[name] = scores.rule_score_counts.get(name, 0) + 1
                # Pull up rule metadata once so the UI can render correctly
                if name not in scores.rule_modes:
                    rule = RULES_BY_NAME.get(name)
                    if rule:
                        scores.rule_modes[name] = rule.mode
                        scores.rule_thresholds[name] = rule.threshold
                        scores.rule_max_scores[name] = float(rule.max_score)
                    else:
                        scores.rule_modes[name] = info.get("mode", "binary")

    scores.num_scored = len(docs)
    total_judged = scores.high_count + scores.low_count
    scores.high_rate = scores.high_count / total_judged if total_judged > 0 else 0.0

    return scores.to_dict()
