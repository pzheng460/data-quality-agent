"""4-stage pipeline: ingest → extract → curate → package.

Each stage reads from the previous stage's output directory and writes
to its own output directory, enabling resume and debugging.
"""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from dq.shared.shard import ShardWriter, read_shards
from dq.shared.stats import PhaseStats, PhaseTimer

if TYPE_CHECKING:
    from dq.runner.engine import PhaseEngine

logger = logging.getLogger(__name__)


# ── Stage 1: Ingestion ──────────────────────────────────────────────


def stage_ingest(engine: PhaseEngine) -> PhaseStats:
    """Fetch raw data via registered source. Write to stage1_ingested/."""
    stats = PhaseStats(phase="ingestion")
    kept_dir = engine.stage_dir("stage1_ingested", "kept")

    with PhaseTimer(stats), \
         ShardWriter(kept_dir, target_bytes=engine.shard_target_bytes) as w:
        for doc in engine.iter_input():
            w.write(doc)
            stats.output_count += 1
            stats.input_count += 1

    return stats


# ── Stage 2: Extraction ─────────────────────────────────────────────


def _extract_one(doc: dict, extractor_name: str) -> tuple[bool, dict]:
    """Worker: run extraction on a single doc. Returns (kept, doc)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from dq.stages.extraction import ensure_extractors_registered
    ensure_extractors_registered()
    from dq.stages.extraction.registry import (
        get_extractor_class,
        get_extractor_for_format,
    )

    if extractor_name and extractor_name != "auto":
        extractor = get_extractor_class(extractor_name)()
    else:
        fmt = _detect_format(doc)
        extractor = get_extractor_for_format(fmt)()

    try:
        result = extractor.extract(doc)
    except Exception as e:
        logger.warning("Extraction error on %s: %s", doc.get("id", "?"), e)
        return False, doc

    if result is None:
        return False, doc
    return True, result


def stage_extract(engine: PhaseEngine) -> PhaseStats:
    """Convert raw format to clean text via registered extractors.

    Parallelized with multiprocessing to overlap LaTeXML subprocess calls.
    """
    stats = PhaseStats(phase="extraction")

    from dq.stages.extraction import ensure_extractors_registered
    ensure_extractors_registered()

    extractor_name = engine.extra_config.get("extraction", {}).get("extractor", "auto")

    input_dir = engine.stage_dir("stage1_ingested", "kept")
    kept_dir = engine.stage_dir("stage2_extracted", "kept")
    rejected_dir = engine.stage_dir("stage2_extracted", "rejected")

    workers = max(1, engine.workers)
    logger.info("Extraction: %d worker(s), extractor=%s", workers, extractor_name)

    with PhaseTimer(stats), \
         ShardWriter(kept_dir, target_bytes=engine.shard_target_bytes) as kept_w, \
         ShardWriter(rejected_dir, target_bytes=engine.shard_target_bytes) as rej_w:

        def _record(kept: bool, doc: dict) -> None:
            stats.input_count += 1
            if kept:
                kept_w.write(doc)
                stats.output_count += 1
            else:
                doc["__dq_rejections"] = [{"filter": "extraction", "rule": "extraction_failed"}]
                rej_w.write(doc)
                stats.rejected_count += 1
                stats.reject_reasons["extraction_failed"] = (
                    stats.reject_reasons.get("extraction_failed", 0) + 1
                )

        from tqdm import tqdm
        fn = partial(_extract_one, extractor_name=extractor_name)
        results_iter = engine.backend.map(fn, read_shards(input_dir), kind="cpu")
        for kept, result in tqdm(results_iter, desc="extract", unit="doc"):
            _record(kept, result)

    return stats


def _detect_format(doc: dict) -> str:
    """Detect the format of a raw doc."""
    text = doc.get("text", "")
    if not text:
        return "text"
    # HTML detection
    if "<html" in text[:500].lower() or "<!doctype" in text[:500].lower():
        return "html"
    # LaTeX detection
    if "\\begin{document}" in text or "\\documentclass" in text[:500]:
        return "latex"
    return "text"


# ── Stage 3: Curation ───────────────────────────────────────────────


def stage_curate(engine: PhaseEngine) -> PhaseStats:
    """Filter + quality scoring + dedup + contamination.

    Runs sub-steps sequentially within one stage.
    """
    stats = PhaseStats(phase="curation")

    input_dir = engine.stage_dir("stage2_extracted", "kept")
    kept_dir = engine.stage_dir("stage3_curated", "kept")
    rejected_dir = engine.stage_dir("stage3_curated", "rejected")

    with PhaseTimer(stats):
        docs = list(read_shards(input_dir))
        stats.input_count = len(docs)
        logger.info("Curation: %d docs from extraction", len(docs))

        all_rejected: list[dict] = []

        # 3a. Heuristic filters
        docs, rejected = _substep_filter(engine, docs)
        all_rejected.extend(rejected)
        for doc in rejected:
            for r in doc.get("__dq_rejections", []):
                key = f"{r.get('filter', '?')}.{r.get('rule', '?')}"
                stats.reject_reasons[key] = stats.reject_reasons.get(key, 0) + 1

        # 3b. Dedup
        docs, dedup_rejected = _substep_dedup(engine, docs)
        all_rejected.extend(dedup_rejected)
        for doc in dedup_rejected:
            for r in doc.get("__dq_rejections", []):
                key = f"dedup.{r.get('rule', '?')}"
                stats.reject_reasons[key] = stats.reject_reasons.get(key, 0) + 1

        # 3c. Contamination
        contam_cfg = engine.extra_config.get("phase4", {})
        if contam_cfg:
            docs, contam_rejected = _substep_contamination(engine, docs, contam_cfg)
            all_rejected.extend(contam_rejected)

        # 3d. LLM quality judge (runs last — on final survivor set to minimize cost)
        quality_cfg = (
            engine.extra_config.get("quality_scoring")
            or engine.extra_config.get("arxiv", {}).get("quality_scoring", {})
        )
        if quality_cfg.get("enabled", False):
            docs, llm_rejected = _substep_quality_score(engine, docs, quality_cfg)
            for doc in llm_rejected:
                for r in doc.get("__dq_rejections", []):
                    key = f"llm_judge.{r.get('rule', '?')}"
                    stats.reject_reasons[key] = stats.reject_reasons.get(key, 0) + 1
            all_rejected.extend(llm_rejected)

        # Write results
        stats.output_count = len(docs)
        stats.rejected_count = stats.input_count - stats.output_count

        with ShardWriter(kept_dir, target_bytes=engine.shard_target_bytes) as kw, \
             ShardWriter(rejected_dir, target_bytes=engine.shard_target_bytes) as rw:
            for doc in docs:
                kw.write(doc)
            for doc in all_rejected:
                rw.write(doc)

    return stats


# ── Curation sub-steps ──


def _filter_chunk(chunk, filter_configs, text_field):
    """Worker for parallel filtering (imported from phases.py logic)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from dq.stages.curation.filters import ensure_registered
    ensure_registered()
    from dq.pipeline import get_filter_class

    filters = [get_filter_class(fc["name"])(text_field=text_field, **fc["params"])
               for fc in filter_configs]

    kept, rejected, reasons = [], [], {}
    for doc in chunk:
        pass_all = True
        rejections = []
        for f in filters:
            ok, failures = f.filter_detailed(doc)
            if not ok:
                rejections.extend(failures)
                pass_all = False
            for fail in failures:
                key = f"{fail.get('filter', '')}.{fail.get('rule', '')}"
                reasons[key] = reasons.get(key, 0) + 1
        if pass_all:
            kept.append(doc)
        else:
            doc["__dq_rejections"] = rejections
            rejected.append(doc)
    return kept, rejected, reasons


def _substep_filter(engine, docs):
    """Run heuristic filters on docs (parallel via engine.backend)."""
    filter_configs = [{"name": fc.name, "params": fc.params}
                      for fc in engine.config.filters if fc.enabled]
    if not filter_configs:
        return docs, []

    workers = engine.workers
    if workers > 1 and len(docs) >= workers * 10:
        chunk_size = max(1, (len(docs) + workers - 1) // workers)
        chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
        fn = partial(_filter_chunk,
                     filter_configs=filter_configs,
                     text_field=engine.config.text_field)
        all_kept, all_rejected = [], []
        for kept, rejected, _ in engine.backend.map(fn, chunks, kind="cpu"):
            all_kept.extend(kept)
            all_rejected.extend(rejected)
        return all_kept, all_rejected

    kept, rejected, _ = _filter_chunk(docs, filter_configs, engine.config.text_field)
    return kept, rejected


def _substep_quality_score(engine, docs, quality_cfg):
    """LLM quality judge — scores each doc HIGH/LOW, rejects LOW when min_quality="high".

    Runs LLM calls in parallel via ThreadPoolExecutor (IO-bound).

    Config keys (quality_scoring):
        enabled: bool
        method: "llm" (only option currently)
        sample_size: 0 = judge all docs; N = judge only N random docs (others pass-through)
        min_quality: "high" | "low" — drop docs judged below this
        workers: int — parallel API calls (default 8)
        max_chars: int — truncate each doc to this many chars before sending (default 3000)

    Returns:
        (kept, rejected): both are lists of doc dicts.
    """
    method = quality_cfg.get("method", "llm")
    sample_size = int(quality_cfg.get("sample_size", quality_cfg.get("llm_samples", 0)))
    min_quality = quality_cfg.get("min_quality", "high")
    workers = int(quality_cfg.get("workers", 8))
    max_chars = int(quality_cfg.get("max_chars", 3000))

    if method != "llm" or not docs:
        return docs, []

    # Decide which docs to judge. Un-sampled docs pass through un-judged.
    import random
    if sample_size > 0 and sample_size < len(docs):
        to_judge_idx = set(random.sample(range(len(docs)), sample_size))
    else:
        to_judge_idx = set(range(len(docs)))

    try:
        from dq.judge import LLMJudge
        judge = LLMJudge()
    except Exception as e:
        logger.warning("LLM judge unavailable, skipping quality scoring: %s", e)
        return docs, []

    def _judge_one(i_doc):
        i, doc = i_doc
        if i not in to_judge_idx:
            return i, None  # skipped — keep as-is
        try:
            return i, judge.judge_text(doc.get("text", "")[:max_chars])
        except Exception as e:
            logger.warning("judge_text failed for doc %d: %s", i, e)
            return i, {"quality": "unknown", "error": str(e)}

    from concurrent.futures import ThreadPoolExecutor
    results: dict[int, dict | None] = {}
    logger.info("LLM judge: %d/%d docs, %d concurrent workers",
                len(to_judge_idx), len(docs), max(1, workers))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, res in pool.map(_judge_one, enumerate(docs)):
            results[i] = res

    kept, rejected = [], []
    for i, doc in enumerate(docs):
        res = results.get(i)
        if res is None:
            kept.append(doc)
            continue
        doc.setdefault("quality_scores", {})["llm_judge"] = res
        if min_quality == "high" and res.get("quality") != "high":
            doc.setdefault("__dq_rejections", []).append({
                "filter": "llm_judge",
                "rule": "below_min_quality",
                "failed_rules": res.get("failed_rules", []),
            })
            rejected.append(doc)
        else:
            kept.append(doc)

    logger.info("LLM judge: %d kept, %d rejected", len(kept), len(rejected))
    return kept, rejected


def _substep_dedup(engine, docs):
    """Exact + MinHash dedup."""
    from dq.stages.curation.dedup.exact import ExactDedup

    rejected = []

    # Version dedup
    phase3_cfg = engine.extra_config.get("phase3", {})
    if phase3_cfg.get("version_dedup", False):
        by_aid = {}
        for doc in docs:
            aid = doc.get("metadata", {}).get("arxiv_id", doc.get("id", ""))
            by_aid.setdefault(aid, []).append(doc)
        kept = []
        for aid, versions in by_aid.items():
            if len(versions) == 1:
                kept.append(versions[0])
            else:
                versions.sort(key=lambda d: d.get("metadata", {}).get("version", "v0"))
                kept.append(versions[-1])
                for v in versions[:-1]:
                    v["__dq_rejections"] = [{"filter": "dedup", "rule": "older_version"}]
                    rejected.append(v)
        docs = kept

    # Exact dedup
    exact = ExactDedup(text_field=engine.config.text_field)
    docs = list(exact.dedup(docs))

    # MinHash dedup
    minhash_cfg = engine.config.dedup.minhash
    if minhash_cfg.get("enabled", False):
        from dq.stages.curation.dedup.minhash import MinHashDedup
        mh = MinHashDedup(
            text_field=engine.config.text_field,
            num_perm=minhash_cfg.get("num_perm", 112),
            bands=minhash_cfg.get("bands", 14),
            rows=minhash_cfg.get("rows", 8),
            ngram_size=minhash_cfg.get("ngram_size", 5),
        )
        docs = list(mh.dedup(docs))

    return docs, []


def _substep_contamination(engine, docs, contam_cfg):
    """N-gram contamination check."""
    from dq.stages.curation.contamination.ngram import NgramContaminationDetector, load_benchmark

    ngram_size = contam_cfg.get("ngram_size", 13)
    threshold = contam_cfg.get("threshold", 0.8)
    benchmark_names = contam_cfg.get("benchmarks", [])

    if not benchmark_names:
        return docs, []

    detector = NgramContaminationDetector(n=ngram_size, threshold=threshold)
    for name in benchmark_names:
        try:
            texts = load_benchmark(name)
            detector.build_index(texts, benchmark_name=name)
        except Exception as e:
            logger.warning("Failed to load benchmark %s: %s", name, e)

    kept, rejected = [], []
    for doc in docs:
        result = detector.check_contamination(doc.get(engine.config.text_field, ""))
        if result.is_contaminated:
            bm = result.matched_benchmark or "unknown"
            doc["__dq_rejections"] = [{"filter": "contamination", "rule": f"contaminated_{bm}"}]
            rejected.append(doc)
        else:
            kept.append(doc)

    return kept, rejected


# ── Stage 4: Packaging ───────────────────────────────────────────────


def stage_package(engine: PhaseEngine) -> PhaseStats:
    """Sort, shard, and write manifest."""
    from dq.shared.shard import ShardWriter, write_manifest

    stats = PhaseStats(phase="packaging")
    pkg_cfg = engine.extra_config.get("phase5", {})
    sort_field = pkg_cfg.get("sort_by", "id")

    input_dir = engine.stage_dir("stage3_curated", "kept")
    final_dir = engine.stage_dir("stage4_final")

    with PhaseTimer(stats):
        docs = list(read_shards(input_dir))
        stats.input_count = len(docs)
        docs.sort(key=lambda d: d.get(sort_field, ""))

        with ShardWriter(final_dir, target_bytes=engine.shard_target_bytes) as w:
            for doc in docs:
                w.write(doc)
                stats.output_count += 1

        version = engine.extra_config.get("version", engine.version)
        write_manifest(final_dir, w.shard_info, version=version)

    stats.rejected_count = 0
    return stats


# ── Timer helper ──


