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

from dq.shared.shard import (
    ShardWriter,
    SingleShardWriter,
    read_shards,
    read_shard,
    list_shards,
    is_shard_done,
    mark_shard_done,
)
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
    """Convert raw format to clean text. Shard-level resumable.

    Processes one input shard at a time: reads docs → extracts in parallel →
    writes matching output shards → drops `.done/<shard>` marker. A re-run
    skips any input shard whose marker already exists.
    """
    stats = PhaseStats(phase="extraction")

    from dq.stages.extraction import ensure_extractors_registered
    ensure_extractors_registered()

    extractor_name = engine.extra_config.get("extraction", {}).get("extractor", "auto")

    input_dir = engine.stage_dir("stage1_ingested", "kept")
    stage_output = engine.stage_dir("stage2_extracted")
    kept_dir = stage_output / "kept"
    rej_dir = stage_output / "rejected"
    kept_dir.mkdir(parents=True, exist_ok=True)
    rej_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, engine.workers)
    shard_files = list_shards(input_dir)
    logger.info("Extraction: %d worker(s), %d input shards, extractor=%s",
                workers, len(shard_files), extractor_name)

    with PhaseTimer(stats):
        from tqdm import tqdm
        fn = partial(_extract_one, extractor_name=extractor_name)

        # Partition input shards into done vs pending for clear progress reporting.
        pending = [p for p in shard_files if not is_shard_done(stage_output, p.name)]
        skipped = len(shard_files) - len(pending)
        if skipped:
            logger.info("extract: resuming — %d/%d shards already done, %d to go",
                        skipped, len(shard_files), len(pending))

        shard_bar = tqdm(pending, desc="extract (shards)", unit="shard",
                         disable=not pending)
        for shard_path in shard_bar:
            name = shard_path.name
            shard_bar.set_postfix_str(name)

            kept_tmp = kept_dir / (name + ".part")
            rej_tmp = rej_dir / (name + ".part")
            local_in = local_kept = local_rej = 0

            try:
                with SingleShardWriter(kept_tmp) as kept_w, SingleShardWriter(rej_tmp) as rej_w:
                    results_iter = engine.backend.map(fn, read_shard(shard_path), kind="cpu")
                    # Nested per-doc bar; `leave=False` keeps the display tidy on completion.
                    for kept, result in tqdm(results_iter, desc=f"  {name}",
                                             unit="doc", leave=False):
                        local_in += 1
                        if kept:
                            kept_w.write(result)
                            local_kept += 1
                        else:
                            result["__dq_rejections"] = [{"filter": "extraction", "rule": "extraction_failed"}]
                            rej_w.write(result)
                            local_rej += 1
            except BaseException:
                for p in (kept_tmp, rej_tmp):
                    if p.exists():
                        try: p.unlink()
                        except OSError: pass
                raise

            # Atomic rename → final shard
            (kept_dir / name).unlink(missing_ok=True)
            (rej_dir / name).unlink(missing_ok=True)
            kept_tmp.rename(kept_dir / name)
            rej_tmp.rename(rej_dir / name)

            mark_shard_done(stage_output, name, {
                "input": local_in, "kept": local_kept, "rejected": local_rej,
            })

            stats.input_count += local_in
            stats.output_count += local_kept
            stats.rejected_count += local_rej
            if local_rej:
                stats.reject_reasons["extraction_failed"] = (
                    stats.reject_reasons.get("extraction_failed", 0) + local_rej
                )

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
    stage_output = engine.stage_dir("stage3_curated")
    kept_dir = stage_output / "kept"
    rejected_dir = stage_output / "rejected"
    filtered_dir = stage_output / "_filtered"  # intermediate after per-shard filter pass
    filtered_dir.mkdir(parents=True, exist_ok=True)

    with PhaseTimer(stats):
        # 3a. Heuristic filters — shard-by-shard with resume markers
        _run_filter_per_shard(engine, input_dir, filtered_dir, stats)

        # Collect survivors for cross-shard phases (dedup, contamination, quality)
        docs = list(read_shards(filtered_dir))
        stats.input_count = max(stats.input_count, len(docs) + stats.rejected_count)
        logger.info("Curation: %d docs after filter pass", len(docs))

        all_rejected: list[dict] = []

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


def _run_filter_per_shard(engine, input_dir, filtered_dir, stats):
    """Run heuristic filters per-shard with resume markers.

    Writes surviving docs to `filtered_dir/shard-NNNNN.jsonl.zst` (no
    kept/rejected split — rejected are stored under `_rejected/` for audit).
    Drops markers in `filtered_dir/.done/`. Re-running skips completed shards.
    """
    from tqdm import tqdm

    filter_configs = [{"name": fc.name, "params": fc.params}
                      for fc in engine.config.filters if fc.enabled]
    shard_files = list_shards(input_dir)
    pending = [p for p in shard_files if not is_shard_done(filtered_dir, p.name)]
    skipped = len(shard_files) - len(pending)
    if skipped:
        logger.info("curate.filter: resuming — %d/%d shards done", skipped, len(shard_files))
    logger.info("curate.filter: %d input shards, %d to process", len(shard_files), len(pending))

    # Per-shard rejected files live alongside _filtered/
    rejected_dir = filtered_dir.parent / "_filter_rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    shard_bar = tqdm(total=len(pending), desc="curate filters", unit="shard",
                    leave=True, disable=not pending)
    for shard_path in pending:
        name = shard_path.name
        shard_bar.set_postfix_str(name)
        chunk = list(read_shard(shard_path))
        kept, rejected, _ = _filter_chunk(chunk, filter_configs, engine.config.text_field)

        kept_tmp = filtered_dir / (name + ".part")
        rej_tmp = rejected_dir / (name + ".part")
        try:
            with SingleShardWriter(kept_tmp) as w:
                for d in kept:
                    w.write(d)
            with SingleShardWriter(rej_tmp) as w:
                for d in rejected:
                    w.write(d)
        except BaseException:
            for p in (kept_tmp, rej_tmp):
                if p.exists():
                    try: p.unlink()
                    except OSError: pass
            raise
        (filtered_dir / name).unlink(missing_ok=True)
        (rejected_dir / name).unlink(missing_ok=True)
        kept_tmp.rename(filtered_dir / name)
        rej_tmp.rename(rejected_dir / name)
        mark_shard_done(filtered_dir, name, {
            "input": len(chunk), "kept": len(kept), "rejected": len(rejected),
        })
        # stats accumulator
        for doc in rejected:
            for r in doc.get("__dq_rejections", []):
                key = f"{r.get('filter', '?')}.{r.get('rule', '?')}"
                stats.reject_reasons[key] = stats.reject_reasons.get(key, 0) + 1
        stats.rejected_count += len(rejected)
        shard_bar.update(1)
    shard_bar.close()


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
    """Quality scoring substep.

    Supports two methods:
      method: "llm"        — LLMJudge (slow, accurate, expensive)
      method: "classifier" — local FineWeb-Edu classifier (fast, cheap, good-enough for scale)

    Config keys (quality_scoring):
        enabled:        bool
        method:         "llm" | "classifier"
        sample_size:    0 = score all docs; N = score N random docs, others pass through
        min_quality:    "high" | "low" — drops docs judged "low"   (LLM mode)
        min_score:      float  — drops docs with score < min_score (classifier mode, 0..5)
        workers:        int    — parallel LLM API calls (LLM mode only, default 8)
        max_chars:      int    — truncate before scoring (default 3000)
        model:          str    — HF model name for classifier (classifier mode, default fineweb-edu)
        device:         "auto" | "cpu" | "cuda" (classifier mode)
        batch_size:     int    — classifier batch size (default 32)

    Returns:
        (kept, rejected): both are lists of doc dicts.
    """
    method = quality_cfg.get("method", "llm")
    sample_size = int(quality_cfg.get("sample_size", quality_cfg.get("llm_samples", 0)))
    min_quality = quality_cfg.get("min_quality", "high")
    workers = int(quality_cfg.get("workers", 8))
    max_chars = int(quality_cfg.get("max_chars", 3000))

    if not docs:
        return docs, []

    if method == "classifier":
        return _classifier_score(docs, quality_cfg, max_chars, sample_size=int(quality_cfg.get("sample_size", 0)))

    if method != "llm":
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


def _classifier_score(docs, quality_cfg, max_chars, sample_size):
    """Classifier-based quality scoring (FineWeb-Edu by default)."""
    model_name = quality_cfg.get("model") or "HuggingFaceFW/fineweb-edu-classifier"
    device = quality_cfg.get("device", "auto")
    batch_size = int(quality_cfg.get("batch_size", 32))
    min_score = float(quality_cfg.get("min_score", 3.0))

    try:
        from dq.model_filters.fineweb_edu_classifier import get_classifier
        clf = get_classifier(model_name=model_name, device=device, batch_size=batch_size)
    except Exception as e:
        logger.warning("Classifier unavailable, skipping quality scoring: %s", e)
        return docs, []

    # Subsample if configured
    import random
    if sample_size := int(sample_size or 0):
        if sample_size < len(docs):
            to_score = set(random.sample(range(len(docs)), sample_size))
        else:
            to_score = set(range(len(docs)))
    else:
        to_score = set(range(len(docs)))

    # Score in order so text indices match back
    ordered_texts: list[str] = []
    ordered_idx: list[int] = []
    for i, doc in enumerate(docs):
        if i in to_score:
            ordered_idx.append(i)
            ordered_texts.append((doc.get("text") or "")[: int(quality_cfg.get("max_chars", max_chars))])

    logger.info("Classifier %s: scoring %d/%d docs (batch=%d)", model_name, len(ordered_texts), len(docs), batch_size)
    scores = clf.score_batch(ordered_texts) if ordered_texts else []

    score_by_idx: dict[int, float] = {idx: s for idx, s in zip(ordered_idx, scores)}

    kept, rejected = [], []
    for i, doc in enumerate(docs):
        if i not in score_by_idx:
            kept.append(doc)
            continue
        s = score_by_idx[i]
        doc.setdefault("quality_scores", {})[model_name] = round(s, 3)
        if s < min_score:
            doc["__dq_rejections"] = doc.get("__dq_rejections", []) + [{
                "filter": "classifier",
                "rule": f"below_{min_score}",
                "value": round(s, 3),
                "threshold": min_score,
            }]
            rejected.append(doc)
        else:
            kept.append(doc)

    logger.info("Classifier: kept %d rejected %d (min_score=%.2f)",
                len(kept), len(rejected), min_score)
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
    """Sort, shard, and write manifest.

    Output format is controlled by `packaging.format` in the yaml config:
        - "jsonl_zst" (default): classic zstd-JSONL shards + manifest.json
        - "webdataset":         tar shards with text/json/figures per sample
    """
    from dq.shared.shard import ShardWriter, write_manifest

    stats = PhaseStats(phase="packaging")
    pkg_cfg = engine.extra_config.get("phase5", {})
    sort_field = pkg_cfg.get("sort_by", "id")
    fmt = engine.extra_config.get("packaging", {}).get("format", "jsonl_zst")

    input_dir = engine.stage_dir("stage3_curated", "kept")
    final_dir = engine.stage_dir("stage4_final")

    with PhaseTimer(stats):
        docs = list(read_shards(input_dir))
        stats.input_count = len(docs)
        docs.sort(key=lambda d: d.get(sort_field, ""))

        if fmt == "webdataset":
            from dq.shared.webdataset import WebDatasetWriter
            samples_per_shard = int(engine.extra_config.get("packaging", {}).get(
                "samples_per_shard", 1000))
            with WebDatasetWriter(final_dir, samples_per_shard=samples_per_shard) as w:
                for doc in docs:
                    w.write(doc)
                    stats.output_count += 1
            logger.info("Packaging: wrote WebDataset shards to %s", final_dir)
        else:
            with ShardWriter(final_dir, target_bytes=engine.shard_target_bytes) as w:
                for doc in docs:
                    w.write(doc)
                    stats.output_count += 1
            version = engine.extra_config.get("version", engine.version)
            write_manifest(final_dir, w.shard_info, version=version)

    stats.rejected_count = 0
    return stats


# ── Timer helper ──


