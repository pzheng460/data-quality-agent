"""Phase implementations for the production cleaning pipeline.

phase1_parse:          Raw data -> dq schema, basic validation
phase2_filter:         Quality filtering via Pipeline (parallel)
phase3_dedup:          Version + exact + MinHash dedup
phase4_contamination:  N-gram contamination detection
phase5_package:        Sort, shard, manifest
"""

from __future__ import annotations

import logging
import os
import re
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from dq.runner.shard import ShardWriter, read_shards
from dq.runner.stats import PhaseStats, PhaseTimer

if TYPE_CHECKING:
    from dq.runner.engine import PhaseEngine

logger = logging.getLogger(__name__)


# ── Shared regexes ──────────────────────────────────────────────────

_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+(?:\{[^}]*\})*")
_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(?!\s)[^$]+?\$(?!\$)")
_SECTION_RE = re.compile(r"^#+\s+", re.MULTILINE)
_ABSTRACT_RE = re.compile(r"(?:^|\n)#+\s*abstract", re.IGNORECASE)


def _latex_residual_frac(text: str) -> float:
    """Fraction of non-math text that is LaTeX commands."""
    no_math = _DISPLAY_MATH_RE.sub("", text)
    no_math = _INLINE_MATH_RE.sub("", no_math)
    if not no_math:
        return 0.0
    latex_chars = sum(len(m.group()) for m in _LATEX_CMD_RE.finditer(no_math))
    return latex_chars / len(no_math)


def _structural_checks(text: str) -> dict:
    """Compute structural check fields for arxiv documents."""
    return {
        "has_title": text.lstrip().startswith("#"),
        "has_abstract": bool(_ABSTRACT_RE.search(text)),
        "has_sections": bool(_SECTION_RE.search(text)),
        "num_sections": len(_SECTION_RE.findall(text)),
        "num_equations_display": text.count("$$") // 2,
        "num_equations_inline": len(_INLINE_MATH_RE.findall(text)),
        "num_code_blocks": text.count("```") // 2,
    }


# ── Phase 1: Parse ──────────────────────────────────────────────────


def phase1_parse(engine: PhaseEngine) -> PhaseStats:
    """Parse raw input -> dq schema with validation."""
    stats = PhaseStats(phase="phase1_parse")

    phase1_cfg = engine.arxiv_config.get("phase1", {})
    max_residual = phase1_cfg.get("max_latex_residual", 0.20)
    preview_len = phase1_cfg.get("preview_length", 500)

    kept_dir = engine.stage_dir("stage1_parsed", "kept")
    rejected_dir = engine.stage_dir("stage1_parsed", "rejected")

    with PhaseTimer(stats), \
         ShardWriter(kept_dir, target_bytes=engine.shard_target_bytes) as kept_w, \
         ShardWriter(rejected_dir, target_bytes=engine.shard_target_bytes) as rej_w:

        for doc in engine.iter_input():
            stats.input_count += 1
            text = doc.get("text", "")

            out = dict(doc)
            out.setdefault("id", f"arxiv_{stats.input_count}")
            out.setdefault("source", "arxiv")
            out["structural_checks"] = _structural_checks(text)
            out["__raw_preview_head"] = text[:preview_len]
            out["__raw_preview_tail"] = text[-preview_len:] if len(text) > preview_len else text

            # Rejection checks
            reject_rule = None
            if not text.strip():
                reject_rule = "empty_text"
            elif len(text.split()) < 10:
                reject_rule = "too_short"
            else:
                frac = _latex_residual_frac(text)
                if frac > max_residual:
                    reject_rule = "latex_residual_high"

            if reject_rule:
                out.setdefault("trace", {})["phase1_parse"] = {"status": "rejected", "rule": reject_rule}
                out["__dq_rejections"] = [{"filter": "phase1", "rule": reject_rule}]
                rej_w.write(out)
                stats.rejected_count += 1
                stats.reject_reasons[reject_rule] = stats.reject_reasons.get(reject_rule, 0) + 1
            else:
                out.setdefault("trace", {})["phase1_parse"] = {"status": "ok"}
                kept_w.write(out)
                stats.output_count += 1

    return stats


# ── Phase 2: Filter ──────────────────────────────────────────────────


def _filter_chunk(
    chunk: list[dict],
    filter_configs: list[dict],
    text_field: str,
) -> dict:
    """Worker function for parallel filtering. Returns kept/rejected docs + stats."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    from dq.filters import ensure_registered
    ensure_registered()
    from dq.pipeline import get_filter_class

    filters = []
    for fc in filter_configs:
        cls = get_filter_class(fc["name"])
        filters.append(cls(text_field=text_field, **fc["params"]))

    kept: list[dict] = []
    rejected: list[dict] = []
    rule_counts: dict[str, int] = {}

    for doc in chunk:
        pass_all = True
        rejections: list[dict] = []
        for f in filters:
            ok, failures = f.filter_detailed(doc)
            if not ok:
                rejections.extend(failures)
                pass_all = False
            for fail in failures:
                key = f"{fail.get('filter', f.name)}.{fail.get('rule', 'unknown')}"
                rule_counts[key] = rule_counts.get(key, 0) + 1

        if pass_all:
            kept.append(doc)
        else:
            doc["__dq_rejections"] = rejections
            rejected.append(doc)

    return {"kept": kept, "rejected": rejected, "rule_counts": rule_counts}


def phase2_filter(engine: "PhaseEngine") -> "PhaseStats":
    """Phase 2: Quality filtering using the existing filter chain (parallel)."""
    stats = PhaseStats(phase="phase2_filter")

    filter_configs = []
    for fc in engine.config.filters:
        if fc.enabled:
            filter_configs.append({"name": fc.name, "params": fc.params})

    input_dir = engine.stage_dir("stage1_parsed", "kept")
    kept_dir = engine.stage_dir("stage2_filtered", "kept")
    rejected_dir = engine.stage_dir("stage2_filtered", "rejected")

    with PhaseTimer(stats):
        docs = list(read_shards(input_dir))
        stats.input_count = len(docs)
        logger.info("Phase 2: filtering %d documents", len(docs))

        # Run filtering
        if engine.workers > 1 and len(docs) >= engine.workers * 10:
            results = _run_parallel_filter(docs, filter_configs, engine.config.text_field, engine.workers)
        else:
            results = [_filter_chunk(docs, filter_configs, engine.config.text_field)]

        # Write results
        with ShardWriter(kept_dir, target_bytes=engine.shard_target_bytes) as kept_w, \
             ShardWriter(rejected_dir, target_bytes=engine.shard_target_bytes) as rej_w:
            for r in results:
                for doc in r["kept"]:
                    kept_w.write(doc)
                    stats.output_count += 1
                for doc in r["rejected"]:
                    rej_w.write(doc)
                    stats.rejected_count += 1
                for rule, count in r.get("rule_counts", {}).items():
                    stats.reject_reasons[rule] = stats.reject_reasons.get(rule, 0) + count

    return stats


def _run_parallel_filter(
    docs: list[dict],
    filter_configs: list[dict],
    text_field: str,
    workers: int,
) -> list[dict]:
    """Split docs into chunks and process in parallel."""
    from multiprocessing import get_context

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    chunk_size = max(1, (len(docs) + workers - 1) // workers)
    chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]

    worker_fn = partial(_filter_chunk, filter_configs=filter_configs, text_field="text")

    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        return pool.map(worker_fn, chunks)


# ── Phase 2.5: Quality Scoring (Nemotron-CC style) ──────────────────


def phase2b_quality_score(engine: "PhaseEngine") -> "PhaseStats":
    """Score documents using LLM judge and/or classifier ensemble.

    Follows Nemotron-CC approach:
    - Score every document with quality classifier(s)
    - Store score in doc metadata (quality_score: 0-5)
    - Optionally reject docs below a threshold
    - For LLM judge: sample N docs (API calls are expensive)

    This is an optional phase — skipped if no quality scoring is configured.
    """
    stats = PhaseStats(phase="phase2b_quality_score")
    quality_cfg = engine.arxiv_config.get("quality_scoring", {})

    if not quality_cfg.get("enabled", False):
        # Pass through all docs unchanged
        input_dir = engine.stage_dir("stage2_filtered", "kept")
        kept_dir = engine.stage_dir("stage2b_scored", "kept")
        with PhaseTimer(stats), \
             ShardWriter(kept_dir, target_bytes=engine.shard_target_bytes) as kw:
            for doc in read_shards(input_dir):
                kw.write(doc)
                stats.input_count += 1
                stats.output_count += 1
        return stats

    min_score = quality_cfg.get("min_score", 0)
    sample_size = quality_cfg.get("llm_samples", 0)
    method = quality_cfg.get("method", "llm")  # "llm" or "classifier"

    input_dir = engine.stage_dir("stage2_filtered", "kept")
    kept_dir = engine.stage_dir("stage2b_scored", "kept")
    rejected_dir = engine.stage_dir("stage2b_scored", "rejected")

    with PhaseTimer(stats), \
         ShardWriter(kept_dir, target_bytes=engine.shard_target_bytes) as kept_w, \
         ShardWriter(rejected_dir, target_bytes=engine.shard_target_bytes) as rej_w:

        docs = list(read_shards(engine.stage_dir("stage2_filtered", "kept")))
        stats.input_count = len(docs)

        if method == "llm" and sample_size > 0:
            # LLM judge on a sample (expensive, use for validation)
            import random
            sample_idx = set(random.sample(range(len(docs)), min(sample_size, len(docs))))
            try:
                from dq.judge import LLMJudge
                judge = LLMJudge()
                for i, doc in enumerate(docs):
                    if i in sample_idx:
                        result = judge.judge_text(doc.get("text", "")[:3000])
                        score = 5 if result.get("quality") == "high" else 2
                        doc.setdefault("quality_scores", {})["llm_judge"] = {
                            "score": score,
                            "quality": result.get("quality", "unknown"),
                            "failed_rules": result.get("failed_rules", []),
                        }
                    # Write all docs (scoring is metadata enrichment, not filtering)
                    if min_score > 0 and doc.get("quality_scores", {}).get("llm_judge", {}).get("score", 5) < min_score:
                        doc["__dq_rejections"] = [{"filter": "quality_score", "rule": "low_quality", "value": score}]
                        rej_w.write(doc)
                        stats.rejected_count += 1
                    else:
                        kept_w.write(doc)
                        stats.output_count += 1
            except Exception as e:
                logger.warning("LLM judge failed: %s. Passing all docs through.", e)
                for doc in docs:
                    kept_w.write(doc)
                    stats.output_count += 1
        else:
            # No scoring — pass all through with metadata tag
            for doc in docs:
                kept_w.write(doc)
                stats.output_count += 1

    stats.extra["method"] = method
    stats.extra["sample_size"] = sample_size
    return stats


# ── Phase 3: Dedup ──────────────────────────────────────────────────


def phase3_dedup(engine: PhaseEngine) -> PhaseStats:
    """Phase 3: Version dedup -> exact dedup -> MinHash dedup."""
    from dq.dedup.exact import ExactDedup
    from dq.dedup.minhash import MinHashDedup

    stats = PhaseStats(phase="phase3_dedup")
    phase3_cfg = engine.arxiv_config.get("phase3", {})
    minhash_cfg = engine.config.dedup.minhash

    # Read from scored stage if it exists, otherwise from filtered
    scored_dir = engine.stage_dir("stage2b_scored", "kept")
    filtered_dir = engine.stage_dir("stage2_filtered", "kept")
    input_dir = scored_dir if scored_dir.exists() else filtered_dir
    logger.info("Phase 3: reading from %s", input_dir)
    kept_dir = engine.stage_dir("stage3_dedup", "kept")
    rejected_dir = engine.stage_dir("stage3_dedup", "rejected")

    with PhaseTimer(stats):
        docs = list(read_shards(input_dir))
        stats.input_count = len(docs)
        logger.info("Phase 3: dedup on %d docs", len(docs))

        rejected_docs: list[dict] = []

        # 1. Version dedup: keep latest version per arxiv_id
        if phase3_cfg.get("version_dedup", True):
            by_aid: dict[str, list[dict]] = {}
            for doc in docs:
                aid = doc.get("metadata", {}).get("arxiv_id", doc.get("id", ""))
                by_aid.setdefault(aid, []).append(doc)

            kept_docs = []
            for aid, versions in by_aid.items():
                if len(versions) == 1:
                    kept_docs.append(versions[0])
                else:
                    versions.sort(key=lambda d: d.get("metadata", {}).get("version", "v0"))
                    kept_docs.append(versions[-1])
                    for v in versions[:-1]:
                        v["__dq_rejections"] = [{"filter": "dedup", "rule": "older_version"}]
                        rejected_docs.append(v)
            version_rejected = len(docs) - len(kept_docs)
            stats.reject_reasons["older_version"] = version_rejected
            docs = kept_docs

        # 2. Exact dedup
        exact = ExactDedup(text_field=engine.config.text_field)
        exact_kept = list(exact.dedup(docs))
        exact_removed = len(docs) - len(exact_kept)
        stats.reject_reasons["exact_duplicate"] = exact_removed
        docs = exact_kept

        # 3. MinHash dedup
        if engine.config.dedup.minhash.get("enabled", False):
            from dq.dedup.minhash import MinHashDedup
            mh = MinHashDedup(
                text_field=engine.config.text_field,
                num_perm=minhash_cfg.get("num_perm", 112),
                bands=minhash_cfg.get("bands", 14),
                rows=minhash_cfg.get("rows", 8),
                ngram_size=minhash_cfg.get("ngram_size", 5),
            )
            mh_kept = list(mh.dedup(docs))
            mh_removed = len(docs) - len(mh_kept)
            stats.reject_reasons["minhash_duplicate"] = mh_removed
            docs = mh_kept

        # Write results
        with ShardWriter(kept_dir, target_bytes=engine.shard_target_bytes) as kw, \
             ShardWriter(rejected_dir, target_bytes=engine.shard_target_bytes) as rw:
            for doc in docs:
                kw.write(doc)
                stats.output_count += 1
            stats.rejected_count = stats.input_count - stats.output_count

    return stats


# ── Phase 4: Contamination ──────────────────────────────────────────


def phase4_contamination(engine: PhaseEngine) -> PhaseStats:
    """Check for n-gram contamination against evaluation benchmarks."""
    from dq.contamination.ngram import NgramContaminationDetector, load_benchmark

    stats = PhaseStats(phase="phase4_contamination")
    phase4_cfg = engine.arxiv_config.get("phase4", {})
    ngram_size = phase4_cfg.get("ngram_size", 13)
    threshold = phase4_cfg.get("threshold", 0.8)
    benchmark_names = phase4_cfg.get("benchmarks", [])

    detector = NgramContaminationDetector(n=ngram_size, threshold=threshold)

    # Build benchmark indices
    for bm_name in benchmark_names:
        try:
            texts = load_benchmark(bm_name)
            detector.build_index(texts, benchmark_name=bm_name)
            logger.info("Loaded benchmark '%s': %d texts", bm_name, len(texts))
        except Exception as e:
            logger.warning("Skipping benchmark '%s': %s", bm_name, e)

    input_dir = engine.stage_dir("stage3_dedup", "kept")
    kept_dir = engine.stage_dir("stage4_contamination", "kept")
    rejected_dir = engine.stage_dir("stage4_contamination", "rejected")

    with PhaseTimer(stats), \
         ShardWriter(kept_dir, target_bytes=engine.shard_target_bytes) as kw, \
         ShardWriter(rejected_dir, target_bytes=engine.shard_target_bytes) as rw:

        for doc in read_shards(input_dir):
            stats.input_count += 1
            text = doc.get(engine.config.text_field, "")
            result = detector.check_contamination(text)

            if result.is_contaminated:
                bm = result.matched_benchmark or "unknown"
                doc["__dq_rejections"] = [{"filter": "contamination", "rule": f"contaminated_{bm}"}]
                doc.setdefault("trace", {})["phase4_contamination"] = {
                    "status": "rejected", "benchmark": bm,
                    "overlap_ratio": round(result.overlap_ratio, 4),
                }
                rw.write(doc)
                stats.rejected_count += 1
                key = f"contaminated_{bm}"
                stats.reject_reasons[key] = stats.reject_reasons.get(key, 0) + 1
            else:
                doc.setdefault("trace", {})["phase4_contamination"] = {"status": "ok"}
                kw.write(doc)
                stats.output_count += 1

    return stats


# ── Phase 5: Package ──────────────────────────────────────────────────


def phase5_package(engine: PhaseEngine) -> PhaseStats:
    """Sort docs by id, write final shards, and generate manifest."""
    from dq.runner.shard import write_manifest

    stats = PhaseStats(phase="phase5_package")
    phase5_cfg = engine.arxiv_config.get("phase5", {})
    sort_key = phase5_cfg.get("sort_by", "id")

    input_dir = engine.stage_dir("stage4_contamination", "kept")
    final_dir = engine.stage_dir("stage5_final")

    with PhaseTimer(stats):
        docs = list(read_shards(input_dir))
        stats.input_count = len(docs)
        docs.sort(key=lambda d: d.get(sort_key, ""))

        with ShardWriter(final_dir, target_bytes=engine.shard_target_bytes) as w:
            for doc in docs:
                w.write(doc)
                stats.output_count += 1

        version = engine.arxiv_config.get("version", "")
        write_manifest(final_dir, w.shard_info, version=version)

    stats.rejected_count = 0
    return stats
