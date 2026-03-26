"""Production data cleaning pipeline.

Reads data from local / S3 paths, applies quality filters in parallel,
writes cleaned output. Supports JSONL and Parquet formats.

Uses multiprocessing for single-node parallelism. The architecture is
designed so a Ray Data or Spark backend can be swapped in by replacing
only the execution layer (``_run_parallel_filter``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── I/O ──────────────────────────────────────────────────────────────

def _read_docs(path: str, text_field: str = "text") -> list[dict]:
    """Read documents from a file.

    Supports:
    - JSONL (.jsonl / .json)
    - Parquet (.parquet)
    - Directories of Parquet files
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if p.suffix == ".parquet" or p.is_dir():
        import pyarrow.parquet as pq
        table = pq.read_table(str(p))
        return table.to_pylist()
    else:
        # JSONL
        docs = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs


def _write_docs(docs: list[dict], path: str):
    """Write documents to a file.

    Format inferred from extension:
    - .parquet → Parquet
    - .jsonl / .json / other → JSONL
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        if not docs:
            # Empty write
            pq.write_table(pa.table({}), str(p))
            return
        table = pa.Table.from_pylist(docs)
        pq.write_table(table, str(p))
    else:
        # JSONL
        with open(p, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")


# ── Filter worker ────────────────────────────────────────────────────

def _filter_chunk(
    chunk: list[dict],
    filter_configs: list[dict],
    text_field: str,
) -> list[dict]:
    """Process a chunk, returning only docs that pass all filters.

    Runs in a worker process. Loads filters and spacy once per process.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    from dq.filters import ensure_registered
    ensure_registered()
    from dq.pipeline import get_filter_class

    filters = []
    for fc in filter_configs:
        cls = get_filter_class(fc["name"])
        filters.append(cls(text_field=text_field, **fc["params"]))

    passing = []
    for doc in chunk:
        keep = True
        for f in filters:
            passed, _ = f.filter(doc)
            if not passed:
                keep = False
                break
        if keep:
            passing.append(doc)

    return passing


def _run_parallel_filter(
    docs: list[dict],
    filter_configs: list[dict],
    text_field: str,
    workers: int,
) -> list[dict]:
    """Apply filters in parallel using multiprocessing."""
    if not filter_configs:
        return docs

    chunk_size = max(1, (len(docs) + workers - 1) // workers)
    chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]

    worker_fn = partial(_filter_chunk, filter_configs=filter_configs, text_field=text_field)

    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.map(worker_fn, chunks)

    # Flatten
    return [doc for chunk in results for doc in chunk]


# ── Exact dedup ──────────────────────────────────────────────────────

def _exact_dedup(docs: list[dict], text_field: str = "text") -> list[dict]:
    """Single-pass exact dedup using SHA256."""
    seen: set[str] = set()
    unique = []
    for doc in docs:
        text = doc.get(text_field, "") or ""
        normalized = " ".join(text.lower().split())
        h = hashlib.sha256(normalized.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


# ── Main entry point ─────────────────────────────────────────────────

def _get_default_workers() -> int:
    cpus = os.cpu_count() or 1
    return max(1, min(cpus // 4, 16))


def run_cleaning(
    input_path: str,
    output_path: str,
    config,  # PipelineConfig
    text_field: str = "text",
    parallelism: int | None = None,
    dedup: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run data cleaning pipeline.

    Reads input, applies quality filters in parallel, optionally deduplicates,
    writes cleaned output.

    Args:
        input_path: Source file path (JSONL or Parquet).
        output_path: Destination file path.
        config: PipelineConfig with filter and dedup settings.
        text_field: Field name containing document text.
        parallelism: Number of parallel workers (None = auto).
        dedup: Whether to run exact dedup.

    Returns:
        Dict with stats: input_rows, output_rows, drop_rate, elapsed_s, throughput.
    """
    workers = parallelism or _get_default_workers()
    t0 = time.time()

    # 1. Read
    logger.info("Reading from %s ...", input_path)
    docs = _read_docs(input_path, text_field)
    input_rows = len(docs)
    logger.info("Input: %d rows", input_rows)

    # 2. Build serializable filter configs
    filter_configs = [
        {"name": fc.name, "params": fc.params}
        for fc in config.filters
        if fc.enabled
    ]

    # 3. Filter
    if filter_configs:
        filter_names = [fc["name"] for fc in filter_configs]
        logger.info("Filters: %s (%d workers)", ", ".join(filter_names), workers)

        if workers > 1 and input_rows >= workers * 10:
            docs = _run_parallel_filter(docs, filter_configs, text_field, workers)
        else:
            docs = _filter_chunk(docs, filter_configs, text_field)
    else:
        logger.warning("No filters enabled — passing all docs through.")

    # 4. Dedup
    if dedup and config.dedup and config.dedup.exact:
        before_dedup = len(docs)
        docs = _exact_dedup(docs, text_field)
        deduped = before_dedup - len(docs)
        if deduped > 0:
            logger.info("Dedup removed %d exact duplicates", deduped)

    # 5. Write
    output_rows = len(docs)
    logger.info("Writing %d docs to %s ...", output_rows, output_path)
    _write_docs(docs, output_path)

    elapsed = time.time() - t0
    drop_rate = 1.0 - (output_rows / input_rows) if input_rows > 0 else 0.0

    stats = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "dropped": input_rows - output_rows,
        "drop_rate": drop_rate,
        "elapsed_s": round(elapsed, 1),
        "throughput": round(input_rows / elapsed) if elapsed > 0 else 0,
    }
    logger.info(
        "Done: %d → %d rows (%.1f%% dropped) in %.1fs (%d docs/s)",
        input_rows, output_rows, drop_rate * 100, elapsed, stats["throughput"],
    )
    return stats
