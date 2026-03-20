"""IO utilities for reading and writing jsonl, parquet, and csv files."""

import csv
import json
import random
from pathlib import Path
from typing import Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Read documents from a JSONL file, yielding one dict per line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(docs: Iterator[dict], path: str | Path) -> int:
    """Write documents to a JSONL file. Returns count of docs written."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_parquet(path: str | Path) -> Iterator[dict]:
    """Read documents from a Parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    for batch in table.to_batches():
        for row in batch.to_pylist():
            yield row


def write_parquet(docs: Iterator[dict], path: str | Path) -> int:
    """Write documents to a Parquet file. Returns count of docs written."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = list(docs)
    if not rows:
        return 0
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path))
    return len(rows)


def read_csv(path: str | Path) -> Iterator[dict]:
    """Read documents from a CSV file."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def write_csv(docs: Iterator[dict], path: str | Path) -> int:
    """Write documents to a CSV file. Returns count of docs written."""
    rows = list(docs)
    if not rows:
        return 0
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_docs(path: str | Path) -> Iterator[dict]:
    """Auto-detect format and read documents."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from read_jsonl(path)
    elif suffix == ".parquet":
        yield from read_parquet(path)
    elif suffix == ".csv":
        yield from read_csv(path)
    else:
        # Default to jsonl
        yield from read_jsonl(path)


def count_lines(path: str | Path) -> int:
    """Fast line count for JSONL files (without parsing JSON)."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def sample_docs(
    path: str | Path,
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Reservoir sampling: randomly sample n docs from a file without loading all into memory.

    Uses reservoir sampling (Vitter's Algorithm R) — O(n) memory regardless of file size.
    Works with jsonl, parquet, csv.

    Args:
        path: Path to data file.
        n: Number of samples to draw.
        seed: Random seed for reproducibility.

    Returns:
        List of n randomly sampled documents (or all docs if file has fewer than n).
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []

    for i, doc in enumerate(read_docs(path)):
        if i < n:
            reservoir.append(doc)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = doc

    return reservoir


def write_docs(docs: Iterator[dict], path: str | Path) -> int:
    """Auto-detect format and write documents. Returns count written."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return write_jsonl(docs, path)
    elif suffix == ".parquet":
        return write_parquet(docs, path)
    elif suffix == ".csv":
        return write_csv(docs, path)
    else:
        return write_jsonl(docs, path)
