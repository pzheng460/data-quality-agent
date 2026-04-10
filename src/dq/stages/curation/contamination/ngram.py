"""N-gram overlap detection for benchmark contamination.

Checks if training data overlaps with benchmark test sets using n-gram matching.
Based on decontamination methods from GPT-3, Llama, and various decontamination papers.
"""

from __future__ import annotations

import json
import logging
import re
import string
from pathlib import Path

from dq.stages.curation.contamination.report import (
    BenchmarkContamination,
    ContaminationReport,
    ContaminationResult,
)

logger = logging.getLogger(__name__)

# Punctuation translation table for normalization
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    """Normalize text: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Extract word-level n-grams as a set of tuples."""
    words = _normalize(text).split()
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


# Known benchmark loaders (name -> HuggingFace dataset config)
_BENCHMARK_CONFIGS: dict[str, dict] = {
    "mmlu": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "test",
        "text_fields": ["question"],
    },
    "hellaswag": {
        "path": "Rowan/hellaswag",
        "split": "validation",
        "text_fields": ["ctx"],
    },
    "arc": {
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "test",
        "text_fields": ["question"],
    },
    "truthfulqa": {
        "path": "truthfulqa/truthful_qa",
        "name": "multiple_choice",
        "split": "validation",
        "text_fields": ["question"],
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "text_fields": ["question"],
    },
    "humaneval": {
        "path": "openai/openai_humaneval",
        "split": "test",
        "text_fields": ["prompt"],
    },
}


def load_benchmark(name: str, split: str | None = None, text_field: str = "text", n: int = 0) -> list[str]:
    """Load benchmark texts by name, file path, or HuggingFace dataset ID.

    Args:
        name: One of:
            - Built-in name (mmlu, hellaswag, arc, truthfulqa, gsm8k, humaneval)
            - Local file path (text/jsonl)
            - HuggingFace dataset ID (e.g. 'cais/mmlu', 'allenai/ai2_arc')
        split: Override split (default: 'test' for built-in, 'train' for HF).
        text_field: Text field name for HF datasets (default: 'text').
        n: Max samples for HF datasets (0 = all).

    Returns:
        List of benchmark text strings.
    """
    # 1. Local file
    path = Path(name)
    if path.exists():
        return _load_benchmark_file(path)

    # 2. Built-in benchmark
    name_lower = name.lower()
    if name_lower in _BENCHMARK_CONFIGS:
        return _load_builtin_benchmark(name_lower, split)

    # 3. HuggingFace dataset ID (contains '/')
    if "/" in name:
        return _load_hf_benchmark(name, split or "test", text_field, n)

    raise ValueError(
        f"Unknown benchmark: {name}. "
        f"Available: {', '.join(_BENCHMARK_CONFIGS.keys())}, a HF dataset ID (org/name), or a local file path."
    )

def _load_builtin_benchmark(name: str, split: str | None = None) -> list[str]:
    """Load a built-in benchmark by name."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` library is required to load benchmarks. "
            "Install with: uv pip install 'dq[bench]'"
        )

    cfg = _BENCHMARK_CONFIGS[name]
    ds_split = split or cfg["split"]

    logger.info("Loading benchmark '%s' (split=%s)...", name, ds_split)

    kwargs: dict = {"path": cfg["path"], "split": ds_split}
    if "name" in cfg:
        kwargs["name"] = cfg["name"]

    ds = load_dataset(**kwargs)

    texts: list[str] = []
    text_fields = cfg["text_fields"]
    for item in ds:
        parts = [str(item.get(f, "")) for f in text_fields if item.get(f)]
        if parts:
            texts.append(" ".join(parts))

    logger.info("Loaded %d texts from benchmark '%s'.", len(texts), name)
    return texts


def _load_hf_benchmark(dataset_id: str, split: str, text_field: str, n: int) -> list[str]:
    """Load benchmark texts from any HuggingFace dataset via streaming."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` library is required to load HF benchmarks. "
            "Install with: uv pip install 'dq[bench]'"
        )

    logger.info("Loading HF benchmark '%s' (split=%s, streaming)...", dataset_id, split)
    ds = load_dataset(dataset_id, split=split, streaming=True)

    texts: list[str] = []
    for item in ds:
        text = item.get(text_field, "")
        if text:
            texts.append(str(text))
        if n > 0 and len(texts) >= n:
            break

    logger.info("Loaded %d texts from HF benchmark '%s'.", len(texts), dataset_id)
    return texts


def _load_benchmark_file(path: Path) -> list[str]:
    """Load benchmark texts from a text or JSONL file."""
    texts: list[str] = []
    suffix = path.suffix.lower()

    with open(path, encoding="utf-8") as f:
        if suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Try common text fields
                for field in ("text", "question", "prompt", "content", "input"):
                    if field in obj and obj[field]:
                        texts.append(str(obj[field]))
                        break
        else:
            # Plain text: one document per line
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    logger.info("Loaded %d texts from file '%s'.", len(texts), path)
    return texts


class NgramContaminationDetector:
    """Detect benchmark contamination via n-gram overlap.

    Builds a set of n-grams from benchmark texts and checks training
    documents for overlap. Fast, no model needed.

    Args:
        n: N-gram size (default 13, following GPT-3/Llama convention).
        threshold: Fraction of doc n-grams that must match to flag contamination.
    """

    def __init__(self, n: int = 13, threshold: float = 0.8) -> None:
        self.n = n
        self.threshold = threshold
        self._index: set[tuple[str, ...]] = set()
        self._benchmark_indices: dict[str, set[tuple[str, ...]]] = {}

    def build_index(self, benchmark_texts: list[str], benchmark_name: str = "default") -> None:
        """Build n-gram index from benchmark texts.

        Args:
            benchmark_texts: List of benchmark text strings.
            benchmark_name: Name for this benchmark index.
        """
        ngrams: set[tuple[str, ...]] = set()
        for text in benchmark_texts:
            ngrams.update(_extract_ngrams(text, self.n))

        self._benchmark_indices[benchmark_name] = ngrams
        # Merge into combined index
        self._index = set()
        for idx_ngrams in self._benchmark_indices.values():
            self._index.update(idx_ngrams)

        logger.info(
            "Built index for '%s': %d n-grams (%d total across all benchmarks).",
            benchmark_name,
            len(ngrams),
            len(self._index),
        )

    def check_contamination(self, doc: str, benchmark_name: str | None = None) -> ContaminationResult:
        """Check a single document for contamination.

        Args:
            doc: Document text.
            benchmark_name: Check against specific benchmark. None checks all.

        Returns:
            ContaminationResult with overlap details.
        """
        doc_ngrams = _extract_ngrams(doc, self.n)
        total = len(doc_ngrams)

        if total == 0:
            return ContaminationResult(
                is_contaminated=False,
                overlap_ratio=0.0,
                matched_ngrams=0,
                total_ngrams=0,
                matched_benchmark="",
                method="ngram",
            )

        # Check against specific or all benchmarks
        if benchmark_name and benchmark_name in self._benchmark_indices:
            index = self._benchmark_indices[benchmark_name]
            matched = len(doc_ngrams & index)
            overlap = matched / total
            return ContaminationResult(
                is_contaminated=overlap >= self.threshold,
                overlap_ratio=overlap,
                matched_ngrams=matched,
                total_ngrams=total,
                matched_benchmark=benchmark_name,
                method="ngram",
            )

        # Check all benchmarks, find best match
        best_match = ""
        best_overlap = 0.0
        best_matched = 0

        for bm_name, index in self._benchmark_indices.items():
            matched = len(doc_ngrams & index)
            overlap = matched / total
            if overlap > best_overlap:
                best_overlap = overlap
                best_matched = matched
                best_match = bm_name

        return ContaminationResult(
            is_contaminated=best_overlap >= self.threshold,
            overlap_ratio=best_overlap,
            matched_ngrams=best_matched,
            total_ngrams=total,
            matched_benchmark=best_match,
            method="ngram",
        )

    def scan_dataset(
        self,
        docs: list[dict],
        text_field: str = "text",
        benchmarks: dict[str, list[str]] | None = None,
        dataset_name: str = "dataset",
    ) -> ContaminationReport:
        """Scan a full dataset against multiple benchmarks.

        Args:
            docs: List of document dicts.
            text_field: Field containing text.
            benchmarks: Dict of {benchmark_name: texts}. If None, uses pre-built indices.
            dataset_name: Name for the report.

        Returns:
            ContaminationReport with per-benchmark stats.
        """
        # Build indices if benchmarks provided
        if benchmarks:
            for bm_name, bm_texts in benchmarks.items():
                self.build_index(bm_texts, bm_name)

        if not self._benchmark_indices:
            raise ValueError("No benchmark indices built. Call build_index() or pass benchmarks.")

        report = ContaminationReport(
            dataset_name=dataset_name,
            total_docs=len(docs),
            method="ngram",
        )

        # Initialize per-benchmark tracking
        bm_stats: dict[str, list[ContaminationResult]] = {
            name: [] for name in self._benchmark_indices
        }

        contaminated_set: set[int] = set()

        for i, doc in enumerate(docs):
            text = doc.get(text_field, "")
            if not text:
                continue

            for bm_name in self._benchmark_indices:
                result = self.check_contamination(text, benchmark_name=bm_name)
                bm_stats[bm_name].append(result)

                if result.is_contaminated and i not in contaminated_set:
                    contaminated_set.add(i)
                    if len(report.sample_contaminated) < 10:
                        report.sample_contaminated.append({
                            "index": i,
                            "text": text[:300],
                            "overlap_ratio": result.overlap_ratio,
                            "matched_benchmark": result.matched_benchmark,
                            "matched_ngrams": result.matched_ngrams,
                        })

        # Aggregate per-benchmark
        for bm_name, results in bm_stats.items():
            contaminated = sum(1 for r in results if r.is_contaminated)
            overlaps = [r.overlap_ratio for r in results]
            avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

            report.per_benchmark[bm_name] = BenchmarkContamination(
                benchmark_name=bm_name,
                total_docs=len(results),
                contaminated_docs=contaminated,
                contamination_rate=contaminated / len(results) if results else 0.0,
                avg_overlap=avg_overlap,
                sample_contaminated=[
                    s for s in report.sample_contaminated
                    if s.get("matched_benchmark") == bm_name
                ][:5],
            )

        report.contaminated_docs = len(contaminated_set)
        report.contamination_rate = (
            len(contaminated_set) / len(docs) if docs else 0.0
        )

        return report
