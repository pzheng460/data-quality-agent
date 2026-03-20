"""Exact deduplication using SHA256 hashing."""

import hashlib
import re
from collections import Counter
from typing import Iterator


def normalize_text(text: str) -> str:
    """Normalize text for hashing: lowercase, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def sha256_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ExactDedup:
    """Exact deduplication using SHA256 hashing.

    Two-pass approach:
    1. First pass: compute hashes for all documents.
    2. Second pass: yield only first occurrence of each hash.
    """

    def __init__(self, text_field: str = "text") -> None:
        self.text_field = text_field
        self.hash_counts: Counter[str] = Counter()
        self.total_docs: int = 0
        self.unique_docs: int = 0
        self.duplicate_docs: int = 0

    def dedup(self, docs: list[dict]) -> Iterator[dict]:
        """Deduplicate a list of documents. Yields unique documents."""
        seen: set[str] = set()
        self.hash_counts = Counter()
        self.total_docs = 0
        self.unique_docs = 0

        for doc in docs:
            self.total_docs += 1
            text = doc.get(self.text_field, "")
            h = sha256_hash(text)
            self.hash_counts[h] += 1

            if h not in seen:
                seen.add(h)
                self.unique_docs += 1
                yield doc

        self.duplicate_docs = self.total_docs - self.unique_docs

    def stats(self) -> dict:
        """Return dedup statistics."""
        return {
            "method": "exact_sha256",
            "total_docs": self.total_docs,
            "unique_docs": self.unique_docs,
            "duplicate_docs": self.duplicate_docs,
            "dedup_rate": round(self.duplicate_docs / self.total_docs, 4) if self.total_docs > 0 else 0.0,
        }
