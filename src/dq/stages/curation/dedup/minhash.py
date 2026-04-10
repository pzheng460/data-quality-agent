"""MinHash LSH near-deduplication.

Uses datasketch library with FineWeb parameters:
- 5-gram shingling
- 112 hash functions (permutations)
- 14 bands x 8 rows
- Jaccard threshold ~0.75
"""

from typing import Iterator

from datasketch import MinHash, MinHashLSH

from dq.utils.tokenizer import char_ngrams


class MinHashDedup:
    """MinHash LSH near-deduplication.

    Two-pass approach:
    1. First pass: build LSH index from all documents.
    2. Second pass: yield one document per cluster.
    """

    def __init__(
        self,
        text_field: str = "text",
        num_perm: int = 112,
        bands: int = 14,
        rows: int = 8,
        ngram_size: int = 5,
        threshold: float | None = None,
    ) -> None:
        self.text_field = text_field
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows
        self.ngram_size = ngram_size
        # Threshold is approximately (1/bands)^(1/rows)
        self.threshold = threshold or (1.0 / bands) ** (1.0 / rows)
        self.total_docs: int = 0
        self.unique_docs: int = 0
        self.duplicate_docs: int = 0

    def _make_minhash(self, text: str) -> MinHash:
        """Create a MinHash signature from text."""
        m = MinHash(num_perm=self.num_perm)
        shingles = char_ngrams(text.lower(), self.ngram_size)
        for s in shingles:
            m.update(s.encode("utf-8"))
        return m

    def dedup(self, docs: list[dict]) -> Iterator[dict]:
        """Deduplicate documents using MinHash LSH.

        Yields one document per near-duplicate cluster.
        """
        lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm,
        )

        # First pass: insert all docs into LSH
        minhashes: list[MinHash] = []
        for i, doc in enumerate(docs):
            text = doc.get(self.text_field, "")
            mh = self._make_minhash(text)
            minhashes.append(mh)
            try:
                lsh.insert(str(i), mh)
            except ValueError:
                # Duplicate key — already inserted a near-duplicate
                pass

        # Second pass: find clusters, keep first in each
        seen: set[int] = set()
        self.total_docs = len(docs)
        self.unique_docs = 0

        for i, doc in enumerate(docs):
            if i in seen:
                continue
            # Query for near-duplicates
            result = lsh.query(minhashes[i])
            cluster = {int(r) for r in result}
            # Mark all cluster members as seen
            seen.update(cluster)
            # Yield the first one
            self.unique_docs += 1
            yield doc

        self.duplicate_docs = self.total_docs - self.unique_docs

    def stats(self) -> dict:
        """Return dedup statistics."""
        return {
            "method": "minhash_lsh",
            "num_perm": self.num_perm,
            "bands": self.bands,
            "rows": self.rows,
            "ngram_size": self.ngram_size,
            "threshold": round(self.threshold, 4),
            "total_docs": self.total_docs,
            "unique_docs": self.unique_docs,
            "duplicate_docs": self.duplicate_docs,
            "dedup_rate": round(self.duplicate_docs / self.total_docs, 4) if self.total_docs > 0 else 0.0,
        }
