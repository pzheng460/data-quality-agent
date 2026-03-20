"""DEITA Repr Filter — embedding-based diversity selection for SFT data.

Uses sentence embeddings to ensure diversity by removing near-duplicate
instructions and keeping the highest-scoring sample from each cluster.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_sentence_transformers_available = None


def _check_sentence_transformers() -> bool:
    global _sentence_transformers_available
    if _sentence_transformers_available is None:
        try:
            import sentence_transformers  # noqa: F401
            _sentence_transformers_available = True
        except ImportError:
            _sentence_transformers_available = False
    return _sentence_transformers_available


class DiversityFilter:
    """Select diverse training samples using embedding-based clustering.

    This is a batch filter — it needs all documents at once to compute
    pairwise similarities. It removes near-duplicate instructions, keeping
    the highest-scoring sample from each cluster.

    Args:
        model_name: Sentence-transformers model for embeddings.
        threshold: Cosine similarity threshold for near-duplicate detection.
        batch_size: Batch size for embedding computation.
        instruction_field: Document field containing the instruction text.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        threshold: float = 0.95,
        batch_size: int = 32,
        instruction_field: str = "instruction",
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.instruction_field = instruction_field
        self._model = None
        self._available = _check_sentence_transformers()

        if not self._available:
            logger.warning("sentence-transformers not installed — DiversityFilter will keep all docs. "
                           "Install with: pip install sentence-transformers")

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return
        if not self._available:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
        logger.info("Loaded embedding model %s", self.model_name)

    def _compute_embeddings(self, texts: list[str]):
        """Compute embeddings for a list of texts."""
        import numpy as np
        self._load_model()
        if self._model is None:
            return np.zeros((len(texts), 1))
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _doc_score(self, doc: dict) -> float:
        """Compute composite score for ranking within a cluster."""
        complexity = doc.get("complexity_score", 3.0)
        quality = doc.get("quality_score", 3.0)
        # Treat -1 (skipped) as neutral
        if complexity < 0:
            complexity = 3.0
        if quality < 0:
            quality = 3.0
        return complexity * quality

    def filter_batch(self, docs: list[dict]) -> list[dict]:
        """Filter a batch of documents for diversity.

        Args:
            docs: List of document dicts. Should have `instruction` field
                  and optionally `complexity_score` / `quality_score`.

        Returns:
            Filtered list with near-duplicates removed, keeping the
            highest-scoring document from each cluster.
        """
        if not docs:
            return []

        if not self._available:
            logger.warning("DiversityFilter: sentence-transformers not available, returning all docs")
            return docs

        import numpy as np

        texts = [doc.get(self.instruction_field, "") for doc in docs]
        embeddings = self._compute_embeddings(texts)

        n = len(docs)
        # Track which docs to keep (not yet merged into another cluster)
        keep_mask = [True] * n
        # Map each doc to its cluster representative
        cluster_rep: dict[int, int] = {i: i for i in range(n)}

        # Compute pairwise cosine similarity for near-duplicate detection
        # Since embeddings are normalized, cosine sim = dot product
        sim_matrix = np.dot(embeddings, embeddings.T)

        # Process pairs by similarity (highest first)
        for i in range(n):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n):
                if not keep_mask[j]:
                    continue
                if sim_matrix[i, j] >= self.threshold:
                    # Near-duplicate found — keep the one with higher score
                    score_i = self._doc_score(docs[i])
                    score_j = self._doc_score(docs[j])
                    if score_j > score_i:
                        keep_mask[i] = False
                        break
                    else:
                        keep_mask[j] = False

        result = [doc for doc, keep in zip(docs, keep_mask) if keep]
        logger.info("DiversityFilter: %d → %d docs (removed %d near-duplicates)",
                     n, len(result), n - len(result))
        return result
