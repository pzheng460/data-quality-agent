"""SemDeDup: Semantic deduplication using embeddings and clustering.

Phase 2 stub — not yet implemented.
"""


class SemanticDedup:
    """Semantic deduplication using sentence embeddings + clustering.

    Uses embedding models to find semantically similar documents,
    clusters them, and keeps representative samples.

    Requires: sentence-transformers
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "SemanticDedup is a Phase 2 feature. "
            "Install optional deps with: uv sync --extra model"
        )
