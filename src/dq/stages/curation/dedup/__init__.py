"""Deduplication modules."""

from dq.stages.curation.dedup.exact import ExactDedup
from dq.stages.curation.dedup.minhash import MinHashDedup

__all__ = ["ExactDedup", "MinHashDedup"]
