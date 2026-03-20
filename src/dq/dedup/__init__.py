"""Deduplication modules."""

from dq.dedup.exact import ExactDedup
from dq.dedup.minhash import MinHashDedup

__all__ = ["ExactDedup", "MinHashDedup"]
