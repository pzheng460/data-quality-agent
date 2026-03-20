"""Tests for deduplication modules."""

from dq.dedup.exact import ExactDedup, sha256_hash
from dq.dedup.minhash import MinHashDedup


class TestExactDedup:
    def test_removes_exact_duplicates(self):
        docs = [
            {"text": "Hello world this is a test."},
            {"text": "Hello world this is a test."},
            {"text": "A different document entirely."},
        ]
        dedup = ExactDedup()
        result = list(dedup.dedup(docs))
        assert len(result) == 2
        assert dedup.stats()["duplicate_docs"] == 1

    def test_normalization(self):
        # Same content with different whitespace -> same hash
        h1 = sha256_hash("hello  world")
        h2 = sha256_hash("hello world")
        assert h1 == h2

    def test_case_normalization(self):
        h1 = sha256_hash("Hello World")
        h2 = sha256_hash("hello world")
        assert h1 == h2

    def test_no_duplicates(self):
        docs = [
            {"text": "Document one."},
            {"text": "Document two."},
            {"text": "Document three."},
        ]
        dedup = ExactDedup()
        result = list(dedup.dedup(docs))
        assert len(result) == 3
        assert dedup.stats()["duplicate_docs"] == 0

    def test_all_duplicates(self):
        docs = [{"text": "same"} for _ in range(5)]
        dedup = ExactDedup()
        result = list(dedup.dedup(docs))
        assert len(result) == 1

    def test_custom_text_field(self):
        docs = [
            {"content": "Hello world."},
            {"content": "Hello world."},
        ]
        dedup = ExactDedup(text_field="content")
        result = list(dedup.dedup(docs))
        assert len(result) == 1


class TestMinHashDedup:
    def test_removes_near_duplicates(self):
        # Two identical docs must be caught
        base = "The quick brown fox jumps over the lazy dog and runs through fields. " * 20
        docs = [
            {"text": base},
            {"text": base},  # exact duplicate — must be caught by LSH too
            {"text": "Completely different content about machine learning and AI research papers. " * 20},
        ]
        dedup = MinHashDedup(num_perm=128, ngram_size=5)
        result = list(dedup.dedup(docs))
        assert len(result) == 2

    def test_keeps_different_docs(self):
        docs = [
            {"text": "Alpha beta gamma delta epsilon. " * 20},
            {"text": "One two three four five six seven. " * 20},
            {"text": "Machine learning deep neural networks. " * 20},
        ]
        dedup = MinHashDedup(num_perm=128, ngram_size=5)
        result = list(dedup.dedup(docs))
        assert len(result) == 3

    def test_stats(self):
        docs = [{"text": f"Document number {i} with some content."} for i in range(5)]
        dedup = MinHashDedup()
        list(dedup.dedup(docs))
        stats = dedup.stats()
        assert stats["total_docs"] == 5
        assert stats["method"] == "minhash_lsh"
