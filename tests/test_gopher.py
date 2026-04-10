"""Tests for Gopher quality and repetition filters."""

from dq.stages.curation.filters import ensure_registered; ensure_registered()
from dq.stages.curation.filters.gopher import GopherQualityFilter, GopherRepetitionFilter


class TestGopherQualityFilter:
    def test_good_doc_passes(self, good_docs):
        f = GopherQualityFilter()
        for doc in good_docs:
            keep, info = f.filter(doc)
            assert keep, f"Good doc should pass: {info}"

    def test_too_few_words(self):
        f = GopherQualityFilter(min_words=50)
        keep, info = f.filter({"text": "short text here"})
        assert not keep
        assert info["reason"] == "too_few_words"

    def test_too_many_words(self):
        f = GopherQualityFilter(min_words=1, max_words=10, min_stopwords=0, max_bullet_lines_ratio=1.0, max_ellipsis_lines_ratio=1.0)
        text = " ".join(["word"] * 20) + "."
        keep, info = f.filter({"text": text})
        assert not keep
        assert info["reason"] == "too_many_words"

    def test_low_alpha_ratio(self):
        f = GopherQualityFilter(min_words=1, min_stopwords=0, max_bullet_lines_ratio=1.0, max_ellipsis_lines_ratio=1.0)
        text = "12345 67890. " * 50
        keep, info = f.filter({"text": text})
        assert not keep
        assert info["reason"] == "low_alpha_ratio"

    def test_configurable_text_field(self):
        f = GopherQualityFilter(text_field="content", min_words=2, min_stopwords=0, max_bullet_lines_ratio=1.0, max_ellipsis_lines_ratio=1.0)
        doc = {"content": "Hello world this is a reasonably long sentence with enough words to pass the filter for sure definitely."}
        keep, info = f.filter(doc)
        assert keep

    def test_too_few_stopwords(self):
        f = GopherQualityFilter(min_words=5, min_stopwords=2, max_bullet_lines_ratio=1.0, max_ellipsis_lines_ratio=1.0)
        # Words with no English stopwords
        text = "xyz abc def ghi jkl mno pqr stu vwx yz1 yz2 yz3 yz4 yz5 yz6 yz7 yz8 yz9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 y21 y22 y23 y24 y25 y26 y27 y28 y29 y30 y31 y32 y33 y34 y35 y36 y37 y38 y39 y40 y41 y42 y43 y44 y45 y46 y47 y48 y49 y50."
        keep, info = f.filter({"text": text})
        assert not keep
        assert info["reason"] == "too_few_stopwords"


class TestGopherRepetitionFilter:
    def test_good_doc_passes(self, good_docs):
        f = GopherRepetitionFilter()
        for doc in good_docs:
            keep, info = f.filter(doc)
            assert keep, f"Good doc should pass: {info}"

    def test_high_2gram_repetition(self):
        f = GopherRepetitionFilter(max_top_2gram=0.10)
        # "hello world" repeated many times -> high 2-gram ratio
        text = "hello world " * 100
        keep, info = f.filter({"text": text})
        assert not keep
        assert "2gram" in info["reason"]

    def test_high_duplicate_lines(self):
        # Use diverse-enough lines that n-gram checks pass but dup line ratio triggers
        f = GopherRepetitionFilter(max_dup_line_ratio=0.20, max_top_2gram=1.0, max_top_3gram=1.0, max_top_4gram=1.0)
        lines = ["This is a duplicate line."] * 10 + ["This is unique."]
        text = "\n".join(lines)
        keep, info = f.filter({"text": text})
        assert not keep
        assert info["reason"] == "high_dup_line_ratio"

    def test_high_dup_ngram_frac(self):
        """Test Gopher's word-level duplicate n-gram detection."""
        # Relax top n-gram and line/para thresholds so we specifically test dup n-gram
        f = GopherRepetitionFilter(
            max_top_2gram=1.0, max_top_3gram=1.0, max_top_4gram=1.0,
            max_dup_line_ratio=1.0, max_dup_para_ratio=1.0,
            max_dup_5gram_frac=0.05,
        )
        # Highly repetitive word sequence
        text = " ".join(["the quick brown fox jumps"] * 50)
        keep, info = f.filter({"text": text})
        assert not keep
        assert "dup_5gram_frac" in info["reason"]

    def test_low_dup_ngram_passes(self):
        """Normal text should pass all dup n-gram checks."""
        f = GopherRepetitionFilter()
        text = (
            "The study of natural language processing has evolved significantly. "
            "Machine learning approaches have replaced many rule-based systems. "
            "Deep neural networks particularly excel at understanding context. "
            "Attention mechanisms revolutionized how models process sequences. "
            "Transfer learning allows models to leverage pre-trained knowledge."
        )
        keep, _ = f.filter({"text": text})
        assert keep
