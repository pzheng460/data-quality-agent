"""Tests for C4 filters."""

from dq.stages.curation.filters import ensure_registered; ensure_registered()
from dq.stages.curation.filters.c4 import C4Filter


class TestC4Filter:
    def test_good_doc_passes(self, good_docs):
        f = C4Filter()
        for doc in good_docs:
            keep, info = f.filter(doc)
            assert keep, f"Good doc should pass: {info}"

    def test_lorem_ipsum_dropped(self):
        f = C4Filter()
        doc = {"text": "This is a lorem ipsum document. It should be dropped. Definitely removed."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "lorem_ipsum"

    def test_curly_brace_off_by_default(self):
        f = C4Filter(remove_curly_brace=False)
        doc = {"text": "This has a { brace. And some more text. It should pass by default. Here is another sentence. And one more for good measure."}
        keep, info = f.filter(doc)
        # Should pass because curly brace removal is off
        assert keep

    def test_curly_brace_when_enabled(self):
        f = C4Filter(remove_curly_brace=True)
        doc = {"text": "This has a { brace. And some more text. It should be dropped."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "curly_brace"

    def test_too_few_sentences(self):
        f = C4Filter(min_sentences=3)
        doc = {"text": "Only one sentence here."}
        keep, info = f.filter(doc)
        assert not keep
        assert info["reason"] == "too_few_sentences"

    def test_javascript_lines_removed(self):
        f = C4Filter(min_sentences=1)
        doc = {"text": "Good content here.\nEnable javascript to continue.\nMore good content here."}
        keep, info = f.filter(doc)
        assert keep
        # The javascript line should be removed
        assert "javascript" not in doc["text"].lower()

    def test_policy_lines_removed(self):
        f = C4Filter(min_sentences=1)
        doc = {"text": "Normal content here.\nPlease accept our terms of use.\nMore content here."}
        keep, info = f.filter(doc)
        assert keep
        assert "terms of use" not in doc["text"].lower()

    def test_no_terminal_punct_lines_removed(self):
        f = C4Filter(min_sentences=1, remove_no_terminal_punct=True)
        doc = {"text": "This has punctuation.\nThis does not\nAnother sentence here."}
        keep, info = f.filter(doc)
        assert keep
        # "This does not" should be removed
        assert "This does not" not in doc["text"]
