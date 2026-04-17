"""Tests for the pipeline-level LLM quality-scoring substep."""

from __future__ import annotations

from contextlib import contextmanager

from dq.runner.stages import _substep_quality_score


class _FakeJudge:
    """Returns a canned judge result per doc prefix."""

    def __init__(self, mapping):
        self._mapping = mapping

    def judge_text(self, text):
        for prefix, res in self._mapping.items():
            if text.startswith(prefix):
                return res
        return {"quality": "high", "failed_rules": []}


@contextmanager
def _patched_judge(mapping):
    import dq.judge as _mod
    orig = _mod.LLMJudge
    _mod.LLMJudge = lambda *a, **kw: _FakeJudge(mapping)
    try:
        yield
    finally:
        _mod.LLMJudge = orig


def _run(docs, cfg, mapping):
    with _patched_judge(mapping):
        return _substep_quality_score(engine=None, docs=docs, quality_cfg=cfg)


def test_rejects_low_when_min_quality_high():
    docs = [
        {"id": 1, "text": "A good paper"},
        {"id": 2, "text": "B spam content"},
        {"id": 3, "text": "C excellent survey"},
    ]
    mapping = {
        "A": {"quality": "high", "failed_rules": []},
        "B": {"quality": "low", "failed_rules": ["originality"]},
        "C": {"quality": "high", "failed_rules": []},
    }
    kept, rejected = _run(docs, {"min_quality": "high", "workers": 1}, mapping)
    assert [d["id"] for d in kept] == [1, 3]
    assert [d["id"] for d in rejected] == [2]
    r = rejected[0]["__dq_rejections"][0]
    assert r["filter"] == "llm_judge"
    assert r["rule"] == "below_min_quality"
    assert r["failed_rules"] == ["originality"]


def test_min_quality_low_tags_but_keeps_all():
    docs = [{"id": 1, "text": "A"}, {"id": 2, "text": "B"}]
    mapping = {
        "A": {"quality": "high", "failed_rules": []},
        "B": {"quality": "low", "failed_rules": ["coherence"]},
    }
    kept, rejected = _run(docs, {"min_quality": "low", "workers": 1}, mapping)
    assert [d["id"] for d in kept] == [1, 2]
    assert rejected == []
    assert all("quality_scores" in d for d in kept)


def test_sample_size_only_scores_subset():
    docs = [{"id": i, "text": f"doc{i}"} for i in range(5)]
    mapping = {}  # everything defaults to high
    kept, rejected = _run(docs, {"sample_size": 2, "workers": 1}, mapping)
    assert len(kept) == 5 and rejected == []
    scored = [d for d in kept if "quality_scores" in d]
    assert len(scored) == 2


def test_noop_when_method_is_not_llm():
    docs = [{"text": "hello"}]
    kept, rejected = _run(docs, {"method": "classifier"}, {})
    assert kept == docs and rejected == []
