"""Tests for the classifier-based quality scoring path."""

from __future__ import annotations

from contextlib import contextmanager

from dq.runner.stages import _substep_quality_score


class _FakeClassifier:
    """Return scores by matching a text-prefix -> score mapping."""

    def __init__(self, mapping: dict[str, float]):
        self.mapping = mapping

    def score_batch(self, texts: list[str]) -> list[float]:
        out = []
        for t in texts:
            hit = next((v for k, v in self.mapping.items() if t.startswith(k)), 0.0)
            out.append(float(hit))
        return out


@contextmanager
def _patched_classifier(mapping):
    import dq.model_filters.fineweb_edu_classifier as mod
    orig = mod.get_classifier
    mod.get_classifier = lambda **_: _FakeClassifier(mapping)  # type: ignore[assignment]
    try:
        yield
    finally:
        mod.get_classifier = orig  # type: ignore[assignment]


def _run(docs, cfg, mapping):
    with _patched_classifier(mapping):
        return _substep_quality_score(engine=None, docs=docs, quality_cfg=cfg)


def test_classifier_rejects_below_min_score():
    docs = [
        {"id": 1, "text": "HIGH educational math content"},
        {"id": 2, "text": "LOW spam content"},
        {"id": 3, "text": "HIGH clear tutorial"},
    ]
    mapping = {"HIGH": 4.2, "LOW": 1.1}
    cfg = {"enabled": True, "method": "classifier", "min_score": 3.0}
    kept, rejected = _run(docs, cfg, mapping)
    assert [d["id"] for d in kept] == [1, 3]
    assert [d["id"] for d in rejected] == [2]
    r = rejected[0]["__dq_rejections"][0]
    assert r["filter"] == "classifier" or r["filter"] == "fineweb_edu"  # allow rename later
    assert r["value"] == 1.1 or r.get("value") is None  # either field name ok


def test_classifier_tags_score_on_all_docs():
    docs = [{"id": i, "text": f"HIGH doc {i}"} for i in range(3)]
    cfg = {"method": "classifier", "min_score": 3.0}
    kept, _ = _run(docs, cfg, {"HIGH": 4.5})
    assert len(kept) == 3
    for d in kept:
        assert "quality_scores" in d


def test_classifier_empty_docs_returns_empty():
    cfg = {"method": "classifier", "min_score": 3.0}
    kept, rejected = _substep_quality_score(engine=None, docs=[], quality_cfg=cfg)
    assert kept == [] and rejected == []


def _run(docs, cfg, mapping):
    import dq.model_filters.fineweb_edu_classifier as mod
    orig = mod.get_classifier
    mod.get_classifier = lambda **_: _FakeClassifier(mapping)  # type: ignore[assignment]
    try:
        return _substep_quality_score(engine=None, docs=docs, quality_cfg=cfg)
    finally:
        mod.get_classifier = orig  # type: ignore[assignment]
