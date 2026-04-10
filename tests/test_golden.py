"""Golden tests — regression tests using real arxiv papers.

Verifies that 5 known papers are correctly cleaned by the full pipeline.
Run after any change to ArxivFilter or pipeline phases.

Papers:
  1. 1706.03762 — Attention Is All You Need (equations + tables)
  2. 2310.06825 — Mistral 7B (wrapfigure tables, custom commands)
  3. 2303.08774 — GPT-4 Technical Report (long, many figures)
  4. 1312.6114  — VAE (math-heavy, short)
  5. 2203.15556 — Chinchilla (heavy tables + email addresses)
"""

import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "golden"
INPUT = GOLDEN_DIR / "input.jsonl"
EXPECTED = GOLDEN_DIR / "expected.json"


def _load_expected():
    with open(EXPECTED) as f:
        return json.load(f)


def _run_pipeline_on_input():
    """Run the full pipeline on golden input and return final docs."""
    from dq.stages.curation.filters import ensure_registered
    ensure_registered()
    from dq.stages.curation.filters.arxiv import _clean_text
    from dq.config import PipelineConfig
    from dq.pipeline import Pipeline

    config_path = Path(__file__).parent.parent / "configs" / "arxiv.yaml"
    config = PipelineConfig.from_yaml(str(config_path))
    # Disable filters that require optional dependencies (fasttext etc)
    config.filters = [f for f in config.filters if f.name not in ("language",)]

    with open(INPUT) as f:
        docs = [json.loads(line) for line in f if line.strip()]

    pipeline = Pipeline(config)
    kept = list(pipeline.run(iter(docs)))
    return kept, docs


@__import__("pytest").fixture(scope="module")
def pipeline_result():
    return _run_pipeline_on_input()


class TestGoldenPapers:
    def test_all_papers_loaded(self, pipeline_result):
        kept, raw = pipeline_result
        assert len(raw) == 5, f"Expected 5 input papers, got {len(raw)}"

    def test_all_kept(self, pipeline_result):
        """All 5 golden papers should pass the pipeline."""
        kept, raw = pipeline_result
        assert len(kept) == 5, f"Expected all 5 papers kept, got {len(kept)}"

    def test_content_not_empty(self, pipeline_result):
        """All kept papers should have substantial content after cleaning."""
        kept, _ = pipeline_result
        for doc in kept:
            assert len(doc["text"]) > 1000, f"{doc.get('id')}: text too short ({len(doc['text'])} chars)"

    def test_low_latex_residual(self, pipeline_result):
        """Cleaned text should have very few LaTeX commands outside math."""
        import re
        kept = pipeline_result[0]
        for doc in kept:
            text = doc["text"]
            no_math = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
            no_math = re.sub(r'\$[^$]+?\$', '', no_math)
            cmds = re.findall(r'\\[a-zA-Z]{2,}', no_math)
            ratio = len(cmds) / max(len(no_math), 1)
            assert ratio < 0.005, f"{doc.get('id')}: LaTeX residual {ratio:.4f} ({len(cmds)} cmds)"

    def test_no_figure_placeholders(self, pipeline_result):
        kept = pipeline_result[0]
        for doc in kept:
            assert "[Figure]" not in doc["text"], f"{doc.get('id')}: [Figure] placeholder found"
            assert "[Table]" not in doc["text"], f"{doc.get('id')}: [Table] placeholder found"

    def test_no_latex_junk(self, pipeline_result):
        """No toprule/midrule/bottomrule or orphaned braces."""
        kept = pipeline_result[0]
        for doc in kept:
            text = doc["text"]
            for junk in ["toprule", "midrule", "bottomrule", "tabcolsep", "textwidth"]:
                assert junk not in text, f"{doc.get('id')}: '{junk}' found in cleaned text"

    def test_expected_results_match(self):
        """Cross-check against saved golden expectations."""
        expected = _load_expected()
        kept, _ = _run_pipeline_on_input()
        kept_ids = {doc.get("metadata", {}).get("arxiv_id") or doc.get("id") for doc in kept}

        for exp in expected:
            if exp["status"] == "kept":
                assert exp["arxiv_id"] in kept_ids, \
                    f"{exp['arxiv_id']} ({exp.get('title')}) expected kept but was rejected"
