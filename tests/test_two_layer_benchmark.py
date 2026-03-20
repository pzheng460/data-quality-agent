"""Tests for the two-layer benchmark: heuristic filters + LLM scoring."""

import json
from unittest.mock import MagicMock, patch

import pytest

import dq.filters  # noqa: F401 — trigger filter registration
from dq.benchmark import (
    BenchmarkReport,
    DatasetResult,
    PretrainScores,
    SFTScores,
    _extract_sft_fields,
    _score_pretrain_docs,
    _score_sft_docs,
    detect_data_type,
    run_benchmark,
    run_llm_scoring,
)
from dq.benchmark_report import (
    benchmark_to_json,
    benchmark_to_markdown,
    print_benchmark_report,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_good_docs(n: int = 20) -> list[dict]:
    docs = []
    for i in range(n):
        text = (
            f"This is a well-written article about topic number {i}. "
            "The researchers discovered several important findings in their study. "
            "According to the data, the results were statistically significant. "
            "Furthermore, the analysis showed that the proposed method outperforms "
            "existing approaches on multiple benchmarks. The experiment was conducted "
            "over a period of several months with careful controls in place. "
            "In conclusion, the study provides valuable insights into the field."
        )
        docs.append({"text": text})
    return docs


def _make_bad_docs(n: int = 20) -> list[dict]:
    docs = []
    for i in range(n):
        text = f"q{i}: how to do thing\na: just do it lol\nok thanks"
        docs.append({"text": text})
    return docs


def _make_sft_docs(n: int = 10) -> list[dict]:
    docs = []
    for i in range(n):
        docs.append({
            "text": f"Explain topic {i}\nTopic {i} is an important concept in science.",
            "instruction": f"Explain topic {i}",
            "input": "",
            "output": f"Topic {i} is an important concept in science that involves complex interactions.",
        })
    return docs


def _make_pretrain_docs(n: int = 10) -> list[dict]:
    docs = []
    for i in range(n):
        docs.append({
            "text": (
                f"Chapter {i}: Introduction to Advanced Topics. "
                "This chapter covers the fundamental principles underlying modern research. "
                "We begin with a brief overview and then delve into specific methodologies."
            ),
        })
    return docs


# ---------------------------------------------------------------------------
# detect_data_type
# ---------------------------------------------------------------------------

class TestDetectDataType:
    def test_sft_with_instruction_field(self):
        docs = [{"instruction": "Do X", "output": "Y", "text": "Do X\nY"}]
        assert detect_data_type(docs) == "sft"

    def test_sft_with_conversations_field(self):
        docs = [{"conversations": [{"role": "user", "content": "Hi"}], "text": "Hi"}]
        assert detect_data_type(docs) == "sft"

    def test_pretrain_text_only(self):
        docs = [{"text": "Some long article about science."}]
        assert detect_data_type(docs) == "pretrain"

    def test_empty_docs(self):
        assert detect_data_type([]) == "pretrain"

    def test_mixed_majority_sft(self):
        docs = [
            {"instruction": "Do X", "text": "X"},
            {"instruction": "Do Y", "text": "Y"},
            {"text": "Just text"},
        ]
        assert detect_data_type(docs) == "sft"

    def test_mixed_majority_pretrain(self):
        docs = [
            {"text": "Article 1"},
            {"text": "Article 2"},
            {"instruction": "Do X", "text": "X"},
        ]
        assert detect_data_type(docs) == "pretrain"


# ---------------------------------------------------------------------------
# _extract_sft_fields
# ---------------------------------------------------------------------------

class TestExtractSFTFields:
    def test_structured_doc(self):
        doc = {"instruction": "Do X", "output": "Result Y"}
        instr, out = _extract_sft_fields(doc)
        assert instr == "Do X"
        assert out == "Result Y"

    def test_merged_text_doc(self):
        doc = {"text": "Do X\nResult Y"}
        instr, out = _extract_sft_fields(doc)
        assert instr == "Do X"
        assert out == "Result Y"

    def test_empty_doc(self):
        doc = {}
        instr, out = _extract_sft_fields(doc)
        assert instr == ""
        assert out == ""

    def test_text_no_newline(self):
        doc = {"text": "Just one line"}
        instr, out = _extract_sft_fields(doc)
        assert instr == "Just one line"
        assert out == ""


# ---------------------------------------------------------------------------
# SFTScores / PretrainScores dataclasses
# ---------------------------------------------------------------------------

class TestScoreDataclasses:
    def test_sft_scores_to_dict(self):
        scores = SFTScores(
            high_count=7,
            low_count=3,
            high_rate=0.7,
            rule_fail_counts={"completeness": 2, "factuality": 1},
            num_scored=10,
            scoring_errors=0,
        )
        d = scores.to_dict()
        assert d["type"] == "sft"
        assert d["high_count"] == 7
        assert d["low_count"] == 3
        assert d["high_rate"] == 0.7
        assert d["num_scored"] == 10
        assert d["rule_fail_counts"]["completeness"] == 2

    def test_pretrain_scores_to_dict(self):
        scores = PretrainScores(
            high_count=6,
            low_count=4,
            high_rate=0.6,
            rule_fail_counts={"coherence": 3, "originality": 1},
            num_scored=10,
            scoring_errors=0,
        )
        d = scores.to_dict()
        assert d["type"] == "pretrain"
        assert d["high_count"] == 6
        assert d["high_rate"] == 0.6
        assert d["num_scored"] == 10

    def test_default_scores(self):
        sft = SFTScores()
        assert sft.high_count == 0
        assert sft.to_dict()["type"] == "sft"

        pt = PretrainScores()
        assert pt.high_count == 0
        assert pt.to_dict()["type"] == "pretrain"


# ---------------------------------------------------------------------------
# DatasetResult with llm_scores
# ---------------------------------------------------------------------------

class TestDatasetResultLLMScores:
    def test_default_no_scores(self):
        dr = DatasetResult(name="test", num_docs=10)
        assert dr.llm_scores is None
        assert dr.data_type == "pretrain"

    def test_with_sft_scores(self):
        scores = SFTScores(high_count=3, low_count=2, high_rate=0.6, num_scored=5).to_dict()
        dr = DatasetResult(name="test", num_docs=10, data_type="sft", llm_scores=scores)
        assert dr.llm_scores["type"] == "sft"
        assert dr.llm_scores["high_rate"] == 0.6

    def test_with_pretrain_scores(self):
        scores = PretrainScores(high_count=4, low_count=1, high_rate=0.8, num_scored=5).to_dict()
        dr = DatasetResult(name="test", num_docs=10, data_type="pretrain", llm_scores=scores)
        assert dr.llm_scores["type"] == "pretrain"
        assert dr.llm_scores["high_rate"] == 0.8


# ---------------------------------------------------------------------------
# BenchmarkReport with LLM scoring fields
# ---------------------------------------------------------------------------

class TestBenchmarkReportLLMFields:
    def test_default_no_llm_scoring(self):
        report = BenchmarkReport()
        assert report.llm_scoring_enabled is False
        assert report.llm_samples == 0

    def test_with_llm_scoring_enabled(self):
        report = BenchmarkReport(llm_scoring_enabled=True, llm_samples=50)
        assert report.llm_scoring_enabled is True
        assert report.llm_samples == 50


# ---------------------------------------------------------------------------
# _score_sft_docs (mocked API)
# ---------------------------------------------------------------------------

class TestScoreSFTDocs:
    def _mock_response(self, content: str):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    def test_score_sft_docs_with_mock(self):
        """Test SFT scoring with mocked LLM Binary Judge."""
        import json
        judge_response_high = json.dumps({
            "instruction_following": {"pass": True, "reason": "ok"},
            "factuality": {"pass": True, "reason": "ok"},
            "completeness": {"pass": True, "reason": "ok"},
            "format_compliance": {"pass": True, "reason": "ok"},
            "harmlessness": {"pass": True, "reason": "ok"},
        })
        judge_response_low = json.dumps({
            "instruction_following": {"pass": False, "reason": "off-topic"},
            "factuality": {"pass": True, "reason": "ok"},
            "completeness": {"pass": False, "reason": "incomplete"},
            "format_compliance": {"pass": True, "reason": "ok"},
            "harmlessness": {"pass": True, "reason": "ok"},
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            self._mock_response(judge_response_high),
            self._mock_response(judge_response_low),
        ]

        docs = [
            {"instruction": "Q1", "output": "A1", "text": "Q1\nA1"},
            {"instruction": "Q2", "output": "A2", "text": "Q2\nA2"},
        ]

        with patch("dq.sft.llm_judge.get_client", return_value=mock_client):
            result = _score_sft_docs(docs, api_key="test", progress=False)

        assert result["type"] == "sft"
        assert result["high_count"] == 1
        assert result["low_count"] == 1
        assert result["num_scored"] == 2

    def test_score_sft_docs_api_failure(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        docs = [{"instruction": "Q1", "output": "A1", "text": "Q1\nA1"}]

        with patch("dq.sft.llm_judge.get_client", return_value=mock_client):
            result = _score_sft_docs(docs, api_key="test", progress=False)

        assert result["scoring_errors"] > 0


# ---------------------------------------------------------------------------
# _score_pretrain_docs (mocked API)
# ---------------------------------------------------------------------------

class TestScorePretrainDocs:
    def _mock_response(self, content: str):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    def test_score_pretrain_docs_with_mock(self):
        """Test pretrain scoring with mocked LLM Binary Judge."""
        import json
        judge_response_high = json.dumps({
            "information_density": {"pass": True, "reason": "ok"},
            "coherence": {"pass": True, "reason": "ok"},
            "originality": {"pass": True, "reason": "ok"},
        })
        judge_response_low = json.dumps({
            "information_density": {"pass": False, "reason": "shallow"},
            "coherence": {"pass": True, "reason": "ok"},
            "originality": {"pass": False, "reason": "generic"},
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            self._mock_response(judge_response_high),
            self._mock_response(judge_response_low),
        ]

        docs = [
            {"text": "Educational article about physics."},
            {"text": "Another educational article about chemistry."},
        ]

        with patch("dq.model_filters.llm_quality_judge.get_client", return_value=mock_client):
            result = _score_pretrain_docs(docs, api_key="test", progress=False)

        assert result["type"] == "pretrain"
        assert result["high_count"] == 1
        assert result["low_count"] == 1
        assert result["num_scored"] == 2

    def test_score_pretrain_docs_api_failure(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        docs = [{"text": "Some text."}]

        with patch("dq.model_filters.llm_quality_judge.get_client", return_value=mock_client):
            result = _score_pretrain_docs(docs, api_key="test", progress=False)

        assert result["scoring_errors"] > 0


# ---------------------------------------------------------------------------
# run_llm_scoring integration (mocked API)
# ---------------------------------------------------------------------------

class TestRunLLMScoring:
    def _mock_response(self, content: str):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    def test_run_llm_scoring_sft(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("4")

        sft_docs = _make_sft_docs(5)

        # Create a minimal report
        report = run_benchmark(
            datasets={"SFT": sft_docs, "SFT2": _make_sft_docs(5)},
            no_dedup=True,
        )

        with patch("dq.sft.complexity.get_client", return_value=mock_client), \
             patch("dq.sft.quality.get_client", return_value=mock_client):
            run_llm_scoring(
                report=report,
                datasets={"SFT": sft_docs, "SFT2": _make_sft_docs(5)},
                llm_samples=3,
                data_type_override="sft",
                progress=False,
            )

        assert report.llm_scoring_enabled is True
        assert report.llm_samples == 3
        assert report.datasets["SFT"].llm_scores is not None
        assert report.datasets["SFT"].llm_scores["type"] == "sft"
        assert report.datasets["SFT"].data_type == "sft"

    def test_run_llm_scoring_pretrain(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("4")

        pt_docs = _make_pretrain_docs(5)

        report = run_benchmark(
            datasets={"PT": pt_docs, "PT2": _make_pretrain_docs(5)},
            no_dedup=True,
        )

        with patch("dq.sft.educational.get_client", return_value=mock_client), \
             patch("dq.sft.writing_quality.get_client", return_value=mock_client):
            run_llm_scoring(
                report=report,
                datasets={"PT": pt_docs, "PT2": _make_pretrain_docs(5)},
                llm_samples=3,
                data_type_override="pretrain",
                progress=False,
            )

        assert report.datasets["PT"].llm_scores is not None
        assert report.datasets["PT"].llm_scores["type"] == "pretrain"

    def test_run_llm_scoring_auto_detect(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("3")

        sft_docs = _make_sft_docs(5)

        report = run_benchmark(
            datasets={"A": sft_docs, "B": _make_sft_docs(5)},
            no_dedup=True,
        )

        with patch("dq.sft.complexity.get_client", return_value=mock_client), \
             patch("dq.sft.quality.get_client", return_value=mock_client):
            run_llm_scoring(
                report=report,
                datasets={"A": sft_docs, "B": _make_sft_docs(5)},
                llm_samples=2,
                progress=False,
            )

        # SFT docs have 'instruction' field, so auto-detect should pick 'sft'
        assert report.datasets["A"].data_type == "sft"

    def test_run_llm_scoring_skips_unknown_dataset(self):
        report = BenchmarkReport()
        report.datasets["Known"] = DatasetResult(name="Known", num_docs=5)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("3")

        with patch("dq.sft.educational.get_client", return_value=mock_client), \
             patch("dq.sft.writing_quality.get_client", return_value=mock_client):
            run_llm_scoring(
                report=report,
                datasets={"Unknown": _make_pretrain_docs(3)},
                llm_samples=2,
                progress=False,
            )

        # "Unknown" not in report.datasets, so nothing should be scored
        assert report.datasets["Known"].llm_scores is None


# ---------------------------------------------------------------------------
# EducationalValueScorer
# ---------------------------------------------------------------------------

class TestEducationalValueScorer:
    def _mock_response(self, content: str):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    def test_score_with_mock(self):
        from dq.sft.educational import EducationalValueScorer
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("5")

        scorer = EducationalValueScorer(api_key="test")
        scorer._get_client = lambda: mock_client
        scorer.max_retries = 1

        doc = {"text": "A detailed explanation of quantum mechanics."}
        result = scorer.score(doc)
        assert result["educational_value_score"] == 5.0

    def test_score_empty_text(self):
        from dq.sft.educational import EducationalValueScorer
        scorer = EducationalValueScorer(api_key="test")
        scorer._get_client = lambda: MagicMock()

        doc = {"text": ""}
        result = scorer.score(doc)
        assert result["educational_value_score"] == -1.0

    def test_score_no_client(self):
        from dq.sft.educational import EducationalValueScorer
        scorer = EducationalValueScorer()
        scorer._get_client = lambda: None

        doc = {"text": "Some text."}
        result = scorer.score(doc)
        assert result["educational_value_score"] == -1.0

    def test_score_clamps_to_range(self):
        from dq.sft.educational import EducationalValueScorer
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("9")

        scorer = EducationalValueScorer(api_key="test")
        scorer._get_client = lambda: mock_client
        scorer.max_retries = 1

        doc = {"text": "Something."}
        result = scorer.score(doc)
        assert result["educational_value_score"] == 6.0

    def test_score_batch(self):
        from dq.sft.educational import EducationalValueScorer
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("4")

        scorer = EducationalValueScorer(api_key="test")
        scorer._get_client = lambda: mock_client
        scorer.max_retries = 1

        docs = [{"text": "A"}, {"text": "B"}]
        results = scorer.score_batch(docs)
        assert len(results) == 2
        assert all("educational_value_score" in d for d in results)


# ---------------------------------------------------------------------------
# WritingQualityScorer
# ---------------------------------------------------------------------------

class TestWritingQualityScorer:
    def _mock_response(self, content: str):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = content
        return mock_resp

    def test_score_with_mock(self):
        from dq.sft.writing_quality import WritingQualityScorer
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("4")

        scorer = WritingQualityScorer(api_key="test")
        scorer._get_client = lambda: mock_client
        scorer.max_retries = 1

        doc = {"text": "A well-written paragraph."}
        result = scorer.score(doc)
        assert result["writing_quality_score"] == 4.0

    def test_score_empty_text(self):
        from dq.sft.writing_quality import WritingQualityScorer
        scorer = WritingQualityScorer(api_key="test")
        scorer._get_client = lambda: MagicMock()

        doc = {"text": ""}
        result = scorer.score(doc)
        assert result["writing_quality_score"] == -1.0

    def test_score_no_client(self):
        from dq.sft.writing_quality import WritingQualityScorer
        scorer = WritingQualityScorer()
        scorer._get_client = lambda: None

        doc = {"text": "Some text."}
        result = scorer.score(doc)
        assert result["writing_quality_score"] == -1.0

    def test_score_batch(self):
        from dq.sft.writing_quality import WritingQualityScorer
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response("3")

        scorer = WritingQualityScorer(api_key="test")
        scorer._get_client = lambda: mock_client
        scorer.max_retries = 1

        docs = [{"text": "A"}, {"text": "B"}]
        results = scorer.score_batch(docs)
        assert len(results) == 2
        assert all("writing_quality_score" in d for d in results)


# ---------------------------------------------------------------------------
# Report output with LLM scores
# ---------------------------------------------------------------------------

class TestReportWithLLMScores:
    def _make_report_with_llm(self, data_type: str = "sft") -> BenchmarkReport:
        """Create a report with LLM scores for testing report output."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(20),
                "Bad": _make_bad_docs(20),
            },
            no_dedup=True,
        )
        report.llm_scoring_enabled = True
        report.llm_samples = 10

        if data_type == "sft":
            scores = SFTScores(
                high_count=7, low_count=3, high_rate=0.7,
                rule_fail_counts={"completeness": 2, "factuality": 1},
                num_scored=10,
            ).to_dict()
        else:
            scores = PretrainScores(
                high_count=6, low_count=4, high_rate=0.6,
                rule_fail_counts={"coherence": 3, "originality": 1},
                num_scored=10,
            ).to_dict()

        for dr in report.datasets.values():
            dr.data_type = data_type
            dr.llm_scores = scores

        return report

    def test_json_includes_llm_scores(self):
        report = self._make_report_with_llm("sft")
        json_str = benchmark_to_json(report)
        data = json.loads(json_str)

        assert data["llm_scoring_enabled"] is True
        assert data["llm_samples"] == 10
        for ds_data in data["datasets"].values():
            assert "llm_scores" in ds_data
            assert ds_data["llm_scores"]["type"] == "sft"
            assert ds_data["data_type"] == "sft"

    def test_json_includes_pretrain_scores(self):
        report = self._make_report_with_llm("pretrain")
        json_str = benchmark_to_json(report)
        data = json.loads(json_str)

        for ds_data in data["datasets"].values():
            assert ds_data["llm_scores"]["type"] == "pretrain"
            assert "high_rate" in ds_data["llm_scores"]
            assert "high_rate" in ds_data["llm_scores"]

    def test_json_no_llm_scores_when_disabled(self):
        report = run_benchmark(
            datasets={"A": _make_good_docs(10), "B": _make_bad_docs(10)},
            no_dedup=True,
        )
        json_str = benchmark_to_json(report)
        data = json.loads(json_str)

        assert data["llm_scoring_enabled"] is False
        for ds_data in data["datasets"].values():
            assert "llm_scores" not in ds_data

    def test_markdown_includes_layer2_section(self):
        report = self._make_report_with_llm("sft")
        md = benchmark_to_markdown(report)

        assert "Layer 2: LLM Binary Judge" in md
        assert "HIGH quality rate" in md

    def test_markdown_includes_pretrain_layer2(self):
        report = self._make_report_with_llm("pretrain")
        md = benchmark_to_markdown(report)

        assert "Layer 2: LLM Binary Judge" in md
        assert "HIGH quality rate" in md

    def test_markdown_no_layer2_when_disabled(self):
        report = run_benchmark(
            datasets={"A": _make_good_docs(10), "B": _make_bad_docs(10)},
            no_dedup=True,
        )
        md = benchmark_to_markdown(report)
        assert "Layer 2" not in md

    def test_rich_print_with_llm_scores_no_crash(self):
        from rich.console import Console

        report = self._make_report_with_llm("sft")
        test_console = Console(file=None, force_terminal=False, no_color=True, width=120)
        # Should not raise
        print_benchmark_report(report, console=test_console)

    def test_rich_print_pretrain_scores_no_crash(self):
        from rich.console import Console

        report = self._make_report_with_llm("pretrain")
        test_console = Console(file=None, force_terminal=False, no_color=True, width=120)
        print_benchmark_report(report, console=test_console)
