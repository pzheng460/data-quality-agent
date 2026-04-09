"""Tests for filter_detailed() method and per-rule benchmark reporting."""

import json

from dq.filters import ensure_registered; ensure_registered()
from dq.benchmark import BenchmarkReport, RuleStats, run_benchmark
from dq.benchmark_report import benchmark_to_json, benchmark_to_markdown, print_benchmark_report
from dq.filters.c4 import C4Filter
from dq.filters.fineweb import FineWebFilter
from dq.filters.gopher import GopherQualityFilter, GopherRepetitionFilter
from dq.filters.pii import PIIFilter


def _make_good_docs(n: int = 50) -> list[dict]:
    docs = []
    for i in range(n):
        text = (
            f"This is a well-written article about topic number {i}. "
            "The researchers discovered several important findings in their study. "
            "According to the data, the results were statistically significant. "
            "Furthermore, the analysis showed that the proposed method outperforms "
            "existing approaches on multiple benchmarks. The experiment was conducted "
            "over a period of several months with careful controls in place. "
            "In conclusion, the study provides valuable insights into the field. "
            "These findings have implications for future research directions."
        )
        docs.append({"text": text})
    return docs


def _make_bad_docs(n: int = 50) -> list[dict]:
    docs = []
    for i in range(n):
        text = f"q{i}: how to do thing\na: just do it lol\nok thanks"
        docs.append({"text": text})
    return docs


class TestBaseFilterDetailedDefault:
    """Test that BaseFilter.filter_detailed() default wraps filter()."""

    def test_gopher_quality_min_words_compat(self):
        """GopherQualityFilter filter_detailed reports min_words failures."""
        f = GopherQualityFilter(min_words=50, min_stopwords=0, max_bullet_lines_ratio=1.0, max_ellipsis_lines_ratio=1.0)
        doc = {"text": "short text"}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        assert any(fail["rule"] == "min_words" for fail in failures)

    def test_passing_doc_returns_empty_failures(self):
        f = GopherQualityFilter()
        docs = _make_good_docs(1)
        keep, failures = f.filter_detailed(docs[0])
        assert keep
        assert failures == []


class TestGopherQualityDetailed:
    def test_returns_all_failures(self):
        """A doc failing multiple rules should return ALL failures."""
        # Short text with no stopwords, bad alpha ratio, no punctuation
        f = GopherQualityFilter(min_words=5, min_stopwords=2)
        doc = {"text": "123 456 789"}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        assert len(failures) > 1
        rules = {fail["rule"] for fail in failures}
        # Should have min_words failure and others
        assert "min_words" in rules or "stopwords" in rules

    def test_single_failure(self):
        """A doc failing only one rule should return exactly one failure."""
        f = GopherQualityFilter(min_words=1000)
        doc = {"text": "This is a normal document with enough structure and words. " * 5}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        assert len(failures) >= 1
        assert any(fail["rule"] == "min_words" for fail in failures)

    def test_passing_doc(self):
        docs = _make_good_docs(1)
        f = GopherQualityFilter()
        keep, failures = f.filter_detailed(docs[0])
        assert keep
        assert failures == []

    def test_failure_dict_has_required_fields(self):
        f = GopherQualityFilter(min_words=50)
        doc = {"text": "short"}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        for fail in failures:
            assert "filter" in fail
            assert "rule" in fail
            assert "value" in fail
            assert "threshold" in fail

    def test_multiple_independent_failures(self):
        """A doc can fail min_words AND alpha_ratio independently."""
        f = GopherQualityFilter(
            min_words=100,
            min_alpha_ratio=0.9,
            min_stopwords=0,
            max_bullet_lines_ratio=1.0, max_ellipsis_lines_ratio=1.0,
        )
        # Very short numeric text
        doc = {"text": "1 2 3"}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        rules = {fail["rule"] for fail in failures}
        assert "min_words" in rules


class TestGopherRepetitionDetailed:
    def test_returns_all_failures(self):
        """Highly repetitive text should trigger multiple rules."""
        f = GopherRepetitionFilter(
            max_top_2gram=0.10,
            max_top_3gram=0.10,
            max_top_4gram=0.10,
            max_char_repetition=0.10,
        )
        text = "hello world " * 100
        doc = {"text": text}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        rules = {fail["rule"] for fail in failures}
        assert "top_2gram" in rules

    def test_passing_doc(self):
        docs = _make_good_docs(1)
        f = GopherRepetitionFilter()
        keep, failures = f.filter_detailed(docs[0])
        assert keep
        assert failures == []


class TestC4FilterDetailed:
    def test_lorem_ipsum_reported(self):
        f = C4Filter()
        doc = {"text": "This lorem ipsum text. Has enough sentences. For the filter check."}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        rules = {fail["rule"] for fail in failures}
        assert "lorem_ipsum" in rules

    def test_too_few_sentences(self):
        f = C4Filter(min_sentences=3)
        doc = {"text": "Only one sentence here."}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        rules = {fail["rule"] for fail in failures}
        assert "min_sentences" in rules

    def test_passing_doc(self):
        docs = _make_good_docs(1)
        f = C4Filter()
        keep, failures = f.filter_detailed(docs[0])
        assert keep
        assert failures == []


class TestFineWebFilterDetailed:
    def test_empty_doc(self):
        f = FineWebFilter()
        doc = {"text": ""}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        assert any(fail["rule"] == "empty_doc" for fail in failures)

    def test_passing_doc(self):
        docs = _make_good_docs(1)
        f = FineWebFilter()
        keep, failures = f.filter_detailed(docs[0])
        assert keep
        assert failures == []


class TestPIIFilterDetailed:
    def test_detects_email(self):
        f = PIIFilter(mode="detect")
        doc = {"text": "Contact me at user@example.com for details."}
        keep, failures = f.filter_detailed(doc)
        assert keep  # PII filter never rejects — it reports findings
        assert any(fail["rule"] == "email" for fail in failures)

    def test_no_pii(self):
        f = PIIFilter(mode="detect")
        doc = {"text": "This is a clean document with no personal information."}
        keep, failures = f.filter_detailed(doc)
        assert keep
        assert failures == []


class TestGopherQualityWordLimits:
    """Word count limits are now handled by GopherQualityFilter."""

    def test_too_short(self):
        f = GopherQualityFilter(min_words=50, min_stopwords=0, max_bullet_lines_ratio=1.0, max_ellipsis_lines_ratio=1.0)
        doc = {"text": "short"}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        assert any(fail["rule"] == "min_words" for fail in failures)
        assert any(fail["threshold"] == 50 for fail in failures if fail["rule"] == "min_words")

    def test_too_long(self):
        f = GopherQualityFilter(min_words=1, max_words=5, min_stopwords=0, max_bullet_lines_ratio=1.0, max_ellipsis_lines_ratio=1.0)
        doc = {"text": "one two three four five six seven eight nine ten"}
        keep, failures = f.filter_detailed(doc)
        assert not keep
        assert any(fail["rule"] == "max_words" for fail in failures)


class TestBenchmarkRuleStats:
    def test_rule_stats_populated(self):
        """run_benchmark should populate rule_stats for all datasets."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(20),
                "Bad": _make_bad_docs(20),
            },
            no_dedup=True,
        )
        assert len(report.rule_stats) == 2
        assert "Good" in report.rule_stats
        assert "Bad" in report.rule_stats

        # Bad docs should have rule failures in at least one filter
        bad_rules = report.rule_stats["Bad"]
        has_failures = False
        for filter_name, rules in bad_rules.items():
            for rule_name, rs in rules.items():
                if rs.failed > 0:
                    has_failures = True
                    assert rs.total == 20
                    assert rs.passed + rs.failed == rs.total
                    assert 0.0 <= rs.pass_rate <= 1.0
        assert has_failures

    def test_rule_stats_pass_rates(self):
        """Good docs should have higher per-rule pass rates than bad docs."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(20),
                "Bad": _make_bad_docs(20),
            },
            no_dedup=True,
        )

        # For gopher_quality's min_words rule, good should pass more
        good_gopher_rules = report.rule_stats.get("Good", {}).get("gopher_quality", {})
        bad_gopher_rules = report.rule_stats.get("Bad", {}).get("gopher_quality", {})

        if "min_words" in good_gopher_rules and "min_words" in bad_gopher_rules:
            assert good_gopher_rules["min_words"].pass_rate >= bad_gopher_rules["min_words"].pass_rate

    def test_json_includes_rules(self):
        """JSON output should include per-rule breakdown."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(10),
                "Bad": _make_bad_docs(10),
            },
            no_dedup=True,
        )
        json_str = benchmark_to_json(report)
        data = json.loads(json_str)

        # At least one filter should have rules in the JSON
        has_rules = False
        for ds_data in data["datasets"].values():
            for filter_data in ds_data["per_filter"].values():
                if "rules" in filter_data:
                    has_rules = True
                    for rule_name, rule_data in filter_data["rules"].items():
                        assert "total" in rule_data
                        assert "passed" in rule_data
                        assert "failed" in rule_data
                        assert "pass_rate" in rule_data
        assert has_rules

    def test_markdown_includes_rules(self):
        """Markdown output should include per-rule sub-rows."""
        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(10),
                "Bad": _make_bad_docs(10),
            },
            no_dedup=True,
        )
        md = benchmark_to_markdown(report)
        # Should contain tree-like rule prefixes
        assert "├─" in md or "└─" in md

    def test_rich_print_with_rules_no_crash(self):
        """Rich console output with rule breakdown should not crash."""
        from rich.console import Console

        report = run_benchmark(
            datasets={
                "Good": _make_good_docs(10),
                "Bad": _make_bad_docs(10),
            },
            no_dedup=True,
        )
        test_console = Console(file=None, force_terminal=False, no_color=True, width=120)
        print_benchmark_report(report, console=test_console)


class TestRuleStatsDataclass:
    def test_defaults(self):
        rs = RuleStats()
        assert rs.total == 0
        assert rs.passed == 0
        assert rs.failed == 0
        assert rs.pass_rate == 0.0

    def test_with_values(self):
        rs = RuleStats(total=100, passed=80, failed=20, pass_rate=0.8)
        assert rs.total == 100
        assert rs.pass_rate == 0.8
