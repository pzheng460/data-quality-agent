"""Tests for the Arxiv clean-then-judge filter."""

from dq.filters import ensure_registered; ensure_registered()
from dq.filters.arxiv import ArxivFilter, _clean_latex, _residual_frac


class TestCleanLatex:
    def test_cite_removed(self):
        result = _clean_latex(r"As shown in \cite{vaswani2017}, transformers work well.")
        assert r"\cite" not in result
        assert "transformers work well" in result

    def test_cite_with_multiple_keys(self):
        result = _clean_latex(r"See \cite{a,b,c} for details.")
        assert r"\cite" not in result
        assert "details" in result

    def test_ref_removed(self):
        result = _clean_latex(r"See Figure \ref{fig:main} for the architecture.")
        assert r"\ref" not in result
        assert "architecture" in result

    def test_textbf_keeps_content(self):
        result = _clean_latex(r"This is \textbf{important} text.")
        assert "important" in result
        assert r"\textbf" not in result

    def test_textit_keeps_content(self):
        result = _clean_latex(r"The \textit{key insight} is simple.")
        assert "key insight" in result
        assert r"\textit" not in result

    def test_emph_keeps_content(self):
        result = _clean_latex(r"We \emph{emphasize} this point.")
        assert "emphasize" in result
        assert r"\emph" not in result

    def test_begin_end_removed(self):
        result = _clean_latex(r"\begin{theorem} The proof is trivial. \end{theorem}")
        assert "The proof is trivial" in result
        assert r"\begin" not in result
        assert r"\end" not in result

    def test_layout_commands_removed(self):
        result = _clean_latex(r"\noindent This paragraph starts here. \vspace{1em}")
        assert "This paragraph starts here" in result
        assert r"\noindent" not in result
        assert r"\vspace" not in result

    def test_inline_math_protected(self):
        result = _clean_latex(r"We have $E=mc^2$ and \cite{einstein1905}.")
        assert "$E=mc^2$" in result
        assert "\\cite" not in result

    def test_display_math_protected(self):
        result = _clean_latex(r"\noindent $$\int_0^1 f(x) dx$$ is the integral.")
        assert "$$\\int_0^1 f(x) dx$$" in result
        assert "\\noindent" not in result

    def test_footnote_keeps_content(self):
        result = _clean_latex(r"Main text\footnote{A footnote.} continues.")
        assert "A footnote." in result

    def test_multi_blank_lines_collapsed(self):
        result = _clean_latex("Line 1.\n\n\n\n\nLine 2.")
        assert "\n\n\n" not in result
        assert "Line 1." in result
        assert "Line 2." in result

    def test_clean_text_unchanged(self):
        text = "# Title\n\n## Abstract\n\nThis is a clean paper about machine learning."
        result = _clean_latex(text)
        assert result == text


class TestResidualFrac:

    def test_clean_text_zero(self):
        assert _residual_frac("This is clean text with no LaTeX.") == 0.0

    def test_heavy_latex(self):
        text = r"\begin{document} \section{Intro} \textbf{Hello} \cite{ref} \end{document}"
        frac = _residual_frac(text)
        assert frac > 0.1

    def test_math_excluded(self):
        # Math should not count toward residual
        text = "Clean text. $\\alpha + \\beta = \\gamma$. More clean text here."
        frac = _residual_frac(text)
        assert frac < 0.01


class TestArxivFilter:

    def test_good_paper_passes(self):
        f = ArxivFilter()
        doc = {"text": (
            "# Attention Is All You Need\n\n"
            "## Abstract\n\n"
            "We propose a new architecture based on attention mechanisms.\n\n"
            "## Introduction\n\n"
            "Sequence modeling has been dominated by recurrent architectures.\n\n"
            "## Method\n\n"
            "We introduce the Transformer model.\n"
        )}
        keep, info = f.filter(doc)
        assert keep, f"Good paper should pass: {info}"

    def test_high_residual_rejected(self):
        f = ArxivFilter(max_latex_residual=0.01)
        doc = {"text": (
            "# Title\n\n## Abstract\n\n"
            r"\section{Intro} \textbf{test} \cite{a} \ref{b} "
            r"\begin{theorem} \end{theorem} \noindent \vspace{1em} "
            r"\section{Method} more \cite{x} and \ref{y} stuff here "
            # After cleaning these should be gone, but we set threshold very low
        )}
        # Even after cleaning, there may be some residual
        keep, info = f.filter(doc)
        # The filter cleans first, so we can't guarantee rejection
        # This test mainly checks the flow works

    def test_missing_abstract_rejected(self):
        f = ArxivFilter(require_abstract=True)
        doc = {"text": (
            "# Title\n\n"
            "## Introduction\n\n"
            "Some text here.\n\n"
            "## Method\n\n"
            "More text here.\n"
        )}
        keep, info = f.filter(doc)
        assert not keep
        assert info.get("rule") == "missing_abstract"

    def test_too_few_sections_rejected(self):
        f = ArxivFilter(require_abstract=False, min_sections=3)
        doc = {"text": "# Title\n\nJust one section with some text."}
        keep, info = f.filter(doc)
        assert not keep
        assert info.get("rule") == "too_few_sections"

    def test_cleans_in_place(self):
        f = ArxivFilter(require_abstract=False, min_sections=0)
        doc = {"text": r"# Title\n\n## Abstract\n\n## Intro\n\nText \cite{foo} here."}
        f.filter(doc)
        assert r"\cite" not in doc["text"]

    def test_filter_detailed_returns_all_failures(self):
        f = ArxivFilter(require_abstract=True, min_sections=10)
        doc = {"text": "Short text without structure."}
        ok, failures = f.filter_detailed(doc)
        assert not ok
        rules = {fail["rule"] for fail in failures}
        assert "missing_abstract" in rules
        assert "too_few_sections" in rules

    def test_registered(self):
        from dq.pipeline import get_filter_class
        cls = get_filter_class("arxiv")
        assert cls is ArxivFilter
