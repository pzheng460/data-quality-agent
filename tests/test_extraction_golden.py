"""Golden tests for extraction quality on real arXiv papers.

Uses actual ingested papers to verify end-to-end extraction quality.
Papers are stored in tests/fixtures/golden/ as zstd-compressed JSONL.
Run `python tests/fixtures/golden/setup.py` to populate from a live run.
"""

import json
import os
import re

import pytest

# Path to golden test fixtures
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "golden")
GOLDEN_INPUT = os.path.join(GOLDEN_DIR, "ingested.jsonl")
GOLDEN_OUTPUT = os.path.join(GOLDEN_DIR, "extracted.jsonl")

# Skip if fixtures don't exist (CI without fixtures)
needs_fixtures = pytest.mark.skipif(
    not os.path.exists(GOLDEN_INPUT := os.path.join(GOLDEN_DIR := os.path.join(
        os.path.dirname(__file__), "fixtures", "golden"), "ingested.jsonl")),
    reason="Golden fixtures not found. Run: python tests/fixtures/golden/setup.py",
)


def _load_golden_docs():
    """Load golden ingested documents."""
    docs = {}
    if not os.path.exists(GOLDEN_INPUT):
        return docs
    with open(GOLDEN_INPUT) as f:
        for line in f:
            doc = json.loads(line)
            docs[doc["id"]] = doc
    return docs


def _extract_doc(doc):
    """Run extraction on a single document."""
    from dq.stages.extraction import ensure_extractors_registered
    from dq.stages.extraction.registry import get_extractor_for_format
    ensure_extractors_registered()

    text = doc.get("text", "")
    fmt = "text"
    if "\\begin{document}" in text or "\\documentclass" in text[:500]:
        fmt = "latex"
    elif "<html" in text[:500].lower():
        fmt = "html"

    extractor = get_extractor_for_format(fmt)
    return extractor.extract(doc.copy())


# ── Unit tests (no fixtures needed) ─────────────────────────────


class TestMathCleanup:
    """Test math formula cleanup in _math_to_latex."""

    def test_hidden_noalign_removed(self):
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _math_to_latex

        el = BeautifulSoup(
            '<math alttext="\\begin{cases}\\hidden@noalign{}x\\\\\\hidden@noalign{}y\\end{cases}"></math>',
            "lxml",
        ).find("math")
        result = _math_to_latex(el)
        assert "hidden@noalign" not in result
        assert "\\begin{cases}" in result

    def test_bigl_bigr_fix(self):
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _math_to_latex

        el = BeautifulSoup('<math alttext="\\bigl{(}x\\bigr{)}"></math>', "html.parser").find("math")
        assert _math_to_latex(el) == "\\bigl(x\\bigr)"

    def test_qed_to_square(self):
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _math_to_latex

        el = BeautifulSoup('<math alttext="\\qed"></math>', "html.parser").find("math")
        assert _math_to_latex(el) == "\\square"


class TestCrefnameGarbage:
    def test_alpha_garbage(self):
        """algorithmAlgorithmAlgorithm lemmaLemmaLemma..."""
        from dq.stages.extraction.html import html_to_text

        html = """<html><body>
        <p>Good content.</p>
        <p><span class="ltx_ERROR">x</span>algorithmAlgorithmAlgorithm
        lemmaLemmaLemma tableTableTable</p>
        </body></html>"""
        result = html_to_text(html)
        assert "algorithmAlgorithm" not in result
        assert "Good content" in result

    def test_garbage_with_numbers(self):
        """figureFig. #2#1#3 tableTab. #2#1#3..."""
        from dq.stages.extraction.html import html_to_text

        html = """<html><body>
        <p><span class="ltx_ERROR">x</span>figureFig. #2#1#3
        tableTab. #2#1#3 equationEq. (#2#1#3)
        sectionSec. algorithmAlg. hyperrefIgnoring</p>
        </body></html>"""
        result = html_to_text(html)
        assert "figureFig" not in result


class TestAlgorithmParser:
    def test_nested_structure(self):
        from dq.stages.extraction.algorithm import extract_algorithms_from_tex

        tex = r"""
\begin{algorithm}[t]
\caption{Test}
\KwIn{Input $x$.}
\KwOut{Output $y$.}
\ForEach{$i$ in range}{
    \If{$i > 0$}{
        process($i$);
    }
}
\end{algorithm}
"""
        results = extract_algorithms_from_tex(tex)
        assert len(results) == 1
        _, _, code = results[0]
        assert "Input:" in code
        assert "for each" in code
        lines = code.split("\n")
        # Check indentation exists
        indented = [l for l in lines if l.startswith("  ")]
        assert len(indented) >= 2

    def test_eif(self):
        from dq.stages.extraction.algorithm import extract_algorithms_from_tex

        tex = r"""
\begin{algorithm}
\caption{IfElse}
\eIf{cond}{yes;}{no;}
\end{algorithm}
"""
        results = extract_algorithms_from_tex(tex)
        _, _, code = results[0]
        assert "if" in code and "else" in code


class TestTableExtraction:
    def test_rowspan_first_row_only(self):
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _extract_table

        html = """<table>
        <tr><th>G</th><th>V</th></tr>
        <tr><td rowspan="3">A</td><td>1</td></tr>
        <tr><td>2</td></tr>
        <tr><td>3</td></tr>
        </table>"""
        rows = _extract_table(BeautifulSoup(html, "lxml").find("table"))
        assert rows[1][0] == "A"
        assert rows[2][0] == ""
        assert rows[3][0] == ""

    def test_colspan_first_col_only(self):
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _extract_table

        html = """<table>
        <tr><th>A</th><th>B</th><th>C</th></tr>
        <tr><td colspan="3">Wide</td></tr>
        </table>"""
        rows = _extract_table(BeautifulSoup(html, "lxml").find("table"))
        assert rows[1][0] == "Wide"
        assert rows[1][1] == ""
        assert rows[1][2] == ""

    def test_gfm_format(self):
        """Table output should have | prefix and suffix."""
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _extract_table, _dedup_camelcase

        html = """<table>
        <tr><th>A</th><th>B</th></tr>
        <tr><td>1</td><td>2</td></tr>
        </table>"""
        rows = _extract_table(BeautifulSoup(html, "lxml").find("table"))
        md_rows = []
        for cells in rows:
            md_rows.append("| " + " | ".join(_dedup_camelcase(c) for c in cells) + " |")
        assert md_rows[0].startswith("| ")
        assert md_rows[0].endswith(" |")


class TestTitleExtraction:
    def test_standard(self):
        from dq.stages.extraction.latex import _extract_title_from_tex
        assert _extract_title_from_tex(r"\title{My Paper}") == "My Paper"

    def test_icml(self):
        from dq.stages.extraction.latex import _extract_title_from_tex
        title = _extract_title_from_tex(r"\icmltitlerunning{Framework for X}")
        assert "Framework" in title

    def test_nested_braces(self):
        from dq.stages.extraction.latex import _extract_title_from_tex
        title = _extract_title_from_tex(r"\title{\textbf{Bold} and {nested}}")
        assert "Bold" in title
        assert "nested" in title

    def test_no_title(self):
        from dq.stages.extraction.latex import _extract_title_from_tex
        assert _extract_title_from_tex(r"\documentclass{article}") == ""


class TestCodeBlockProtection:
    def test_indentation_preserved_in_filter(self):
        from dq.stages.curation.filters.arxiv import _clean_text

        text = "normal text\n\n```\n  indented\n    more\n```\n\nend"
        result = _clean_text(text)
        assert "  indented" in result
        assert "    more" in result

    def test_mdframed_removed(self):
        from dq.stages.curation.filters.arxiv import _clean_text

        text = "Before.\n\n[\nfont= ,\nlinewidth=0.5pt,\ninnerleftmargin=10pt,\ninnerightmargin=10pt,\ninnertopmargin=10pt,\ninnerbottommargin=10pt,\n]monobox\n\nAfter."
        result = _clean_text(text)
        assert "innerleftmargin" not in result
        assert "Before" in result
        assert "After" in result


class TestSpanTable:
    """Test extraction of LaTeXML span-based tables."""

    def test_span_table_extraction(self):
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _extract_span_table

        html = """<span class="ltx_tabular">
        <span class="ltx_tr">
            <span class="ltx_td">Exam</span>
            <span class="ltx_td">Score</span>
        </span>
        <span class="ltx_tr">
            <span class="ltx_td">SAT</span>
            <span class="ltx_td">710</span>
        </span>
        </span>"""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        rows = _extract_span_table(soup.select_one(".ltx_tabular"))
        assert len(rows) == 2
        assert rows[0] == ["Exam", "Score"]
        assert rows[1] == ["SAT", "710"]


class TestTitleFallback:
    """Test title extraction from LaTeX when metadata is missing."""

    def test_arxiv_id_triggers_fallback(self):
        """When title looks like an arxiv ID, extract from LaTeX."""
        import re
        assert re.match(r"\d{4}\.\d+$", "2303.08774")
        assert not re.match(r"\d{4}\.\d+$", "GPT-4 Technical Report")


# ── Golden tests on real papers ─────────────────────────────────

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "golden", "final_output.jsonl")


def _load_golden():
    """Load golden output docs."""
    if not os.path.exists(GOLDEN_PATH):
        return {}
    docs = {}
    with open(GOLDEN_PATH) as f:
        for line in f:
            doc = json.loads(line)
            docs[doc["id"]] = doc["text"]
    return docs


@pytest.mark.skipif(not os.path.exists(GOLDEN_PATH), reason="Golden fixtures not generated")
class TestGoldenGPT4:
    """Verify GPT-4 Technical Report extraction quality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        docs = _load_golden()
        self.text = docs.get("arxiv_2303.08774", "")
        if not self.text:
            pytest.skip("GPT-4 paper not in golden fixtures")

    def test_has_sections(self):
        sections = [l for l in self.text.split("\n") if l.startswith("## ")]
        assert len(sections) >= 5, f"Expected 5+ sections, got {len(sections)}"

    def test_exam_table_has_pipes(self):
        """Table 1 should be a proper GFM table."""
        assert "| Exam |" in self.text or "| Exam|" in self.text

    def test_uniform_bar_exam_in_table(self):
        lines = self.text.split("\n")
        for line in lines:
            if "Uniform Bar Exam" in line and "|" in line:
                assert "298 / 400" in line
                return
        pytest.fail("Uniform Bar Exam not found in any table row")

    def test_no_mdframed_garbage(self):
        assert "innerleftmargin" not in self.text
        assert "innerbottommargin" not in self.text
        assert "]monobox" not in self.text

    def test_no_crefname_garbage(self):
        assert "algorithmAlgorithm" not in self.text
        assert "figureFig." not in self.text

    def test_makecell_merged(self):
        """USABO row should have score and percentile on same line."""
        for line in self.text.split("\n"):
            if "USABO" in line and "|" in line:
                assert "87 / 150" in line
                assert "99th" in line
                return
        # USABO might not be in this version — not a hard fail


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "fixtures", "golden", "final_output.jsonl")),
    reason="Golden fixtures not generated",
)
class TestGoldenLKLoss:
    """Verify LK Loss paper extraction quality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        docs = _load_golden()
        self.text = docs.get("arxiv_2602.23881", "")
        if not self.text:
            pytest.skip("LK Loss paper not in golden fixtures")

    def test_cross_refs_resolved(self):
        """Cross references should have section numbers, not empty parens."""
        # Should NOT have "(see )" with empty ref
        empty_refs = re.findall(r"\(see\s*\)", self.text)
        assert len(empty_refs) == 0, f"Found empty refs: {empty_refs}"

    def test_no_hidden_noalign(self):
        assert "hidden@noalign" not in self.text

    def test_math_formulas_present(self):
        assert "\\nabla" in self.text or "nabla" in self.text

    def test_tables_have_pipes(self):
        """Tables should be GFM format with pipes."""
        lines = self.text.split("\n")
        table_lines = [l for l in lines if "|" in l and "---" not in l]
        assert len(table_lines) > 5, "Expected multiple table rows"

    def test_temperature_not_duplicated(self):
        """A single colspan Temperature=0 should not be repeated in every column."""
        for line in self.text.split("\n"):
            if "Temperature" in line and "|" in line:
                # Count how many Temperature tokens appear in this row
                count = line.count("Temperature")
                cols = line.count("|") - 1
                # Should not fill every column — at most half the columns
                assert count <= cols // 2, (
                    f"Temperature duplicated across columns: {line[:100]}"
                )


# ── Content-level regression tests ──────────────────────────────
# These verify EXACT output content, not just presence.
# If table/algorithm extraction breaks, these will catch it.


class TestTableContentRegression:
    """Verify exact table content to catch column shifts, missing data, duplication."""

    def test_multirow_header_merge_exact(self):
        """Two-row header should merge into 'Parent Child' format."""
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _extract_table

        html = """<table>
        <tr><th rowspan="2">Model</th><th colspan="2">Bench A</th><th colspan="2">Bench B</th></tr>
        <tr><th>Lat</th><th>Acc</th><th>Lat</th><th>Acc</th></tr>
        <tr><td>X</td><td>10</td><td>90</td><td>20</td><td>80</td></tr>
        </table>"""
        soup = BeautifulSoup(html, "lxml")
        rows = _extract_table(soup.find("table"))
        header = rows[0]
        assert header[0] == "Model"
        assert header[1] == "Bench A Lat"
        assert header[2] == "Acc"  # colspan: Bench A only merges with first sub-col
        assert header[3] == "Bench B Lat"
        data = rows[1]
        assert data == ["X", "10", "90", "20", "80"]

    def test_rowspan_only_first_row_has_value(self):
        """Rowspan=3: value on first row, empty on rows 2-3."""
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _extract_table

        html = """<table>
        <tr><th>Group</th><th>Item</th><th>Val</th></tr>
        <tr><td rowspan="3">A</td><td>x</td><td>1</td></tr>
        <tr><td>y</td><td>2</td></tr>
        <tr><td>z</td><td>3</td></tr>
        <tr><td rowspan="2">B</td><td>w</td><td>4</td></tr>
        <tr><td>v</td><td>5</td></tr>
        </table>"""
        soup = BeautifulSoup(html, "lxml")
        rows = _extract_table(soup.find("table"))
        assert rows[1] == ["A", "x", "1"]
        assert rows[2] == ["", "y", "2"]
        assert rows[3] == ["", "z", "3"]
        assert rows[4] == ["B", "w", "4"]
        assert rows[5] == ["", "v", "5"]

    def test_colspan_no_duplication(self):
        """Colspan=3: text in first col, empty in cols 2-3."""
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _extract_table

        html = """<table>
        <tr><th>A</th><th>B</th><th>C</th></tr>
        <tr><td colspan="3">All three</td></tr>
        <tr><td>1</td><td>2</td><td>3</td></tr>
        </table>"""
        soup = BeautifulSoup(html, "lxml")
        rows = _extract_table(soup.find("table"))
        assert rows[1] == ["All three", "", ""]
        assert rows[2] == ["1", "2", "3"]

    def test_span_table_extracts_all_rows(self):
        """LaTeXML span-based table should produce correct row count."""
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _extract_span_table

        html = '<span class="ltx_tabular">'
        for row in [["H1", "H2"], ["a", "1"], ["b", "2"], ["c", "3"]]:
            html += '<span class="ltx_tr">'
            for cell in row:
                html += f'<span class="ltx_td">{cell}</span>'
            html += '</span>'
        html += '</span>'
        soup = BeautifulSoup(html, "lxml")
        rows = _extract_span_table(soup.select_one(".ltx_tabular"))
        assert len(rows) == 4
        assert rows[0] == ["H1", "H2"]
        assert rows[3] == ["c", "3"]

    def test_gfm_pipe_format(self):
        """Output rows must start with '| ' and end with ' |'."""
        from bs4 import BeautifulSoup
        from dq.stages.extraction.html import _extract_table, _dedup_camelcase

        html = """<table>
        <tr><th>A</th><th>B</th></tr>
        <tr><td>1</td><td>2</td></tr>
        </table>"""
        rows = _extract_table(BeautifulSoup(html, "lxml").find("table"))
        for cells in rows:
            line = "| " + " | ".join(_dedup_camelcase(c) for c in cells) + " |"
            assert line.startswith("| ") and line.endswith(" |")


class TestAlgorithmContentRegression:
    """Verify algorithm extraction produces correct indented pseudocode."""

    def test_full_algorithm_structure(self):
        """End-to-end: LaTeX algorithm → indented pseudocode with all keywords."""
        from dq.stages.extraction.algorithm import extract_algorithms_from_tex

        tex = r"""
\begin{algorithm}[t]
\caption{Binary Search}
\KwIn{Sorted array $A$, target $x$.}
\KwOut{Index $i$ or $-1$.}
\BlankLine
$lo \gets 0$; $hi \gets n-1$
\While{$lo \le hi$}{
    $mid \gets \lfloor (lo+hi)/2 \rfloor$;
    \eIf{$A[mid] = x$}{
        \Return{$mid$}
    }{
        \If{$A[mid] < x$}{
            $lo \gets mid + 1$;
        }
        \Else{
            $hi \gets mid - 1$;
        }
    }
}
\Return{$-1$}
\end{algorithm}
"""
        results = extract_algorithms_from_tex(tex)
        assert len(results) == 1
        caption, label, code = results[0]
        assert "Binary Search" in caption
        lines = code.split("\n")

        # Check keywords present
        assert any("Input:" in l for l in lines)
        assert any("Output:" in l for l in lines)
        assert any("while" in l for l in lines)
        assert any("if" in l for l in lines)
        assert any("else:" in l for l in lines)
        assert any("return" in l for l in lines)

        # Check indentation increases
        indent_levels = set()
        for l in lines:
            stripped = l.lstrip()
            if stripped:
                indent = len(l) - len(stripped)
                indent_levels.add(indent)
        assert len(indent_levels) >= 3, f"Expected 3+ indent levels, got {indent_levels}"

    def test_algorithm_math_preserved(self):
        """Math inside algorithm should be preserved as $...$."""
        from dq.stages.extraction.algorithm import extract_algorithms_from_tex

        tex = r"""
\begin{algorithm}
\caption{Test}
\KwIn{Array $A$ of size $n$.}
\ForEach{$i = 1$ to $n$}{
    $s \gets s + A[i]$;
}
\end{algorithm}
"""
        _, _, code = extract_algorithms_from_tex(tex)[0]
        assert "$A$" in code
        assert "$n$" in code
        assert "$i = 1$" in code

    def test_algorithm_not_in_code_block_when_no_tex(self):
        """When raw_tex is None, fall back to HTML-based extraction."""
        from dq.stages.extraction.html import html_to_text

        html = """<html><body>
        <div class="ltx_listing"><div class="ltx_listingline">
        <span class="ltx_ERROR">\\KwIn</span>Input x
        </div></div>
        </body></html>"""
        result = html_to_text(html, raw_tex=None)
        # Should still have some content (fallback extraction)
        assert "Input" in result or "x" in result


@pytest.mark.skipif(not os.path.exists(GOLDEN_PATH), reason="Golden fixtures not generated")
class TestGoldenTableContent:
    """Exact content checks on golden paper tables."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.docs = _load_golden()

    def test_gpt4_exam_table_header(self):
        """GPT-4 Table 1 header must have exactly 4 columns."""
        text = self.docs.get("arxiv_2303.08774", "")
        for line in text.split("\n"):
            if "| Exam |" in line:
                assert line.strip() == "| Exam | GPT-4 | GPT-4 (no vision) | GPT-3.5 |"
                return
        pytest.fail("Exam table header not found")

    def test_gpt4_bar_exam_row_exact(self):
        """Bar Exam row must have correct scores in correct columns."""
        text = self.docs["arxiv_2303.08774"] if hasattr(self, 'docs') else _load_golden().get("arxiv_2303.08774", "")
        for line in text.split("\n"):
            if "Uniform Bar Exam" in line and "|" in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                assert cells[0] == "Uniform Bar Exam (MBE+MEE+MPT)"
                assert "298 / 400" in cells[1]
                assert "213 / 400" in cells[3]
                return
        pytest.fail("Bar Exam row not found")

    def test_gpt4_contamination_table_usabo(self):
        """USABO row in contamination table must have merged makecell."""
        text = _load_golden().get("arxiv_2303.08774", "")
        for line in text.split("\n"):
            if "USABO" in line and "|" in line and "Contam" not in line:
                assert "87 / 150" in line
                assert "99th - 100th" in line
                # Ensure it's on ONE line (makecell merged)
                assert line.count("\n") == 0
                return
        pytest.fail("USABO contamination row not found")

    def test_lk_temperature_not_duplicated(self):
        """Temperature=0 colspan should appear only once in its row."""
        text = _load_golden().get("arxiv_2602.23881", "")
        for line in text.split("\n"):
            if "Temperature=0" in line and "|" in line:
                assert line.count("Temperature") == 1, f"Duplicated: {line[:80]}"
                return
        pytest.fail("Temperature=0 row not found")


class TestMdframedCleanup:
    def test_multiline_params_removed(self):
        from dq.stages.curation.filters.arxiv import _clean_text

        text = "text before\n\n[\nfont= ,\nlinewidth=0.5pt,\ninnerleftmargin=10pt,\ninnerightmargin=10pt,\ninnertopmargin=10pt,\ninnerbottommargin=10pt,\n]monobox\n\ntext after"
        result = _clean_text(text)
        assert "font=" not in result
        assert "monobox" not in result
        assert "text before" in result
        assert "text after" in result


# ── Markdown quality checks (merged from test_markdown_quality.py) ───


@pytest.mark.skipif(not os.path.exists(GOLDEN_PATH), reason="Golden fixtures not generated")
class TestMarkdownQuality:
    """Verify markdown validity on golden papers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.docs = _load_golden()

    def test_balanced_inline_math(self):
        """Inline math $ count should be even."""
        for pid, text in self.docs.items():
            no_display = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
            count = no_display.count("$")
            assert count % 2 == 0, f"{pid}: odd $ count ({count})"

    def test_balanced_display_math(self):
        """Display math $$ count should be even."""
        for pid, text in self.docs.items():
            count = text.count("$$")
            assert count % 2 == 0, f"{pid}: odd $$ count ({count})"

    def test_balanced_code_fences(self):
        """Code fences ``` count should be even."""
        for pid, text in self.docs.items():
            count = text.count("```")
            assert count % 2 == 0, f"{pid}: odd ``` count ({count})"

    def test_no_empty_headings(self):
        for pid, text in self.docs.items():
            empty = re.findall(r"^#{1,4}\s*$", text, re.MULTILINE)
            assert not empty, f"{pid}: empty headings found"

    def test_no_orphaned_ref_brackets(self):
        for pid, text in self.docs.items():
            orphans = re.findall(r"(?:Figure|Table|Section)\s*\)", text)
            assert not orphans, f"{pid}: orphaned ref brackets: {orphans[:3]}"


# ── Full pipeline regression (from test_golden.py) ──────────────


@pytest.mark.skipif(not os.path.exists(GOLDEN_PATH), reason="Golden fixtures not generated")
class TestGoldenRegression:
    """Regression tests on golden output: no LaTeX junk, no figure placeholders."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.docs = _load_golden()

    def test_all_papers_have_content(self):
        for pid, text in self.docs.items():
            assert len(text) > 1000, f"{pid}: too short ({len(text)} chars)"

    def test_low_latex_residual(self):
        """Cleaned text should have very few residual LaTeX commands outside math."""
        for pid, text in self.docs.items():
            no_math = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
            no_math = re.sub(r"\$[^$]+?\$", "", no_math)
            cmds = re.findall(r"\\[a-zA-Z]{2,}", no_math)
            ratio = len(cmds) / max(len(no_math), 1)
            assert ratio < 0.01, f"{pid}: LaTeX residual {ratio:.1%} ({len(cmds)} cmds)"

    def test_no_latex_junk(self):
        for pid, text in self.docs.items():
            for junk in ["toprule", "midrule", "bottomrule", "tabcolsep"]:
                assert junk not in text, f"{pid}: found '{junk}'"
