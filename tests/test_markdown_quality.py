"""Tests that extracted text produces valid, renderable markdown.

Checks for common issues that break markdown rendering:
- Unbalanced $ (broken inline math)
- Unbalanced $$ (broken display math)
- Unbalanced ``` (broken code blocks)
- Empty headings
- Residual LaTeX % comments
- Residual citation keys
- Orphaned brackets/parens from citation removal
"""

import re
import pytest


def validate_markdown(text: str) -> list[str]:
    """Check markdown text for rendering issues. Returns list of problems."""
    issues = []

    # 1. Count single $ (inline math) — must be even
    # First remove $$ to avoid double-counting
    no_display = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    single_dollar_count = no_display.count('$')
    if single_dollar_count % 2 != 0:
        issues = []
        # Find unmatched $ positions
        in_math = False
        for i, ch in enumerate(no_display):
            if ch == '$':
                in_math = not in_math
                if not in_math:
                    continue
                # Check if this $ opens a math block that's not closed on same line
                rest_of_line = no_display[i+1:].split('\n')[0]
                if '$' not in rest_of_line:
                    ctx = no_display[max(0,i-20):i+20]
                    issues.append(f"Unmatched $ near: ...{ctx}...")
        if issues:
            issues = [f"Odd number of $ ({single_dollar_count}): " + issues[0]]

    # 2. Unbalanced $$
    dd_count = text.count('$$')
    if dd_count % 2 != 0:
        issues.append(f"Odd number of $$ ({dd_count}) — unclosed display math")

    # 3. Unbalanced ```
    backtick_count = text.count('```')
    if backtick_count % 2 != 0:
        issues.append(f"Odd number of ``` ({backtick_count}) — unclosed code block")

    # 4. LaTeX % comments leaking (outside math)
    no_math = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    no_math = re.sub(r'\$[^$]+?\$', '', no_math)
    pct_lines = [l for l in no_math.split('\n') if '%' in l and not l.strip().startswith('#')]
    if pct_lines:
        issues.append(f"LaTeX % comment in {len(pct_lines)} non-math lines")

    # 5. Empty headings
    empty_heads = re.findall(r'^#{1,4}\s*$', text, re.MULTILINE)
    if empty_heads:
        issues.append(f"Empty headings: {len(empty_heads)}")

    # 6. Orphaned reference brackets: "Figure )" or "Table )"
    orphans = re.findall(r'(?:Figure|Table|Section)\s*\)', text)
    if orphans:
        issues.append(f"Orphaned reference brackets: {orphans[:3]}")

    return issues


class TestMarkdownQuality:
    """Test markdown quality on the golden test papers."""

    @staticmethod
    def _get_golden_texts():
        """Get cleaned texts from golden input via ArxivFilter."""
        from pathlib import Path
        import json

        golden_input = Path(__file__).parent / "golden" / "input.jsonl"
        if not golden_input.exists():
            return []

        from dq.stages.curation.filters import ensure_registered
        ensure_registered()
        from dq.stages.curation.filters.arxiv import _clean_text

        texts = []
        with open(golden_input) as f:
            for line in f:
                doc = json.loads(line)
                cleaned = _clean_text(doc.get("text", ""))
                texts.append((doc.get("id", "unknown"), cleaned))
        return texts

    def test_no_unmatched_dollars(self):
        """Inline math $ must be balanced."""
        for doc_id, text in self._get_golden_texts():
            no_display = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
            count = no_display.count('$')
            assert count % 2 == 0, f"{doc_id}: odd $ count ({count})"

    def test_no_unmatched_display_math(self):
        """Display math $$ must be balanced."""
        for doc_id, text in self._get_golden_texts():
            count = text.count('$$')
            assert count % 2 == 0, f"{doc_id}: odd $$ count ({count})"

    def test_no_unmatched_code_fences(self):
        """Code fences ``` must be balanced."""
        for doc_text in [t for _, t in self._iter_golden()]:
            count = doc_text.count('```')
            assert count % 2 == 0

    def _iter_golden(self):
        return self._get_golden_texts()

    def test_no_empty_headings(self):
        """No empty # headings."""
        for doc_id, text in self._get_golden_texts():
            empty = re.findall(r'^#{1,4}\s*$', text, re.MULTILINE)
            assert not empty, f"{doc_id}: empty headings found"

    def test_no_latex_percent_in_math(self):
        """No LaTeX % comments leaking into math."""
        for doc_id, text in self._get_golden_texts():
            display_blocks = re.findall(r'\$\$(.*?)\$\$', text, re.DOTALL)
            for block in display_blocks:
                assert '%' not in block, f"{doc_id}: % in display math: {block[:60]}"

    def test_no_citation_keys(self):
        """No bare citation keys like 'touvron2023llama'."""
        for doc_id, text in self._get_golden_texts():
            # Only match outside math regions
            no_math = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
            no_math = re.sub(r'\$[^$]+?\$', '', no_math)
            keys = re.findall(r'\b[A-Za-z]{2,}\d{4}[A-Za-z]{2,}\b', no_math)
            assert not keys, f"{doc_id}: citation keys found: {keys[:5]}"


def test_validate_markdown_clean():
    """Validate markdown on a known clean sample."""
    clean = """# Title

Some text with $x^2$ inline math.

$$E = mc^2$$

## Section

More text.
"""
    issues = validate_markdown(clean)
    assert issues == []


def test_validate_markdown_broken():
    """Detect common markdown issues."""
    broken = "Some text with $unmatched math"
    issues = validate_markdown(broken)
    assert any("$" in i for i in issues)
