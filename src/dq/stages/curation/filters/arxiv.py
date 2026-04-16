"""Arxiv-specific clean-then-judge filter.

Handles all data cleaning for LaTeXML-converted text:
- Citation text and artifacts (numeric [14], author-year keys, empty parens)
- Footnote markers (repeated superscript numbers)
- Section number stripping (1.2 Introduction → Introduction)
- Residual LaTeX commands
- Whitespace normalization

Then rejects docs with too many residual LaTeX artifacts or missing structure.
"""

import re
from typing import Any

from dq.stages.curation.filters.base import BaseFilter
from dq.pipeline import register_filter

# ── Math protection ──

_DISPLAY_MATH_PATS = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(
        r"\\begin\{(?:equation|align|alignat|gather|multline|eqnarray)\*?\}"
        r".*?"
        r"\\end\{(?:equation|align|alignat|gather|multline|eqnarray)\*?\}",
        re.DOTALL,
    ),
]
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(?!\s)[^$\n]+?\$(?!\$)")

# ── Structure detection ──

_ABSTRACT_RE = re.compile(r"(?:^|\n)#{1,4}\s*abstract", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#{1,4}\s+", re.MULTILINE)

# ── Citation patterns ──

# Citation key patterns — LaTeXML renders \cite{key} as plain text.
# Matches: "touvron2023llama2", "Radford2018ImprovingLU", "chen2024eagle3"
# Requires at least 2 alpha chars before the year to avoid false positives
_CITE_KEY_RE = re.compile(
    r"\b[A-Za-z]{2,}\d{4}[A-Za-z]{2,}\b(?:\s*[,;]\s*\b[A-Za-z]{2,}\d{4}[A-Za-z]*\b)*"
    r"|\b[a-z]{2,}\d{4}[a-z]*\b(?:\s*[,;]\s*\b[a-z]+\d{4}[a-z]*\b)*"
)

# ── Section number in headings ──

_SECTION_NUM_RE = re.compile(r"^(#{1,4}\s+)\d+(?:\.\d+)*\s+", re.MULTILINE)


def _protect_math(text: str):
    """Replace math regions with placeholders to avoid mangling them."""
    phs: list[tuple[str, str]] = []
    ctr = [0]

    def _ph(m: re.Match) -> str:
        key = f"\x00M{ctr[0]}\x00"
        phs.append((key, m.group()))
        ctr[0] += 1
        return key

    for pat in _DISPLAY_MATH_PATS:
        text = pat.sub(_ph, text)
    text = _INLINE_MATH_RE.sub(_ph, text)
    return text, phs


def _protect_code_blocks(text: str):
    """Replace ``` code blocks with placeholders to preserve indentation."""
    phs: list[tuple[str, str]] = []
    ctr = 0

    def _ph(m):
        nonlocal ctr
        key = f"\x00CB{ctr}\x00"
        phs.append((key, m.group()))
        ctr += 1
        return key

    text = re.sub(r"```.*?```", _ph, text, flags=re.DOTALL)
    return text, phs


def _clean_text(text: str) -> str:
    """Clean LaTeXML-converted text. All cleaning logic lives here."""

    # Protect code blocks (algorithm pseudocode) from mangling
    text, code_phs = _protect_code_blocks(text)

    text, math_phs = _protect_math(text)

    # --- Citations ---
    # Remove bare citation keys (e.g. "touvron2023llama2")
    text = _CITE_KEY_RE.sub("", text)
    # Clean up empty/near-empty parens/brackets left after citation removal
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
    text = re.sub(r"\[\s*[,;]?\s*\]", "", text)
    text = re.sub(r",\s*\)", ")", text)
    text = re.sub(r"\(\s*,", "(", text)
    text = re.sub(r"\(\s*;\s*\)", "", text)

    # ── Footnote markers ──
    # LaTeXML renders footnote marks as repeated superscript numbers: "1 1 1"
    text = re.sub(r"(\d+)(?:\s+\1){1,}", r"\1", text)

    # ── Section number stripping ──
    # "## 2.1 Introduction" → "## Introduction"
    text = _SECTION_NUM_RE.sub(r"\1 ", text)
    # Also handle headings without markdown prefix (rare)
    text = re.sub(r"^(#{1,4}\s+)\d+(?:\.\d+)*\s+", r"\1", text, flags=re.MULTILINE)

    # ── Fix \$ → % (LaTeXML renders \% as \$ in some contexts) ──
    text = re.sub(r"(\d)\\(\$)", r"\1%", text)  # "37\$" → "37%"
    text = re.sub(r"\\(\$)\s*\.", r"%.", text)  # "R\$ ." → "R%."

    # ── Residual LaTeX commands ──
    text = re.sub(r"\\(?:begin|end)\{[^}]+\}", "", text)
    text = re.sub(r"\\[a-zA-Z]{2,}\b", "", text)
    # ── LaTeXML LABEL:xxx references (from failed \ref) ──
    text = re.sub(r"LABEL:\S+", "", text)

    # ── mdframed / tcolorbox parameter blocks ──
    # LaTeXML leaks multi-line environment options like:
    #   [\nfont= ,\nlinewidth=0.5pt,\n...\n]monobox
    text = re.sub(
        r"\[\s*\n(?:[ \t]*\w+=.*\n)+[ \t]*\]\w*",
        "",
        text,
    )

    # ── pgfplots / tikz / pgf garbage ──
    # LaTeXML leaks tikz/pgfplots configuration as plain text lines like:
    #   "compat=1.14", "/pgfplots/... /.style=", "ybar, ..."
    text = re.sub(r"^[ \t]*compat=[\d.]+,?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*font=.*\bybar\b[^\n]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*/pgfplots/[^\n]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*/tikz/[^\n]*$", "", text, flags=re.MULTILINE)
    # pgfplots package names leaked: "pgfplots.groupplots compat=1.3 patterns"
    text = re.sub(r"^[ \t]*pgfplots[\w.]*\s+compat=[^\n]*$", "", text, flags=re.MULTILINE)
    # tikz drawing commands: "ybar,", "[ybar, fill=...", "; coordinates"
    text = re.sub(r"^[ \t]*(?:ybar|xbar),?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\[ybar[^\n]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*;\s*coordinates\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*font=,?\]?\s*\[?ybar[^\n]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*;\s*\[ybar[^\n]*$", "", text, flags=re.MULTILINE)

    # ── \includegraphics residuals ──
    # LaTeXML leaks image options+paths: "[trim=0cm 19cm 0cm 0cm,clip,width=0.9]images/foo.jpg"
    text = re.sub(r"\[(?:trim|width|height|scale|clip)[^\]]*\]\S*\.(?:png|jpg|jpeg|pdf|eps|svg)\b", "", text)
    # Bare image paths: "images/foo.png" or "figures/bar.pdf"
    text = re.sub(r"(?:images|figures|figs|fig|imgs|assets)/\S*\.(?:png|jpg|jpeg|pdf|eps|svg)\b", "", text)

    # ── Figure/Table reference cleanup ──
    # "Figure )" or "Table )" left after citation removal
    text = re.sub(r"(Figure|Table|Section|Eq\.?)\s*\)", r"\1", text)
    text = re.sub(r"(Figure|Table|Section|Eq\.?)\s*\(\s*\)", r"\1", text)

    # ── Author affiliation residuals ──
    # "Name 1,* Name 2,3 Name 3" — strip trailing affiliation markers from names
    # Only apply to first few lines (author block)
    lines = text.split("\n")
    for i in range(min(10, len(lines))):
        # Strip patterns like "1,*" "2,3" "1" after author names
        if re.search(r"[A-Z][a-z]+\s+\d+[,*\d]*", lines[i]) and "##" not in lines[i]:
            lines[i] = re.sub(r"\s+\d+[,*\d]*(?=\s|$)", "", lines[i])
    text = "\n".join(lines)

    # ── Bullet duplication ──
    text = re.sub(r"^-\s*[•·]\s*", "- ", text, flags=re.MULTILINE)

    # ── Whitespace normalization ──
    # Code blocks are already protected as placeholders, safe to normalize all
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ── Fix LaTeXML spacing around punctuation ──
    # LaTeXML often inserts space before . , ; : ) and after (
    text = re.sub(r" ([.,;:)\]])", r"\1", text)
    text = re.sub(r"([\[(]) ", r"\1", text)

    # Restore math (clean issues inside math blocks)
    for key, val in math_phs:
        val = re.sub(r"%.*?\n", "\n", val)  # strip % line comments
        val = val.rstrip("%")
        # For display math: fix rendering issues
        if val.startswith("$$") and val.endswith("$$"):
            inner = val[2:-2]
            # Remove nested $...$ that break rendering
            inner = re.sub(r"(?<!\$)\$(?!\$)(.*?)\$(?!\$)", r"\1", inner)
            # Collapse to single line (multi-line $$ breaks remark-math)
            inner = inner.replace("\n", " ")
            val = f"$${inner}$$"
        text = text.replace(key, val)

    # Remove empty headings
    text = re.sub(r"^#{1,4}\s*$", "", text, flags=re.MULTILINE)

    # Clean residual \Xhline and similar table commands (remove whole line)
    text = re.sub(r"^\\[Xx]hline[\d.]*(?:pt)?\s*$", "", text, flags=re.MULTILINE)

    # Fix markdown tables: remove blank lines between table rows
    # (blank lines break markdown table rendering)
    lines = text.split("\n")
    result_lines = []
    for i, line in enumerate(lines):
        if not line.strip():
            # Check if this blank line is between two table rows (lines with |)
            prev_is_table = i > 0 and "|" in lines[i-1]
            next_is_table = i < len(lines) - 1 and "|" in lines[i+1]
            if prev_is_table and next_is_table:
                continue  # skip blank line within table
        result_lines.append(line)
    text = "\n".join(result_lines)

    # Restore code blocks (algorithm pseudocode with indentation)
    for key, val in code_phs:
        text = text.replace(key, val)

    return text.strip()


def _residual_frac(text: str) -> float:
    """Fraction of non-math characters that are LaTeX commands."""
    no_math = text
    for pat in _DISPLAY_MATH_PATS:
        no_math = pat.sub("", no_math)
    no_math = _INLINE_MATH_RE.sub("", no_math)
    if not no_math:
        return 0.0
    latex_chars = sum(
        len(m.group()) for m in re.finditer(r"\\[a-zA-Z]+(?:\{[^}]*\})*", no_math)
    )
    return latex_chars / len(no_math)


@register_filter("arxiv")
class ArxivFilter(BaseFilter):
    """Arxiv clean-then-judge filter.

    Step 1: Clean citations, footnotes, section numbering, residual LaTeX.
    Step 2: Reject if residual LaTeX fraction exceeds threshold.
    Step 3: Reject if structural checks fail.
    """

    def __init__(self, text_field: str = "text", **kwargs: Any) -> None:
        super().__init__(text_field=text_field, **kwargs)

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        cleaned = _clean_text(text)
        doc[self.text_field] = cleaned

        max_residual = self.params.get("max_latex_residual", 0.05)
        frac = _residual_frac(cleaned)
        if frac > max_residual:
            return False, {"filter": self.name, "rule": "latex_residual",
                           "value": round(frac, 4), "threshold": max_residual}

        if self.params.get("require_abstract", False) and not _ABSTRACT_RE.search(cleaned):
            return False, {"filter": self.name, "rule": "missing_abstract"}

        min_sections = self.params.get("min_sections", 2)
        num_sections = len(_HEADING_RE.findall(cleaned))
        if num_sections < min_sections:
            return False, {"filter": self.name, "rule": "too_few_sections",
                           "value": num_sections, "threshold": min_sections}

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        text = self.get_text(doc)
        failures: list[dict] = []

        cleaned = _clean_text(text)
        doc[self.text_field] = cleaned

        max_residual = self.params.get("max_latex_residual", 0.05)
        frac = _residual_frac(cleaned)
        if frac > max_residual:
            failures.append({"filter": self.name, "rule": "latex_residual",
                             "value": round(frac, 4), "threshold": max_residual})

        if self.params.get("require_abstract", False) and not _ABSTRACT_RE.search(cleaned):
            failures.append({"filter": self.name, "rule": "missing_abstract",
                             "value": False, "threshold": True})

        min_sections = self.params.get("min_sections", 2)
        num_sections = len(_HEADING_RE.findall(cleaned))
        if num_sections < min_sections:
            failures.append({"filter": self.name, "rule": "too_few_sections",
                             "value": num_sections, "threshold": min_sections})

        return len(failures) == 0, failures
