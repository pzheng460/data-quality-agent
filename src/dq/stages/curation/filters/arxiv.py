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

# LaTeXML renders <cite> as inline text. Common forms after text extraction:
#   "touvron2023llama2" (bare citation key)
#   "[14]" or "[14, 15]" (numeric citations)
#   "(Smith et al., 2023)" (already readable — leave these)
_CITE_KEY_RE = re.compile(
    r"\b[a-z]+\d{4}[a-z]*\b(?:\s*[,;]\s*\b[a-z]+\d{4}[a-z]*\b)*"
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


def _clean_text(text: str) -> str:
    """Clean LaTeXML-converted text. All cleaning logic lives here."""

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

    # ── Residual LaTeX commands ──
    text = re.sub(r"\\(?:begin|end)\{[^}]+\}", "", text)
    text = re.sub(r"\\[a-zA-Z]{2,}\b", "", text)

    # ── Bullet duplication ──
    text = re.sub(r"^-\s*[•·]\s*", "- ", text, flags=re.MULTILINE)

    # ── Whitespace normalization ──
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Restore math
    for key, val in math_phs:
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
