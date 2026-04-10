"""Arxiv-specific clean-then-judge filter.

Uses pylatexenc for robust LaTeX-to-text conversion, preserving math regions.
Then applies lightweight post-processing for arxiv-specific artifacts.

Follows the same in-place mutation pattern as C4Filter and PIIFilter.
"""

import re
from typing import Any

from pylatexenc.latex2text import LatexNodes2Text

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

# ── pylatexenc converter (math kept as-is) ──

_L2T = LatexNodes2Text(
    math_mode="verbatim",       # Keep $...$ and $$...$$ as LaTeX source
    strict_latex_spaces=False,
)

# ── Post-processing patterns ──

# pylatexenc converts \cite → <cit.> — remove these
_CITE_ARTIFACT_RE = __import__("re").compile(r"<cit\.?>")

# pylatexenc converts \includegraphics → "< g r a p h i c s >" — remove
_GRAPHICS_ARTIFACT_RE = __import__("re").compile(r"<\s*g\s*r\s*a\s*p\s*h\s*i\s*c\s*s\s*>")

# [Figure] / [Table] placeholders from tex_to_text preprocessing
_PLACEHOLDER_RE = __import__("re").compile(r"^\[(?:Figure|Table)\]\s*$", re.MULTILINE)

# LaTeX environment option remnants: [leftmargin=10pt]
_ENV_OPTION_RE = __import__("re").compile(r"^\[[\w=.,\s]+\]\s*$", re.MULTILINE)

# Orphaned braces and column specs
_ORPHAN_BRACE_RE = __import__("re").compile(r"^\s*[{}]\s*$", re.MULTILINE)
_COL_SPEC_RE = __import__("re").compile(r"@\{[^}]*\}")
_WRAPFIG_ARGS_RE = __import__("re").compile(
    r"\{[rlc]\}\s*\{[\d.]+\\?(?:textwidth|linewidth|cm|in|pt|em)\}"
)

# LaTeX table formatting words that leak through
_TABLE_JUNK_RE = __import__("re").compile(
    r"\b(?:toprule|midrule|bottomrule|hline|cline|arraystretch|tabcolsep|captionof)\b"
)

# Table & separators → |
_AMP_RE = __import__("re").compile(r"\s*&\s*")

# Cleanup
_MULTI_BLANK_RE = __import__("re").compile(r"\n{3,}")
_LEADING_WS_RE = re.compile(r"^[ \t]+", re.MULTILINE)

# Structure detection
_ABSTRACT_RE = re.compile(r"(?:^|\n)#+\s*abstract", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#+\s+", re.MULTILINE)


def _clean_latex(text: str) -> str:
    """Clean LaTeX from text using pylatexenc + light post-processing."""

    # Step 1: pylatexenc handles the heavy lifting
    # It strips \textbf → content, \cite → <cit.>, \noindent → "", etc.
    # Math ($...$, \begin{equation}...) kept verbatim
    try:
        cleaned = _L2T.latex_to_text(text)
    except Exception:
        # If pylatexenc fails to parse, fall back to the raw text
        cleaned = text

    # Step 2: Post-process artifacts pylatexenc leaves behind
    cleaned = _CITE_ARTIFACT_RE.sub("", cleaned)
    cleaned = _GRAPHICS_ARTIFACT_RE.sub("", cleaned)
    cleaned = _PLACEHOLDER_RE.sub("", cleaned)
    cleaned = _ENV_OPTION_RE.sub("", cleaned)
    cleaned = _ORPHANED_BRACE_RE.sub("", cleaned)

    # Table / figure formatting junk
    cleaned = _COL_SPEC_RE.sub("", cleaned)
    cleaned = _WRAPFIG_ARGS_RE.sub("", cleaned)
    cleaned = _TABLE_JUNK_RE.sub("", cleaned)
    # pandoc may leave \textwidth, \linewidth etc as literal text
    cleaned = re.sub(r"\b(?:textwidth|linewidth|columnwidth|paperwidth)\b", "", cleaned)

    # Step 3: Regex fallback for commands pylatexenc didn't recognize
    # Protect math regions first
    import re as _re
    _math_phs: list[tuple[str, str]] = []
    _ctr = [0]
    def _ph(m: _re.Match) -> str:
        key = f"\x00M{_ctr[0]}\x00"; _math_phs.append((key, m.group())); _ctr[0] += 1; return key
    cleaned = _re.sub(r"\$\$.*?\$\$", _ph, cleaned, flags=_re.DOTALL)
    cleaned = _re.sub(r"(?<!\$)\$(?!\$)(?!\s)[^$\n]+?\$(?!\$)", _ph, cleaned)
    cleaned = _re.sub(r"\\begin\{(?:equation|align|alignat|gather|multline)\*?\}.*?\\end\{(?:equation|align|alignat|gather|multline)\*?\}", _ph, cleaned, flags=_re.DOTALL)

    # Remove remaining \command{...} patterns (citations, refs, etc.)
    cleaned = _re.sub(r"\\(?:cite[pt]?|nocite|citealp|citeauthor|citet|citep)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}", "", cleaned)
    cleaned = _re.sub(r"\\(?:ref|eqref|autoref|hyperref|pageref|nameref|label)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}", "", cleaned)
    # Remove remaining \command{arg}{arg} with up to 2 args
    cleaned = _re.sub(r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])*(?:\s*\{[^}]*\})+", "", cleaned)
    # Strip backslash from remaining \word → keep word
    cleaned = _re.sub(r"\\([a-zA-Z]{2,})\b", r"\1", cleaned)

    # Restore math placeholders
    for key, val in _math_phs:
        cleaned = cleaned.replace(key, val)

    # Typography
    cleaned = re.sub(r"``", "\u201c", cleaned)
    cleaned = re.sub(r"''", "\u201d", cleaned)
    cleaned = re.sub(r"(?<!-)--(?!-)", "\u2013", cleaned)

    # Cleanup whitespace
    cleaned = _LEADING_WS_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned)
    cleaned = cleaned.strip()

    return cleaned


_ORPHANED_BRACE_RE = __import__("re").compile(r"^\s*[{}]\s*$", re.MULTILINE)


def _residual_frac(text: str) -> float:
    """Fraction of non-math characters that are LaTeX commands."""
    no_math = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    no_math = re.sub(r"\\\[.*?\\\]", "", no_math, flags=re.DOTALL)
    no_math = re.sub(r"\$[^$]+?\$", "", no_math)
    if not no_math:
        return 0.0
    latex_chars = sum(len(m.group()) for m in re.finditer(r"\\[a-zA-Z]+(?:\{[^}]*\})*", no_math))
    return latex_chars / len(no_math)


@register_filter("arxiv")
class ArxivFilter(BaseFilter):
    """Arxiv clean-then-judge filter using pylatexenc.

    Step 1: Clean LaTeX via pylatexenc (preserves math, strips commands).
    Step 2: Reject if residual fraction still exceeds threshold.
    Step 3: Reject if structural checks fail.
    """

    def __init__(self, text_field: str = "text", **kwargs: Any) -> None:
        super().__init__(text_field=text_field, **kwargs)

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        cleaned = _clean_latex(text)
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

        cleaned = _clean_latex(text)
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


_RESIDUAL_MATH_RE = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"\\begin\{(?:equation|align|alignat|gather|multline|eqnarray)\*?\}.*?\\end\{(?:equation|align|alignat|gather|multline|eqnarray)\*?\}", re.DOTALL),
]
