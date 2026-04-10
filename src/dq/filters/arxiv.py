"""Arxiv-specific quality filter.

After LaTeXML-based conversion in the ingestion stage, text is mostly clean.
This filter handles all data cleaning: citation artifacts, residual LaTeX
commands, footnote markers, etc. Then rejects docs with too many residuals
or missing structure.
"""

import re
from typing import Any

from dq.filters.base import BaseFilter
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


def _protect_math(text: str):
    """Replace math regions with placeholders."""
    phs: list[tuple[str, str]] = []
    ctr = [0]

    def _ph(m: re.Match) -> str:
        key = f"\x00M{ctr[0]}\x00"
        phs.append((key, m.group()))
        ctr[0] += 1
        return key

    for pat in _DISPLAY_MATH_PATS:
        text = pat.sub(_ph, text)
    text = re.sub(r"(?<!\$)\$(?!\$)(?!\s)[^$\n]+?\$(?!\$)", _ph, text)
    return text, phs


def _clean_text(text: str) -> str:
    """Clean LaTeXML-converted text: remove citation artifacts, residual
    commands, footnote markers, and normalize formatting."""

    text, math_phs = _protect_math(text)

    # --- Citation artifacts ---
    # Citation keys like "touvron2023llama2" left after <cite> removal
    text = re.sub(
        r"\s*\b[a-z]+\d{4}[a-z]*\b(?:\s*[,;]\s*\b[a-z]+\d{4}[a-z]*\b)*",
        " ", text,
    )
    # Empty parens/brackets from removed citations: (, ) (;) [ ] etc.
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
    text = re.sub(r"\[\s*[,;]?\s*\]", "", text)
    text = re.sub(r",\s*\)", ")", text)
    text = re.sub(r"\(\s*,", "(", text)

    # --- Footnote artifacts ---
    # Repeated footnote numbers "1 1 1" from LaTeXML superscripts
    text = re.sub(r"(\d+)(?:\s+\1){1,}", r"\1", text)

    # --- Residual LaTeX commands ---
    text = re.sub(r"\\(?:begin|end)\{[^}]+\}", "", text)
    text = re.sub(r"\\[a-zA-Z]{2,}\b", "", text)

    # --- Bullet duplication ---
    text = re.sub(r"^-\s*[•·]\s*", "- ", text, flags=re.MULTILINE)

    # --- Whitespace normalization ---
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    for key, val in math_phs:
        text = text.replace(key, val)
    return text.strip()


def _residual_frac(text: str) -> float:
    """Fraction of non-math characters that are LaTeX commands."""
    no_math = text
    for pat in [_DISPLAY_MATH_PATS[0], _DISPLAY_MATH_PATS[1], _INLINE_MATH_RE]:
        no_math = pat.sub("", no_math)
    if not no_math:
        return 0.0
    latex_chars = sum(
        len(m.group()) for m in re.finditer(r"\\[a-zA-Z]+(?:\{[^}]*\})*", no_math)
    )
    return latex_chars / len(no_math)


@register_filter("arxiv")
class ArxivFilter(BaseFilter):
    """Arxiv clean-then-judge filter.

    Step 1: Clean citation artifacts, residual LaTeX commands, footnote markers.
    Step 2: Reject if residual LaTeX fraction still exceeds threshold.
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
