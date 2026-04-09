"""Arxiv-specific clean-then-judge filter.

Aggressively cleans LaTeX residuals from parsed arxiv text while preserving
math content and document structure. Then rejects documents that are still
too noisy or structurally incomplete.

Follows the same in-place mutation pattern as C4Filter and PIIFilter.
"""

import re
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

# ── Math region protection ──

_DISPLAY_MATH_PATTERNS = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"\\begin\{(?:equation|align|alignat|gather|multline|eqnarray|math|displaymath)\*?\}.*?\\end\{(?:equation|align|alignat|gather|multline|eqnarray|math|displaymath)\*?\}", re.DOTALL),
]
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(?!\s)[^$\n]+?\$(?!\$)")

# ── Cleaning patterns (applied outside math regions) ──

# Citations and references → remove
_CITE_RE = re.compile(
    r"\\(?:cite[tp]?|nocite|citealp|citeauthor|citeyear)"
    r"(?:\s*\[[^\]]*\])*"       # optional [...]
    r"\s*\{[^}]*\}"
)
_REF_RE = re.compile(
    r"\\(?:ref|eqref|autoref|cref|Cref|pageref|nameref|hyperref)"
    r"(?:\s*\[[^\]]*\])*"
    r"\s*\{[^}]*\}"
)
_LABEL_RE = re.compile(r"\\label\{[^}]*\}")

# Format commands → keep content
_FORMAT_RE = re.compile(
    r"\\(?:textbf|textit|emph|underline|texttt|textrm|textsf|textsc"
    r"|mathrm|mathbf|mathit|mathcal|mathbb|boldsymbol"
    r"|text)\{([^}]*)\}"
)

# Section / structural commands → convert or remove
_SECTION_CMD_RE = re.compile(
    r"\\(?:section|subsection|subsubsection|paragraph|subparagraph)\*?"
    r"\s*\{([^}]*)\}"
)

# Footnotes → keep content
_FOOTNOTE_RE = re.compile(r"\\footnote\s*\{([^}]*)\}")

# URL → keep the URL text
_URL_RE = re.compile(r"\\(?:url|href)\s*\{([^}]*)\}")

# Include/input → remove
_INPUT_RE = re.compile(r"\\(?:input|include|includeonly)\s*\{[^}]*\}")

# Graphics → remove
_GRAPHICS_RE = re.compile(
    r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{[^}]*\}"
)

# Environment begin/end → remove markers
_ENV_RE = re.compile(r"\\(?:begin|end)\s*\{[^}]*\}")

# Layout / formatting commands → remove
_LAYOUT_CMDS = re.compile(
    r"\\(?:"
    # spacing
    r"noindent|center|centering|raggedright|raggedleft"
    r"|smallskip|medskip|bigskip|vfill|hfill|newline|linebreak"
    r"|pagebreak|newpage|clearpage"
    # size
    r"|footnotesize|scriptsize|tiny|small|normalsize|large|Large|LARGE|huge|Huge"
    # structure
    r"|maketitle|tableofcontents|appendix"
    r"|bibliographystyle|bibliography"
    # misc
    r"|looseness|frenchspacing|sloppy|protect|relax|phantom"
    r")\b"
    r"(?:\s*\{[^}]*\})*"  # optional braced args
)

# \vspace, \hspace, \setlength etc with braced args
_SPACING_RE = re.compile(
    r"\\(?:vspace|hspace|setlength|addtolength|setcounter|addtocounter"
    r"|renewcommand|newcommand|def|let"
    r"|DeclareMathOperator)\*?"
    r"(?:\s*\{[^}]*\})*"
    r"(?:\s*\[[^\]]*\])*"
    r"(?:\s*\{[^}]*\})*"
)

# includegraphics, figures, tables with complex args
_FIGURE_BLOCK_RE = re.compile(
    r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{[^}]*\}", re.DOTALL
)

# \looseness, \parindent etc with = assignments
_ASSIGNMENT_RE = re.compile(r"\\[a-zA-Z]+\s*=\s*-?\d+\s*")

# Commands that should just be removed entirely (no content to keep)
_REMOVE_CMDS = re.compile(
    r"\\(?:thanks|acknowledgments?|keywords?|date|affiliation?|institute|email"
    r"|baselineskip|parskip|itemsep|parsep|topsep|partopsep"
    r"|arraystretch|tabcolsep|renewcommand|newcommand|providecommand"
    r"|def|let|makeatletter|makeatother"
    r"|thispagestyle|pagestyle|pagenumbering"
    r"|setcounter|addtocounter|stepcounter"
    r"|phantom|vphantom|hphantom|strut"
    r")\b"
    r"(?:\s*\{[^}]*\})*"
    r"(?:\s*\[[^\]]*\])*"
    r"(?:\s*\{[^}]*\})*"
)

# \item → bullet point
_ITEM_RE = re.compile(r"\\item\b\s*(?:\[[^\]]*\])?\s*")

# Double backslash (LaTeX line break) → newline
_DOUBLE_BS_RE = re.compile(r"\s*\\\\\s*")

# Any remaining \command that we didn't handle → remove
# (aggressive catch-all, applied last)
_REMAINING_CMD_RE = re.compile(
    r"\\(?![$%&_#{}])[a-zA-Z@]+"
    r"(?:\s*\*)??"
    r"(?:\s*\[[^\]]*\])*"
    r"(?:\s*\{[^}]*\})*"
)

# Cleanup
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_SPACES_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_MULTI_SPACES_RE = re.compile(r"  +")

# Structure detection
_ABSTRACT_RE = re.compile(r"(?:^|\n)#+\s*abstract", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#+\s+", re.MULTILINE)


# ── Tabular conversion ──

_TABULAR_RE = re.compile(
    r"\\begin\{tabular\*?\}(?:\{[^}]*\})?\s*(.*?)\\end\{tabular\*?\}",
    re.DOTALL,
)


def _convert_tabulars(text: str) -> str:
    """Convert \\begin{tabular}...\\end{tabular} to markdown tables."""
    def _convert_one(m: re.Match) -> str:
        body = m.group(1)
        body = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b", "", body)
        body = re.sub(r"\\cline\{[^}]*\}", "", body)
        rows = re.split(r"\\\\", body)
        md_rows = []
        for row in rows:
            row = row.strip()
            if not row:
                continue
            cells = [c.strip() for c in row.split("&")]
            cleaned_cells = []
            for cell in cells:
                cell = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", cell)
                cell = re.sub(r"\\texttt\{([^}]*)\}", r"`\1`", cell)
                cell = re.sub(r"\\textit\{([^}]*)\}", r"*\1*", cell)
                cell = re.sub(r"\\small\b", "", cell)
                cell = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", cell)
                cell = re.sub(r"\\[a-zA-Z]+", "", cell)
                cleaned_cells.append(cell.strip())
            md_rows.append(" | ".join(cleaned_cells))
        if not md_rows:
            return ""
        header = md_rows[0]
        ncols = header.count("|") + 1
        sep = " | ".join(["---"] * ncols)
        lines = [header, sep] + md_rows[1:]
        return "\n" + "\n".join(lines) + "\n"

    return _TABULAR_RE.sub(_convert_one, text)


# ── Cleaning ──

def _protect_math(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace math regions with placeholders to protect them from cleaning."""
    protected: list[tuple[str, str]] = []
    counter = 0

    def _sub(m: re.Match) -> str:
        nonlocal counter
        key = f"\x00MATH{counter}\x00"
        protected.append((key, m.group()))
        counter += 1
        return key

    for pat in _DISPLAY_MATH_PATTERNS:
        text = pat.sub(_sub, text)
    text = _INLINE_MATH_RE.sub(_sub, text)
    return text, protected


def _restore_math(text: str, protected: list[tuple[str, str]]) -> str:
    """Restore math regions from placeholders."""
    for key, val in protected:
        text = text.replace(key, val)
    return text


def _clean_latex(text: str) -> str:
    """Aggressively clean LaTeX residuals while preserving math and readable content."""

    # Protect math regions
    text, math_regions = _protect_math(text)

    # 1. Remove citations and references
    text = _CITE_RE.sub("", text)
    text = re.sub(r"\\(?:ref|eqref|autoref|cref|Cref|pageref|nameref)\s*\{[^}]*\}", "", text)
    text = _LABEL_RE.sub("", text)

    # 2. Convert sections to markdown (if not already converted)
    def _section_to_md(m: re.Match) -> str:
        cmd = m.group(0)
        content = m.group(1)
        if "subsub" in cmd:
            return f"\n#### {content}\n"
        elif "sub" in cmd:
            return f"\n### {content}\n"
        elif "paragraph" in cmd:
            return f"\n**{content}**\n"
        return f"\n## {content}\n"
    text = _SECTION_RE.sub(_section_to_md, text)

    # 3. Footnotes → keep content
    text = _FOOTNOTE_RE.sub(r" (\1)", text)

    # 4. URLs → keep URL
    text = re.sub(r"\\url\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\s*\{([^}]*)\}\s*\{([^}]*)\}", r"\2 (\1)", text)

    # 5. Format commands → keep content
    text = _FORMAT_CMD_RE.sub(r"\1", text)

    # 6. Remove graphics, input, figures
    text = _GRAPHICS_RE.sub("", text)
    text = _INPUT_RE.sub("", text)

    # 7. Remove layout/spacing commands
    text = _LAYOUT_CMDS.sub("", text)
    text = _SPACING_RE.sub("", text)

    # 8. Convert tabular environments to markdown tables
    text = _convert_tabulars(text)

    # 9. Remove environment markers (keep content)
    text = re.compile(r"\\(?:begin|end)\s*\{[^}]*\}").sub("", text)

    # 10. \item → bullet
    text = _ITEM_RE.sub("\n- ", text)

    # 10. \\ (line break) → newline
    text = re.sub(r"\s*\\\\\s*", "\n", text)

    # 11. Remove assignment commands (\looseness=-1 etc)
    text = _ASSIGNMENT_RE.sub("", text)

    # 12. Remove \renewcommand, \newcommand, \def etc
    text = _REMOVE_DEFS_RE.sub("", text)

    # 13. Format commands → keep content
    for _ in range(3):  # Iterate for nested: \textbf{\emph{x}}
        text = _FORMAT_CMD_RE.sub(r"\1", text)

    # 14. Remove \appendix, \maketitle etc standalone commands
    text = _LAYOUT_CMDS.sub("", text)
    text = _REMOVE_CMDS.sub("", text)

    # 15. Remove remaining \command{...}{...} patterns with args
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])*(?:\s*\{[^}]*\})+", "", text)

    # 16. Remove wrapfigure/tabular/minipage environment remnants
    text = re.sub(r"\{[rlc]\}\s*\{[\d.]+\\?(?:textwidth|linewidth|cm|in|pt|em)\}", "", text)

    # 17. Remove LaTeX table formatting
    text = re.sub(r"\b(?:toprule|midrule|bottomrule|hline|cline|arraystretch|tabcolsep|centering|center|textwidth|linewidth|parbox|captionof)\b", "", text)
    # Convert table & separators to | for readability
    text = re.sub(r"\s*&\s*", " | ", text)

    # 18. Clean orphaned tildes (LaTeX non-breaking space) → regular space
    text = re.sub(r"~", " ", text)

    # 19. Remove LaTeX environment option remnants: [leftmargin=10pt] etc.
    text = re.sub(r"^\[[\w=.,\s]+\]\s*$", "", text, flags=re.MULTILINE)

    # 20. Clean up citation/ref artifacts: trailing commas, empty parens
    text = re.sub(r"\(\s*\)", "", text)           # empty ()
    text = re.sub(r"\[\s*\]", "", text)           # empty []
    text = re.sub(r",\s*,", ",", text)            # double commas
    text = re.sub(r"\s*,\s*\)", ")", text)        # ,)
    text = re.sub(r"\(\s*,", "(", text)           # (,

    # 20. Remove leftover \\ (LaTeX line breaks that weren't caught)
    text = re.sub(r"\\\\", "\n", text)

    # 21. Remove leftover single backslash before non-alpha
    text = re.sub(r"\\(?=[^a-zA-Z$\\])", "", text)

    # 22. Strip backslash from remaining \word commands, keep the word
    #     e.g. \mistral → mistral, \llama → llama
    text = re.sub(r"\\([a-zA-Z]{2,})\b", r"\1", text)

    # Restore math
    text = _restore_math(text, math_regions)

    # 23. Remove figure/table placeholders — useless for LLM pretraining
    text = re.sub(r"^\[(?:Figure|Table)\]\s*$", "", text, flags=re.MULTILINE)
    # Remove caption remnants (lines that are just "Figure N:" or "Table N:")
    text = re.sub(r"^(?:Figure|Table)\s*\d*\s*[:.]?\s*$", "", text, flags=re.MULTILINE)

    # Final cleanup
    text = re.sub(r"^[ \t]+", "", text, flags=re.MULTILINE)   # leading whitespace per line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)   # trailing whitespace
    text = re.sub(r" {2,}", " ", text)                         # multiple spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                     # excessive blank lines
    # Remove "Figure ." / "Table ." orphaned refs
    text = re.sub(r"(Figure|Table|Section|Equation|Appendix)\s*\.\s*", r"\1. ", text)
    text = text.strip()

    return text


# Compiled patterns used in _clean_latex
_SECTION_RE = re.compile(
    r"\\(?:section|subsection|subsubsection|paragraph|subparagraph)\*?\s*\{([^}]*)\}"
)
_FORMAT_CMD_RE = re.compile(
    r"\\(?:textbf|textit|emph|underline|texttt|textrm|textsf|textsc"
    r"|mathrm|mathbf|mathit|mathcal|mathbb|boldsymbol|text)\{([^}]*)\}"
)
_FOOTNOTE_RE = re.compile(r"\\footnote\s*\{([^}]*)\}")
_LAYOUT_CMDS = re.compile(
    r"\\(?:noindent|centering|raggedright|raggedleft|looseness"
    r"|smallskip|medskip|bigskip|vfill|hfill"
    r"|newline|linebreak|pagebreak|newpage|clearpage"
    r"|footnotesize|scriptsize|small|normalsize|large|Large|LARGE|huge|Huge"
    r"|maketitle|tableofcontents|appendix"
    r"|bibliographystyle|bibliography"
    r"|frenchspacing|sloppy|protect|relax"
    r")\b"
    r"(?:\s*=\s*-?\d+)?"       # \looseness=-1
    r"(?:\s*\{[^}]*\})*"
)
_REMOVE_CMDS = re.compile(
    r"\\(?:thanks|date|affiliation?|institute|email|keywords?"
    r"|phantom|thispagestyle|pagestyle|pagenumbering"
    r"|arraystretch|tabcolsep|baselineskip|parskip|itemsep"
    r")\b"
    r"(?:\s*\{[^}]*\})*"
)
_REMOVE_DEFS_RE = re.compile(
    r"\\(?:renewcommand|newcommand|def|let|DeclareMathOperator|providecommand"
    r"|makeatletter|makeatother|setcounter|addtocounter"
    r")\b"
    r"(?:\s*\*)??"
    r"(?:\s*\{[^}]*\})*"
    r"(?:\s*\[[^\]]*\])*"
    r"(?:\s*\{[^}]*\})*"
)
_GRAPHICS_RE = re.compile(
    r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{[^}]*\}"
)
_INPUT_RE_COMPILED = re.compile(r"\\(?:input|include|bibliography)\s*\{[^}]*\}")
_REMAINING_CMD_RE = re.compile(
    r"\\[a-zA-Z]+\*?"
    r"(?:\s*\[[^\]]*\])*"
    r"(?:\s*\{[^}]*\})*"
)


def _residual_frac(text: str) -> float:
    """Fraction of non-math characters that are LaTeX commands."""
    # Strip math regions
    no_math = text
    for pat in _DISPLAY_MATH_PATTERNS:
        no_math = pat.sub("", no_math)
    no_math = _INLINE_MATH_RE.sub("", no_math)
    if not no_math:
        return 0.0
    latex_chars = sum(len(m.group()) for m in re.finditer(r"\\[a-zA-Z]+(?:\{[^}]*\})*", no_math))
    return latex_chars / len(no_math)


# _DISPLAY_MATH_PATTERNS defined at top of file

# Structure detection
_ABSTRACT_DETECT = re.compile(r"(?:^|\n)#+\s*abstract", re.IGNORECASE)
_SECTION_DETECT = re.compile(r"^#+\s+", re.MULTILINE)


_CITE_REF_RE = None  # old, not used


@register_filter("arxiv")
class ArxivFilter(BaseFilter):
    """Arxiv clean-then-judge filter.

    Step 1: Aggressively clean LaTeX commands outside math regions.
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
            return False, {
                "filter": self.name, "rule": "latex_residual",
                "value": round(frac, 4), "threshold": max_residual,
            }

        if self.params.get("require_abstract", True) and not _ABSTRACT_DETECT.search(cleaned):
            return False, {"filter": self.name, "rule": "missing_abstract"}

        min_sections = self.params.get("min_sections", 2)
        num_sections = len(_SECTION_DETECT.findall(cleaned))
        if num_sections < min_sections:
            return False, {
                "filter": self.name, "rule": "too_few_sections",
                "value": num_sections, "threshold": min_sections,
            }

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

        if self.params.get("require_abstract", True) and not _ABSTRACT_DETECT.search(cleaned):
            failures.append({"filter": self.name, "rule": "missing_abstract",
                             "value": False, "threshold": True})

        min_sections = self.params.get("min_sections", 2)
        num_sections = len(_SECTION_DETECT.findall(cleaned))
        if num_sections < min_sections:
            failures.append({"filter": self.name, "rule": "too_few_sections",
                             "value": num_sections, "threshold": min_sections})

        return len(failures) == 0, failures
