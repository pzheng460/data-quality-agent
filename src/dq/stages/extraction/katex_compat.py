"""KaTeX compatibility layer — fix non-KaTeX LaTeX commands in math.

Applies to all $...$ and $$...$$ regions in the final text output.
Converts commands from common LaTeX packages to KaTeX-supported equivalents.
"""

from __future__ import annotations

import re

# ── Simple replacements: \oldcmd → \newcmd ──
# Maps non-KaTeX commands to KaTeX-supported equivalents.
_SIMPLE_REPLACEMENTS: dict[str, str] = {
    # bbm / dsfont packages → \mathbb
    "\\mathbbm": "\\mathbb",
    "\\mathds": "\\mathbb",
    # stmaryrd package
    "\\llbracket": "[\\![",
    "\\rrbracket": "]\\!]",
    # amsmath extras
    "\\defeq": "\\coloneqq",
    "\\eqdef": "\\eqqcolon",
    "\\coloneq": "\\coloneqq",
    # font commands not in KaTeX
    "\\textsc": "\\text",
    "\\textsf": "\\text",
    "\\texttt": "\\text",
    "\\textnormal": "\\text",
    "\\mathnormal": "\\mathit",
    # bm package → boldsymbol
    "\\bm": "\\boldsymbol",
    # obsolete/non-standard
    "\\cal": "\\mathcal",
    "\\Bbb": "\\mathbb",
    "\\bold": "\\mathbf",
    "\\rm": "\\mathrm",
    "\\it": "\\mathit",
    "\\sf": "\\mathsf",
    "\\tt": "\\mathtt",
    "\\bf": "\\mathbf",
    # physics / diffcoeff packages
    "\\dd": "d",
    "\\dv": "\\frac{d}{d",  # approximate
    # xcolor
    "\\textcolor": "\\color",
    # misc
    "\\texorpdfstring": "",
    "\\ensuremath": "",
    "\\protect": "",
    "\\nobreakspace": "~",
    "\\allowbreak": "",
}

# Commands to simply remove (no replacement)
_STRIP_COMMANDS: set[str] = {
    "\\label",
    "\\tag",
    "\\nonumber",
    "\\notag",
    "\\vspace",
    "\\hspace",
    "\\phantom",
    "\\vphantom",
    "\\hphantom",
    "\\smash",
    "\\centering",
    "\\raggedright",
    "\\raggedleft",
    "\\footnotesize",
    "\\scriptsize",
    "\\small",
    "\\normalsize",
    "\\large",
    "\\Large",
    "\\LARGE",
    "\\huge",
    "\\Huge",
    "\\displaystyle",
    "\\textstyle",
    "\\scriptstyle",
    "\\scriptscriptstyle",
}

# Prefix artifacts: \v + real command → strip \v prefix
_V_PREFIX_COMMANDS = re.compile(
    r"\\v(math[a-z]+|tilde|hat|bar|dot|vec|widetilde|widehat)\b"
)

# LaTeXML internal commands: \xxx@yyy
_INTERNAL_CMD = re.compile(r"\\[a-zA-Z]*@[a-zA-Z@]*(?:\{[^}]*\})*")

# Package-specific aliases: \mathbbm → \mathbb, etc.
_PACKAGE_ALIASES = {
    "\\mathbbm": "\\mathbb",
    "\\mathscr": "\\mathcal",  # fallback if mathrsfs not in KaTeX
    "\\operatornamewithlimits": "\\operatorname",
    "\\numproduct": "\\prod",
    "\\nsum": "\\sum",
}


def make_katex_compatible(text: str) -> str:
    """Fix all known KaTeX incompatibilities in a text string.

    Applies to math regions ($...$, $$...$$) and also to any
    stray LaTeX commands in the text.
    """
    # 1. Package aliases (exact string replacement)
    for old, new in _PACKAGE_ALIASES.items():
        text = text.replace(old, new)

    # 2. Simple command replacements
    for old, new in _SIMPLE_REPLACEMENTS.items():
        if old in text:
            text = text.replace(old, new)

    # 3. Strip \v prefix from compound commands
    text = _V_PREFIX_COMMANDS.sub(r"\\\1", text)

    # 4. Remove LaTeXML internal commands (\xxx@yyy)
    text = _INTERNAL_CMD.sub("", text)

    # 5. Strip commands that should be removed with their brace arg
    for cmd in _STRIP_COMMANDS:
        # \cmd{arg} → arg  (keep content)
        text = re.sub(rf"{re.escape(cmd)}\{{([^}}]*)\}}", r"\1", text)
        # \cmd without arg → remove
        text = re.sub(rf"{re.escape(cmd)}(?![a-zA-Z])", "", text)

    # 6. Remove \label{...} (common in extracted display math)
    text = re.sub(r"\\label\{[^}]*\}", "", text)

    return text
