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

# Commands to DELETE entirely (including their {arg})
_DELETE_WITH_ARG: set[str] = {
    "\\label", "\\tag", "\\vspace", "\\hspace",
    "\\phantom", "\\vphantom", "\\hphantom",
}

# Commands to UNWRAP (remove cmd, keep {arg} content)
_UNWRAP_COMMANDS: set[str] = {
    "\\displaystyle", "\\textstyle", "\\scriptstyle", "\\scriptscriptstyle",
    "\\smash",
}

# Commands to STRIP (no arg expected)
_STRIP_NO_ARG: set[str] = {
    "\\nonumber", "\\notag",
    "\\centering", "\\raggedright", "\\raggedleft",
    "\\footnotesize", "\\scriptsize", "\\small", "\\normalsize",
    "\\large", "\\Large", "\\LARGE", "\\huge", "\\Huge",
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
    # 1. Package aliases (word-boundary safe to avoid substring matches)
    for old, new in _PACKAGE_ALIASES.items():
        text = re.sub(rf"{re.escape(old)}(?![a-zA-Z])", lambda _, r=new: r, text)

    # 2. Simple command replacements (word-boundary safe)
    for old, new in _SIMPLE_REPLACEMENTS.items():
        if old in text:
            text = re.sub(rf"{re.escape(old)}(?![a-zA-Z])", lambda _, r=new: r, text)

    # 3. Strip \v prefix from compound commands
    text = _V_PREFIX_COMMANDS.sub(r"\\\1", text)

    # 4. Remove LaTeXML internal commands (\xxx@yyy)
    text = _INTERNAL_CMD.sub("", text)

    # 5. Fix bare accent + command: \hat\mathbf{x} → \hat{\mathbf{x}}
    #    KaTeX requires braces around the argument of accent commands
    text = re.sub(
        r"\\(hat|tilde|bar|dot|ddot|vec|widehat|widetilde|overline|underline)"
        r"\\(mathbf|mathbb|mathcal|mathrm|mathit|mathsf|boldsymbol)\{([^}]*)\}",
        r"\\\1{\\\2{\3}}",
        text,
    )

    # 6. Delete commands with their {arg} entirely (\label{eq:foo} → "")
    for cmd in _DELETE_WITH_ARG:
        text = re.sub(rf"{re.escape(cmd)}\{{[^}}]*\}}", "", text)
        text = re.sub(rf"{re.escape(cmd)}(?![a-zA-Z])", "", text)

    # 6. Unwrap commands, keep {arg} content (\displaystyle{x} → x)
    for cmd in _UNWRAP_COMMANDS:
        text = re.sub(rf"{re.escape(cmd)}\{{([^}}]*)\}}", r"\1", text)
        text = re.sub(rf"{re.escape(cmd)}(?![a-zA-Z])", "", text)

    # 7. Remove standalone commands (no arg)
    for cmd in _STRIP_NO_ARG:
        text = re.sub(rf"{re.escape(cmd)}(?![a-zA-Z])", "", text)

    # 8. Remove environments KaTeX doesn't support (inside aligned/$$)
    text = re.sub(r"\\begin\{split\}", "", text)
    text = re.sub(r"\\end\{split\}", "", text)

    # 9. Fix bare sub/superscript + command: _	ext{x} -> _{	ext{x}}
    text = re.sub(
        r"([_^])\\(text|mathrm|mathbf|mathit|mathcal|operatorname|boldsymbol)\{([^}]*)\}",
        r"\1{\\\2{\3}}",
        text,
    )

    return text
