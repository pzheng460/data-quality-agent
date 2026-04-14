r"""LaTeX pre-processor — neutralize environments LaTeXML handles poorly.

Called BEFORE LaTeXML. Extracts problematic environments, replaces them
with placeholders so LaTeXML only processes what it's good at. After
LaTeXML + html_to_text, placeholders are restored with properly
formatted content.

Flow:
  raw .tex → preprocess_tex() → cleaned .tex + PreprocessResult
  cleaned .tex → LaTeXML → HTML → html_to_text() → text
  text → restore_placeholders(text, result) → final text
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PH = "DQPLACEHOLDER"


@dataclass
class PreprocessResult:
    """Modified tex + extracted content."""
    tex: str
    placeholders: dict[str, str] = field(default_factory=dict)
    _counter: int = field(default=0, repr=False)

    def add(self, content: str) -> str:
        """Register content, return placeholder tag."""
        self._counter += 1
        tag = f"[{_PH}{self._counter}]"
        self.placeholders[tag] = content
        return tag


# ── Environments LaTeXML consistently fails on ──

_TIKZ_ENVS = {"tikzpicture", "pgfonlayer", "axis", "groupplot", "tikzcd"}
_MATH_ENVS = {"align", "align*", "alignat", "alignat*",
              "gather", "gather*", "multline", "multline*",
              "flalign", "flalign*", "eqnarray", "eqnarray*"}
_FRAME_ENVS = {"mdframed", "tcolorbox"}


def preprocess_tex(tex: str) -> PreprocessResult:
    """Pre-process LaTeX to neutralize environments LaTeXML can't handle."""
    r = PreprocessResult(tex=tex)

    # ── 1. Algorithm blocks → parsed pseudocode ──
    from dq.stages.extraction.algorithm import extract_algorithms_from_tex
    algos = extract_algorithms_from_tex(r.tex)
    for _cap, _lab, code in algos:
        tag = r.add(f"\n```\n{code}\n```\n")
        # Replace the first remaining \begin{algorithm}...\end{algorithm}
        r.tex = _replace_first_env(r.tex, "algorithm", tag)

    # ── 2. tikz/pgfplots → remove (images, no text value) ──
    for env in _TIKZ_ENVS:
        r.tex = re.sub(
            rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
            "", r.tex, flags=re.DOTALL,
        )
    # Also remove standalone \begin{tikzpicture}...\end{tikzpicture} inside figures
    r.tex = re.sub(r"\\pgfplotsset\{[^}]*\}", "", r.tex)

    # ── 3. Failed math environments → preserve as $$...$$ ──
    for env in _MATH_ENVS:
        esc = re.escape(env)
        # Process in reverse order so indices don't shift
        for m in reversed(list(re.finditer(
            rf"\\begin\{{{esc}\}}(.*?)\\end\{{{esc}\}}",
            r.tex, re.DOTALL,
        ))):
            body = m.group(1).strip()
            body = re.sub(r"\\textsc\{([^}]*)\}", r"\\text{\1}", body)
            if "&" in body or "\\\\" in body:
                formatted = f"$$\\begin{{aligned}}{body}\\end{{aligned}}$$"
            else:
                formatted = f"$${body}$$"
            tag = r.add(formatted)
            r.tex = r.tex[:m.start()] + tag + r.tex[m.end():]

    # ── 4. mdframed/tcolorbox → unwrap (keep content, strip frame) ──
    for env in ("mdframed", "tcolorbox"):
        r.tex = re.sub(rf"\\begin\{{{env}\}}(?:\[[^\]]*\])?", "", r.tex)
        r.tex = re.sub(rf"\\end\{{{env}\}}", "", r.tex)

    return r


def restore_placeholders(text: str, prep: PreprocessResult) -> str:
    """Replace placeholder tags in final output with extracted content."""
    for tag, content in prep.placeholders.items():
        text = text.replace(tag, content)
    return text


def _replace_first_env(tex: str, env: str, replacement: str) -> str:
    """Replace the first \\begin{env}...\\end{env} with replacement text."""
    pat = rf"\\begin\{{{re.escape(env)}\}}(?:\[[^\]]*\])?\s*\n?.*?\\end\{{{re.escape(env)}\}}"
    return re.sub(pat, replacement, tex, count=1, flags=re.DOTALL)
