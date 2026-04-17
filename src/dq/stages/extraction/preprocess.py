r"""LaTeX pre-processor — neutralize environments LaTeXML handles poorly.

Called BEFORE LaTeXML. Extracts problematic environments, replaces them
with placeholders so LaTeXML only processes what it's good at. After
LaTeXML + html_to_markdown, placeholders are restored with properly
formatted content.

Flow:
  raw .tex → preprocess_tex() → cleaned .tex + PreprocessResult
  cleaned .tex → LaTeXML → HTML → html_to_markdown() → text
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
    _macros: dict = field(default_factory=dict, repr=False)
    _figures: list = field(default_factory=list, repr=False)

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


def _extract_macros(tex: str) -> dict[str, tuple[int, str]]:
    r"""Extract \newcommand / \def definitions from preamble.

    Returns {name: (num_args, body)} for simple macros (0-2 args).
    """
    macros: dict[str, tuple[int, str]] = {}
    # \newcommand{\name}[nargs]{body} or \newcommand{\name}{body}
    for m in re.finditer(
        r"\\(?:newcommand|renewcommand)\s*\{\\(\w+)\}\s*(?:\[(\d+)\])?\s*\{",
        re.split(r"\\begin\{document\}", tex := "")[0] if False else "",  # placeholder
    ):
        pass  # Will be filled below

    # Parse from the actual tex
    preamble_end = tex.find("\\begin{document}") if "\\begin{document}" in tex else len(tex)
    preamble = tex[:preamble_end]

    for m in re.finditer(r"\\(?:newcommand|renewcommand)\s*\{?\\(\w+)\}?\s*(?:\[(\d+)\])?\s*\{", preamble):
        name = m.group(1)
        nargs = int(m.group(2)) if m.group(2) else 0
        # Find matching closing brace
        start = m.end()
        depth = 1
        for i in range(start, min(start + 500, len(preamble))):
            if preamble[i] == '{':
                depth += 1
            elif preamble[i] == '}':
                depth -= 1
                if depth == 0:
                    body = preamble[start:i]
                    macros[name] = (nargs, body)
                    break
    # Also handle \def\name{body} (no args)
    for m in re.finditer(r"\\def\\(\w+)\s*\{", preamble):
        name = m.group(1)
        start = m.end()
        depth = 1
        for i in range(start, min(start + 500, len(preamble))):
            if preamble[i] == '{':
                depth += 1
            elif preamble[i] == '}':
                depth -= 1
                if depth == 0:
                    macros[name] = (0, preamble[start:i])
                    break
    return macros


def _expand_macros(text: str, macros: dict[str, tuple[int, str]], max_passes: int = 5) -> str:
    """Expand user-defined macros in math text. Handles nested braces."""
    for _ in range(max_passes):
        changed = False
        for name, (nargs, body) in macros.items():
            if nargs == 0:
                old = f"\\{name}"
                # Avoid partial matches: \bx should not match inside \bxyz
                pat = rf"\\{re.escape(name)}(?![a-zA-Z])"
                if re.search(pat, text):
                    text = re.sub(pat, lambda _: body, text)
                    changed = True
            else:
                # Find \name followed by nargs brace groups (nested-safe)
                search_pat = rf"\\{re.escape(name)}\{{"
                idx = 0
                while True:
                    m = re.search(search_pat, text[idx:])
                    if not m:
                        break
                    pos = idx + m.start()
                    arg_start = idx + m.end() - 1  # at the {
                    args = []
                    cur = arg_start
                    for _ in range(nargs):
                        arg, end = _match_brace(text, cur)
                        if arg is None:
                            break
                        args.append(arg)
                        cur = end + 1
                    if len(args) == nargs:
                        # Build replacement
                        repl = body
                        for ai, arg in enumerate(args):
                            repl = repl.replace(f"#{ai+1}", arg)
                        text = text[:pos] + repl + text[cur:]
                        changed = True
                    else:
                        idx = pos + 1
        if not changed:
            break
    return text


def _match_brace(text: str, pos: int) -> tuple[str | None, int]:
    """Match {content} starting at pos, handling nesting. Returns (content, end_pos)."""
    if pos >= len(text) or text[pos] != '{':
        return None, pos
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[pos+1:i], i
    return None, pos


def extract_figures(tex: str, figure_paths: dict[str, str] | None = None) -> list[dict]:
    r"""Find \begin{figure}...\end{figure} blocks and extract their components.

    For each figure block, returns a dict with:
      - graphics: list of \includegraphics filenames (stripped of path/ext)
      - caption:  merged \caption{...} text (may be empty)
      - label:    first \label{fig:...} inside the block (may be None)
      - start:    char offset in input tex (for in-place replacement)
      - end:      end offset
      - resolved: best guess at on-disk path (if figure_paths maps stem->path)
    """
    figures: list[dict] = []
    for m in re.finditer(
        r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}",
        tex, re.DOTALL,
    ):
        body = m.group(1)
        graphics = []
        for g in re.finditer(r"\\includegraphics(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}", body):
            graphics.append(g.group(1).strip())
        cap_m = re.search(r"\\caption\s*(?:\[[^\]]*\])?\s*\{(.*?)\}\s*(?:%|\n|$)", body, re.DOTALL)
        caption = _clean_latex(cap_m.group(1)) if cap_m else ""
        lab_m = re.search(r"\\label\s*\{(fig:[^}]+)\}", body)
        label = lab_m.group(1) if lab_m else None

        resolved: list[str] = []
        fmap = figure_paths or {}
        for g in graphics:
            stem = _strip_ext(g)
            hit = fmap.get(g) or fmap.get(stem)
            if hit:
                resolved.append(hit)

        figures.append({
            "start": m.start(), "end": m.end(),
            "graphics": graphics, "caption": caption, "label": label,
            "resolved_paths": resolved,
        })
    return figures


def caption_escape(s: str) -> str:
    """Escape characters that would break Markdown image alt text."""
    return s.replace("]", "\\]").replace("[", "\\[").replace("\n", " ")


def _strip_ext(name: str) -> str:
    for ext in (".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def _clean_latex(s: str) -> str:
    r"""Strip common LaTeX commands from short text like captions."""
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\ref\{[^}]*\}", "", s)
    s = re.sub(r"\\cite\w*\{[^}]*\}", "", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def preprocess_tex(tex: str, figure_paths: dict[str, str] | None = None) -> PreprocessResult:
    """Pre-process LaTeX to neutralize environments LaTeXML can't handle."""
    r = PreprocessResult(tex=tex)

    # ── 0. Extract user-defined macros for later expansion ──
    r._macros = _extract_macros(tex)

    # ── 0a. Figure blocks → Markdown image references (before anything else) ──
    # We do this BEFORE other passes so tikz-removal, algorithm extraction etc.
    # don't eat our \includegraphics or \caption{}.
    figures = extract_figures(r.tex, figure_paths=figure_paths)
    # Process in reverse so slicing the string doesn't invalidate later offsets.
    for fig in reversed(figures):
        caption = fig["caption"]
        # Pick the first resolved image path; fall back to literal graphics name.
        if fig["resolved_paths"]:
            ref = fig["resolved_paths"][0]
        elif fig["graphics"]:
            ref = fig["graphics"][0]
        else:
            ref = ""
        if not ref:
            continue
        md = f"\n\n![{caption_escape(caption)}]({ref})\n\n"
        tag = r.add(md)
        r.tex = r.tex[: fig["start"]] + tag + r.tex[fig["end"] :]
    # Stash figure metadata on the result so the extractor can attach it to doc.
    r._figures = figures

    # ── 1. Algorithm blocks → parsed pseudocode ──
    from dq.stages.extraction.algorithm import extract_algorithms_from_tex
    algos = extract_algorithms_from_tex(r.tex)
    for _cap, _lab, code in algos:
        # Expand user macros in algorithm math ($...$)
        if r._macros:
            code = re.sub(
                r"\$([^$]+)\$",
                lambda m: "$" + _expand_macros(m.group(1), r._macros) + "$",
                code,
            )
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
            # Expand user-defined macros so KaTeX can render
            body = _expand_macros(body, r._macros)
            from dq.stages.extraction.katex_compat import make_katex_compatible
            body = make_katex_compatible(body)
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
