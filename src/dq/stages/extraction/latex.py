"""LaTeX extractor — converts raw LaTeX to text via LaTeXML → HTML → text."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from dq.stages.extraction.base import Extractor
from dq.stages.extraction.registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor("latex")
class LatexExtractor(Extractor):
    """Convert raw LaTeX to clean text via LaTeXML."""

    input_format = "latex"

    def extract(self, doc: dict) -> dict | None:
        raw = doc.get("text", "")
        if not raw:
            return None

        meta = doc.get("metadata", {})
        title = meta.get("title", "Untitled")

        # If title looks like an arxiv ID, try to extract from LaTeX source
        if re.match(r"^\d{4}\.\d+$", title):
            tex_title = _extract_title_from_tex(raw)
            if tex_title:
                title = tex_title
                meta["title"] = title

        text = _latexml_convert(raw, title)
        if text is None or len(text) < 200:
            return None

        doc["text"] = text
        return doc


def _latexml_convert(tex: str, title: str) -> str | None:
    """LaTeXML pipeline: preprocess → LaTeXML → html_to_markdown → restore."""
    import tempfile
    from dq.stages.extraction.preprocess import preprocess_tex, restore_placeholders

    # Pre-process: extract environments LaTeXML can't handle
    prep = preprocess_tex(tex)
    cleaned_tex = prep.tex

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = f"{tmpdir}/paper.tex"
            xml_path = f"{tmpdir}/paper.xml"
            html_path = f"{tmpdir}/paper.html"

            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(cleaned_tex)

            # Step 1: LaTeX → XML
            r1 = subprocess.run(
                ["latexml", "--dest", xml_path, tex_path],
                capture_output=True, timeout=120,
            )
            if r1.returncode != 0 or not Path(xml_path).exists():
                logger.warning("latexml failed: %s", r1.stderr[:500] if r1.stderr else "unknown")
                return _fallback(tex, title)

            # Step 2: XML → HTML
            r2 = subprocess.run(
                ["latexmlpost", "--dest", html_path, "--format=html5",
                 "--nodefaultresources", xml_path],
                capture_output=True, timeout=60,
            )
            if r2.returncode != 0 or not Path(html_path).exists():
                logger.warning("latexmlpost failed: %s", r2.stderr[:500] if r2.stderr else "unknown")
                return _fallback(tex, title)

            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()

            from dq.stages.extraction.html import html_to_markdown
            text = html_to_markdown(html, raw_tex=tex, macros=prep._macros)

            # Restore placeholders (algorithms, math, etc.)
            text = restore_placeholders(text, prep)
            return text

    except FileNotFoundError:
        logger.warning("latexml not installed, using fallback")
        return _fallback(tex, title)
    except subprocess.TimeoutExpired:
        logger.warning("latexml timed out, using fallback")
        return _fallback(tex, title)
    except Exception as e:
        logger.warning("latexml error: %s", e)
        return _fallback(tex, title)


def _extract_title_from_tex(tex: str) -> str:
    r"""Extract title from \title{...} or conference-specific variants."""
    # Try common title commands (icmltitle, neuripsTitle, etc.)
    for cmd in (r"icmltitlerunning", r"icmltitle", r"title"):
        m = re.search(rf"\\{cmd}(?:\[[^\]]*\])?\{{", tex)
        if m:
            break
    else:
        return ""
    # Match nested braces from the opening {
    start = m.end()
    depth = 1
    for i in range(start, min(start + 500, len(tex))):
        if tex[i] == '{':
            depth += 1
        elif tex[i] == '}':
            depth -= 1
            if depth == 0:
                title = tex[start:i]
                # Clean LaTeX commands: \textbf{X} → X, \alg{} → remove
                title = re.sub(r"\\texorpdfstring\{[^}]*\}\{([^}]*)\}", r"\1", title)
                title = re.sub(r"\\includegraphics[^}]*\}", "", title)
                title = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", title)
                title = re.sub(r"\\[a-zA-Z]+", "", title)
                title = re.sub(r"[{}]", "", title)
                title = re.sub(r"\s+", " ", title).strip()
                title = re.sub(r"^[:\s]+", "", title)
                return title
    return ""


def _fallback(tex: str, title: str) -> str:
    """Lightweight regex fallback when LaTeXML is unavailable."""
    m = re.search(r"\\begin\{document\}", tex)
    if m:
        tex = tex[m.end():]
    tex = re.sub(r"\\end\{document\}.*", "", tex, flags=re.DOTALL)
    tex = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.MULTILINE)
    tex = re.sub(r"\\(?:maketitle|tableofcontents|newpage|clearpage|pagebreak)\b", "", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return f"# {title}\n\n{tex.strip()}"
