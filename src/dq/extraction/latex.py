"""LaTeX extractor — converts raw LaTeX to text via LaTeXML → HTML → text."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from dq.extraction.base import Extractor
from dq.extraction.registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor("latex")
class LatexExtractor(Extractor):
    """Convert raw LaTeX to clean text via LaTeXML."""

    input_format = "latex"

    def extract(self, doc: dict) -> dict | None:
        raw = doc.get("text", "")
        if not raw:
            return None

        title = doc.get("metadata", {}).get("title", "Untitled")
        text = _latexml_convert(raw, title)
        if text is None or len(text) < 200:
            return None

        doc["text"] = text
        return doc


def _latexml_convert(tex: str, title: str) -> str | None:
    """LaTeXML pipeline: .tex → .xml → .html → clean text."""
    import subprocess
    import tempfile
    from pathlib import Path

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = f"{tmpdir}/paper.tex"
            xml_path = f"{tmpdir}/paper.xml"
            html_path = f"{tmpdir}/paper.html"

            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex)

            r1 = subprocess.run(
                ["latexml", "--dest", xml_path, tex_path],
                capture_output=True, timeout=120,
            )
            if r1.returncode != 0 or not Path(xml_path).exists():
                logger.warning("latexml failed: %s", r1.stderr[:500])
                return _fallback(tex, title)

            r2 = subprocess.run(
                ["latexmlpost", "--dest", html_path, "--format=html5",
                 "--nocrossref", "--nodefaultresources", xml_path],
                capture_output=True, timeout=60,
            )
            if r2.returncode != 0 or not Path(html_path).exists():
                logger.warning("latexmlpost failed: %s", r2.stderr[:500])
                return _fallback(tex, title)

            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()

            from dq.extraction.html import html_to_text
            return html_to_text(html)

    except FileNotFoundError:
        logger.warning("latexml not installed, using fallback")
        return _fallback(tex, title)
    except subprocess.TimeoutExpired:
        logger.warning("latexml timed out")
        return _fallback(tex, title)
    except Exception as e:
        logger.warning("latexml error: %s", e)
        return _fallback(tex, title)


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


import re  # noqa: E402 — needed by _fallback
