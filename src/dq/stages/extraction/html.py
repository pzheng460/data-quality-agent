"""HTML extractor — converts LaTeXML/ar5iv HTML to text via BeautifulSoup."""

from __future__ import annotations

import re

from dq.stages.extraction.base import Extractor
from dq.stages.extraction.registry import register_extractor


@register_extractor("html")
class HtmlExtractor(Extractor):
    """Convert HTML (from ar5iv or LaTeXML) to clean text."""

    input_format = "html"

    def extract(self, doc: dict) -> dict | None:
        raw_html = doc.get("text", "")
        if not raw_html:
            return None
        text = html_to_text(raw_html)
        if len(text) < 200:
            return None
        doc["text"] = text
        return doc


def html_to_text(html: str) -> str:
    """Extract text from LaTeXML HTML. Pure format conversion.

    All data cleaning (citations, footnotes, residual LaTeX) is
    handled by ArxivFilter in the filter stage.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Remove non-content HTML elements
    for tag in soup.find_all(["style", "script", "nav", "header", "footer"]):
        tag.decompose()

    # Replace <math> with LaTeX source (MathML → LaTeX string)
    for math_el in soup.find_all("math"):
        latex_src = _math_to_latex(math_el)
        if latex_src:
            display = math_el.get("display", "inline")
            if display == "block":
                math_el.replace_with(f"\n$${latex_src}$$\n")
            else:
                math_el.replace_with(f"${latex_src}$")
        else:
            math_el.replace_with(math_el.get_text())

    # Remove images from figures (keep captions)
    for fig in soup.find_all("figure"):
        for img in fig.find_all(["img", "embed", "object", "picture", "svg"]):
            img.decompose()

    # Convert tables to pipe-delimited text
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                row_text = " | ".join(cells).strip()
                if row_text:
                    rows.append(row_text)
        table.replace_with("\n".join(rows) + "\n")

    # Extract block elements into markdown-like structure
    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                              "p", "li", "figcaption", "blockquote", "div"]):
        if el.find_parent(["p", "li", "figcaption", "blockquote"]):
            continue
        if el.name == "div" and el.find(["h1", "h2", "h3", "h4", "h5", "h6",
                                          "p", "li", "figcaption", "blockquote"]):
            continue

        text = el.get_text(separator=" ", strip=True)
        if not text:
            continue

        tag = el.name
        if tag == "h1":
            lines.append(f"\n# {text}\n")
        elif tag == "h2":
            lines.append(f"\n## {text}\n")
        elif tag == "h3":
            lines.append(f"\n### {text}\n")
        elif tag in ("h4", "h5", "h6"):
            lines.append(f"\n#### {text}\n")
        elif tag == "figcaption":
            lines.append(f"\n[Caption: {text}]\n")
        elif tag == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    result = "\n\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _math_to_latex(el) -> str:
    """Extract LaTeX source from a LaTeXML <math> element."""
    alt = el.get("alttext", "")
    if alt:
        return alt
    tex_attr = el.get("tex", "")
    if tex_attr:
        return tex_attr
    return el.get_text()
