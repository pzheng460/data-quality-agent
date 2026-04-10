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

    Handles LaTeXML-specific elements:
    - Author affiliations (strip superscript numbers)
    - Footnote marks (remove)
    - Algorithm blocks (extract pseudocode text)
    - Table headers (deduplicate LaTeXML's doubled rendering)
    - Math (preserve as $LaTeX$)
    - Citations (remove <cite> tags)
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # ── Remove non-content elements ──
    for tag in soup.find_all(["style", "script", "nav", "header", "footer"]):
        tag.decompose()

    # ── Fix 5: Remove footnote marks (superscript numbers like ¹ ² *) ──
    for fn in soup.select(".ltx_note_mark, .ltx_tag_note"):
        fn.decompose()

    # ── Fix 1: Clean author affiliation superscripts ──
    for creator in soup.select(".ltx_creator.ltx_role_author"):
        # Remove contact-related elements
        for contact in creator.select(".ltx_contact"):
            contact.decompose()
        # Remove affiliation number superscripts (class ltx_note in author context)
        for note in creator.select(".ltx_note"):
            note.decompose()
        for sup in creator.find_all("sup"):
            sup.decompose()

    # ── Remove footnote content blocks (they appear at bottom, not useful) ──
    for note in soup.select(".ltx_note_content, .ltx_note_outer"):
        note.decompose()

    # ── Convert equation tables to display math BEFORE generic math replacement ──
    # LaTeXML wraps display equations in <table class="ltx_equation">.
    # Process innermost first (reversed) to handle nesting correctly.
    for eq_table in reversed(soup.select("table.ltx_equation, table.ltx_equationgroup")):
        math_parts = []
        for m in eq_table.find_all("math"):
            alt = _math_to_latex(m)
            if alt:
                math_parts.append(alt)
        if math_parts:
            new_p = soup.new_tag("p")
            new_p.string = f"$${ ' '.join(math_parts) }$$"
            eq_table.replace_with(new_p)
        else:
            eq_table.decompose()

    # ── Replace remaining <math> (inline) with LaTeX source ──
    for math_el in soup.find_all("math"):
        latex_src = _math_to_latex(math_el)
        if not latex_src:
            latex_src = math_el.get_text()
        math_el.replace_with(f"${latex_src}$")

    # ── Remove citations (LaTeXML <cite> tags) ──
    for cite_el in soup.find_all("cite"):
        cite_el.decompose()

    # ── Fix 3: Handle algorithm/pseudocode blocks ──
    for algo in soup.select(".ltx_listing, .ltx_algorithm, [class*='algorithm']"):
        # Extract just the text content, stripping LaTeXML algorithm macros
        algo_text = algo.get_text(separator="\n", strip=True)
        # Remove \SetKw... \DontPrintSemicolon etc.
        algo_text = re.sub(r"\\(?:SetKw\w+|DontPrintSemicolon|SetAlgoLined)\w*", "", algo_text)
        # Clean up
        algo_text = re.sub(r"\n{2,}", "\n", algo_text).strip()
        if algo_text:
            algo.replace_with(f"\n```\n{algo_text}\n```\n")
        else:
            algo.decompose()

    # ── Remove images from figures (keep captions) ──
    for fig in soup.find_all("figure"):
        for img in fig.find_all(["img", "embed", "object", "picture", "svg"]):
            img.decompose()

    # ── Convert data tables — handle LaTeXML's doubled headers ──
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                cell_text = td.get_text(strip=True)
                cell_text = _dedup_camelcase(cell_text)
                cells.append(cell_text)
            if any(cells):
                row_text = " | ".join(cells).strip()
                if row_text:
                    rows.append(row_text)
        table.replace_with("\n".join(rows) + "\n")

    # ── Extract block elements ──
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
        # Strip LaTeX line-continuation comments (%\n or trailing %)
        alt = re.sub(r"%\s*\n\s*", "", alt)
        alt = alt.rstrip("%").strip()
        # Fix LaTeXML delimiter syntax for KaTeX compatibility:
        # \big{(} → \big(   \Big{\{} → \Big\{   \big{\lceil} → \big\lceil
        alt = re.sub(
            r"\\(big|Big|bigg|Bigg|left|right)\{(\\?[^}]+)\}",
            r"\\\1\2",
            alt,
        )
        return alt
    tex = el.get("tex", "")
    if tex:
        return tex
    return el.get_text()


def _dedup_camelcase(text: str) -> str:
    """Fix LaTeXML's doubled table headers like 'Mainmain' -> 'Main'.

    LaTeXML sometimes renders \\textbf{Main} as 'Mainmain' where the
    styled and unstyled versions are concatenated.
    """
    # Pattern: CapitalizedWord immediately followed by same word lowercase
    # e.g. "Mainmain" "Verifierverifier" "Speculatorspeculator"
    return re.sub(r"([A-Z][a-z]+)(\1)", lambda m: m.group(1), text, flags=re.IGNORECASE)
