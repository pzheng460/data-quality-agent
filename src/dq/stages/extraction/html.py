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


def html_to_text(html: str, raw_tex: str | None = None) -> str:
    """Extract text from LaTeXML HTML. Pure format conversion.

    Handles LaTeXML-specific elements:
    - Author affiliations (strip superscript numbers)
    - Footnote marks (remove)
    - Algorithm blocks (extract pseudocode text)
    - Table headers (deduplicate LaTeXML's doubled rendering)
    - Math (preserve as $LaTeX$)
    - Citations (remove <cite> tags)

    If *raw_tex* is provided, algorithm blocks are parsed directly from
    the LaTeX source (full nesting + indentation) instead of from
    LaTeXML's broken HTML output.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # ── Pre-parse from raw LaTeX (if available) ──
    _parsed_algos: list[tuple[str, str, str]] = []
    if raw_tex:
        from dq.stages.extraction.algorithm import extract_algorithms_from_tex
        _parsed_algos = extract_algorithms_from_tex(raw_tex)

    # ── Remove non-content elements ──
    for tag in soup.find_all(["style", "script", "nav", "header", "footer"]):
        tag.decompose()

    # ── Handle algorithm/pseudocode blocks ──
    # If we have parsed algorithms from raw LaTeX, use those (proper indentation).
    # Otherwise fall back to heuristic extraction from LaTeXML's flattened HTML.
    algo_elements = soup.select(".ltx_listing, .ltx_algorithm, [class*='algorithm']")
    for idx, algo in enumerate(algo_elements):
        if idx < len(_parsed_algos):
            caption, _label, pseudocode = _parsed_algos[idx]
            algo_text = pseudocode
        else:
            algo_text = _extract_algorithm(algo)
        if algo_text:
            new_pre = soup.new_tag("pre")
            new_pre.string = "```\n" + algo_text + "\n```"
            algo.replace_with(new_pre)
        else:
            algo.decompose()

    # ── Remove LaTeXML error/undefined elements ──
    # \crefname, \Crefname etc. produce <span class="ltx_ERROR undefined"> + bare text
    # like "algorithmAlgorithmAlgorithm lemmaLemmaLemma..."
    # Known crefname garbage keywords — if 3+ of these appear, the paragraph is junk
    _CREF_KEYWORDS = {"algorithm", "lemma", "table", "theorem", "corollary",
                      "equation", "figure", "section", "appendix", "definition",
                      "proposition", "remark", "example", "assumption", "hyperref",
                      "alg", "tab", "fig", "eq", "sec", "app", "thm", "lem",
                      "cor", "def", "prop", "rem"}
    for err in soup.select(".ltx_ERROR"):
        parent = err.parent
        err.decompose()
        if parent and parent.name == 'p':
            remaining = parent.get_text(strip=True)
            if not remaining or len(remaining) > 600:
                continue
            # Check if it's crefname garbage by looking for known keywords
            lower = remaining.lower()
            hits = sum(1 for kw in _CREF_KEYWORDS if kw in lower)
            if hits >= 3:
                parent.decompose()
                continue
            # Fallback: pure letters/spaces/dots with many uppercase
            cleaned = remaining.replace(" ", "").replace(".", "")
            if cleaned.isalpha():
                uppers = sum(1 for c in remaining if c.isupper())
                if uppers >= 3:
                    parent.decompose()

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

    # ── Remove images from figures (keep captions) ──
    for fig in soup.find_all("figure"):
        for img in fig.find_all(["img", "embed", "object", "picture", "svg"]):
            img.decompose()

    # ── Convert data tables ──
    # Handle both <table> elements and <span class="ltx_tabular"> (LaTeXML
    # sometimes renders tables as spans instead of proper HTML tables).
    def _table_to_markdown(rows):
        """Convert extracted rows to GFM markdown table string."""
        if not rows or len(rows) < 2:
            return None
        total_cells = sum(len(r) for r in rows)
        empty_cells = sum(1 for r in rows for c in r if not c.strip())
        if total_cells > 0 and empty_cells / total_cells >= 0.4:
            return None  # figure-layout table
        md_rows = []
        for cells in rows:
            md_rows.append("| " + " | ".join(_dedup_camelcase(c) for c in cells) + " |")
        ncols = len(rows[0])
        md_rows.insert(1, "| " + " | ".join(["---"] * ncols) + " |")
        return "\n".join(r for r in md_rows if r.strip())

    for table in list(soup.find_all("table")):
        cls = table.get("class", []) if table.parent else []
        if any("equation" in c for c in cls):
            continue
        rows = _extract_table(table)
        md = _table_to_markdown(rows) if rows else None
        if md:
            new_pre = soup.new_tag("pre")
            new_pre.string = md
            table.replace_with(new_pre)
        else:
            table.decompose()

    # ── Handle LaTeXML span-based tables (span.ltx_tabular) ──
    for span_tab in list(soup.select("span.ltx_tabular, div.ltx_tabular")):
        rows = _extract_span_table(span_tab)
        md = _table_to_markdown(rows) if rows else None
        if md:
            new_pre = soup.new_tag("pre")
            new_pre.string = md
            span_tab.replace_with(new_pre)
        else:
            span_tab.decompose()

    # ── Extract block elements ──
    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                              "p", "pre", "li", "figcaption", "blockquote", "div"]):
        if el.find_parent(["p", "li", "figcaption", "blockquote"]):
            continue
        if el.name == "div" and el.find(["h1", "h2", "h3", "h4", "h5", "h6",
                                          "p", "pre", "li", "figcaption", "blockquote"]):
            continue

        # For <pre> (tables, code), preserve newlines but remove blank lines
        if el.name == "pre":
            raw = el.get_text()
            text = "\n".join(l for l in raw.split("\n") if l.strip())
        else:
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
        # Strip trailing LaTeX hard space "\ " or lone "\" that breaks $...$
        alt = re.sub(r"\\[ \t]*$", "", alt)
        # Fix LaTeXML rendering \% as \$ in some contexts
        alt = alt.replace("\\$", "\\%")
        # Replace \qed with KaTeX-compatible symbol, strip \hfill
        alt = alt.replace("\\qed", "\\square")
        alt = alt.replace("\\hfill", " ")
        # Fix LaTeXML delimiter syntax for KaTeX compatibility:
        # \big{(} → \big(   \Big{\{} → \Big\{   \big{\lceil} → \big\lceil
        alt = re.sub(
            r"\\(big|Big|bigg|Bigg|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr|left|right)\{(\\?[^}]+)\}",
            r"\\\1\2",
            alt,
        )
        # Strip \\[Xpt] line spacing hints — breaks remark-math parser
        alt = re.sub(r"\\\\\[[\d.]+(?:pt|em|ex|mm|cm)?\]", r"\\\\", alt)
        # Remove LaTeXML internal commands that leak into alttext
        # e.g. \hidden@noalign{}, \@row@before, \hfil@stuff
        alt = re.sub(r"\\[a-zA-Z]*@[a-zA-Z@]*(?:\{[^}]*\})*", "", alt)
        return alt
    tex = el.get("tex", "")
    if tex:
        return tex
    return el.get_text()


def _extract_span_table(span_tab) -> list[list[str]]:
    """Extract a LaTeXML span-based table (span.ltx_tabular) into rows.

    LaTeXML sometimes renders tables as nested <span> elements with CSS
    classes ltx_tabular/ltx_tr/ltx_td instead of proper <table>/<tr>/<td>.
    """
    trs = span_tab.select(".ltx_tr")
    if not trs:
        return []

    rows: list[list[str]] = []
    for tr in trs:
        cells = tr.select(".ltx_td, .ltx_th")
        row = []
        for td in cells:
            cell_text = td.get_text(separator=" ", strip=True)
            cell_text = cell_text.replace("\xa0", " ")
            cell_text = re.sub(r"\s+", " ", cell_text).strip()
            cell_text = re.sub(r"\s*\(\s*\)\s*$", "", cell_text)
            row.append(cell_text)
        if row:
            rows.append(row)

    if not rows:
        return []

    # Pad rows to same width
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    # Filter out empty rows
    return [r for r in rows if any(c.strip() for c in r)]


def _extract_table(table) -> list[list[str]]:
    """Extract table with rowspan/colspan into a flat Markdown-ready grid.

    Strategy for merged cells (following RedPajama convention):
    - colspan: content goes in first column, rest left empty
    - rowspan in header: merge vertically (e.g. "LLM" spanning 2 header rows
      stays as "LLM" in the merged single header row)
    - rowspan in data: fill content into every spanned row so each row
      is self-contained (e.g. "Qwen3" rowspan=6 → all 6 rows say "Qwen3")
    - Multi-row headers are merged into a single header row by concatenating
      vertically (e.g. "LongBenchV2" + "Latency(ms)" → "LongBenchV2 Latency(ms)")
    """
    trs = table.find_all("tr")
    if not trs:
        return []

    # First pass: determine grid size
    max_cols = 0
    for tr in trs:
        col_count = 0
        for td in tr.find_all(["td", "th"]):
            col_count += int(td.get("colspan", 1))
        max_cols = max(max_cols, col_count)

    if max_cols == 0:
        return []

    # Build raw grid + track which cells are headers
    num_rows = len(trs)
    grid: list[list[str]] = [[""] * max_cols for _ in range(num_rows)]
    is_header: list[list[bool]] = [[False] * max_cols for _ in range(num_rows)]
    occupied: list[list[bool]] = [[False] * max_cols for _ in range(num_rows)]

    for row_idx, tr in enumerate(trs):
        col_idx = 0
        for td in tr.find_all(["td", "th"]):
            while col_idx < max_cols and occupied[row_idx][col_idx]:
                col_idx += 1
            if col_idx >= max_cols:
                break

            cell_text = td.get_text(separator=" ", strip=True)
            # Clean makecell residuals: "[l]..." prefix, newlines, non-breaking spaces
            cell_text = re.sub(r"^\[l\]", "", cell_text)
            cell_text = cell_text.replace("\xa0", " ")
            cell_text = re.sub(r"\s+", " ", cell_text).strip()
            # Clean empty citation markers: "Name ( )" → "Name"
            cell_text = re.sub(r"\s*\(\s*\)\s*$", "", cell_text)
            cell_text = re.sub(r"\s*\(\s*\)\s*", " ", cell_text)
            rowspan = int(td.get("rowspan", 1))
            colspan = int(td.get("colspan", 1))
            cell_is_header = td.name == "th"

            for dr in range(rowspan):
                for dc in range(colspan):
                    r, c = row_idx + dr, col_idx + dc
                    if r < len(grid) and c < max_cols:
                        # Only the top-left cell of any span gets text.
                        # For header rowspan, fill all rows (for merging).
                        if dc == 0 and (cell_is_header or dr == 0):
                            grid[r][c] = cell_text
                        is_header[r][c] = cell_is_header
                        occupied[r][c] = True

            col_idx += colspan

    # Detect header rows: rows where all non-empty cells are <th>
    header_row_count = 0
    for r in range(len(grid)):
        row_has_th = any(is_header[r][c] for c in range(max_cols) if grid[r][c].strip())
        row_has_td = any(not is_header[r][c] for c in range(max_cols) if grid[r][c].strip())
        if row_has_th and not row_has_td:
            header_row_count += 1
        else:
            break

    # Merge multi-row headers into a single header row
    if header_row_count > 1:
        merged = [""] * max_cols
        for c in range(max_cols):
            seen = []
            for r in range(header_row_count):
                val = grid[r][c].strip()
                if val and val not in seen:
                    seen.append(val)
            merged[c] = " ".join(seen)
        result = [merged] + [grid[r] for r in range(header_row_count, len(grid))]
    else:
        result = grid

    # Filter out empty rows
    return [row for row in result if any(cell.strip() for cell in row)]


# algorithm2e keyword classification
_ALGO_KW_META: dict[str, str] = {
    # label: "io" = input/output header, "block" = opens indented block,
    #         "deindent" = closes/reopens block, "stmt" = standalone statement,
    #         "skip" = config macro (discard)
    "\\KwIn": "io", "\\KwOut": "io", "\\KwData": "io", "\\KwResult": "io",
    "\\ForEach": "block", "\\For": "block", "\\While": "block",
    "\\If": "block", "\\ElseIf": "deindent", "\\eIf": "block",
    "\\Else": "deindent",
    "\\Return": "stmt", "\\BlankLine": "blank",
    "\\Repeat": "block", "\\Until": "deindent",
    "\\Switch": "block", "\\Case": "deindent",
    "\\SetKwInOut": "skip", "\\SetKwInput": "skip",
    "\\SetKwFunction": "skip", "\\SetKwData": "skip",
    "\\SetKwComment": "skip", "\\SetAlgoLined": "skip",
    "\\DontPrintSemicolon": "skip", "\\SetAlgoNoLine": "skip",
    "\\SetAlgoNoEnd": "skip",
}

_ALGO_KW_TEXT: dict[str, str] = {
    "\\KwIn": "Input:", "\\KwOut": "Output:", "\\KwData": "Data:",
    "\\KwResult": "Result:",
    "\\ForEach": "for each", "\\For": "for", "\\While": "while",
    "\\If": "if", "\\ElseIf": "else if", "\\eIf": "if", "\\Else": "else",
    "\\Return": "return", "\\BlankLine": "",
    "\\Repeat": "repeat", "\\Until": "until",
    "\\Switch": "switch", "\\Case": "case",
}


def _extract_algorithm(algo) -> str:
    """Extract pseudocode from a LaTeXML algorithm block.

    LaTeXML renders algorithm2e keywords (KwIn, ForEach, etc.) as
    ltx_ERROR spans and flattens all structure into one line.
    We walk the direct children, recognise keyword spans, and
    reconstruct indented pseudocode.
    """
    from bs4 import NavigableString

    # Find the container (ltx_listingline or the algo div itself)
    listing_lines = algo.select(".ltx_listingline")
    containers = listing_lines if listing_lines else [algo]

    # Collect a sequence of tokens: ("kw", cmd) or ("text", string) or ("math", latex)
    tokens: list[tuple[str, str]] = []
    for container in containers:
        for child in container.children:
            if isinstance(child, NavigableString):
                s = str(child).strip()
                if s:
                    tokens.append(("text", s))
            elif hasattr(child, "name"):
                cls = child.get("class", [])
                if "ltx_ERROR" in cls:
                    cmd = child.get_text(strip=True)
                    tokens.append(("kw", cmd))
                elif child.name == "math":
                    alt = _math_to_latex(child)
                    if not alt:
                        alt = child.get_text()
                    tokens.append(("math", "$" + alt + "$"))
                else:
                    # Other inline elements — extract text
                    s = child.get_text(strip=True)
                    if s:
                        tokens.append(("text", s))

    # Now reconstruct lines with indentation.
    # Since LaTeXML flattens all nesting, we use a simple rule:
    #   - io keywords (Input/Output) at indent 0
    #   - block keywords (for/while/if) start a new line at current indent,
    #     then subsequent statements indent +1 until the next block/io keyword
    lines: list[str] = []
    indent = 0
    in_block = False  # True after a block keyword has been seen
    current_parts: list[str] = []

    def flush():
        nonlocal current_parts
        line = " ".join(current_parts).strip()
        line = re.sub(r"[ \t]{2,}", " ", line)
        if line:
            lines.append("  " * indent + line)
        current_parts = []

    for typ, val in tokens:
        if typ == "kw":
            meta = None
            kw_text = None
            for kw in _ALGO_KW_META:
                if val == kw or val.startswith(kw):
                    meta = _ALGO_KW_META[kw]
                    kw_text = _ALGO_KW_TEXT.get(kw, "")
                    break

            if meta == "skip":
                continue
            elif meta == "blank":
                flush()
                lines.append("")
                continue
            elif meta == "io":
                flush()
                indent = 0
                in_block = False
                if kw_text:
                    current_parts.append(kw_text)
            elif meta in ("block", "deindent"):
                flush()
                # block keywords at indent 1 (body at indent 2)
                if meta == "deindent":
                    indent = max(1, indent - 1)
                else:
                    indent = 1 if not in_block else indent
                if kw_text:
                    current_parts.append(kw_text)
                in_block = True
            elif meta == "stmt":
                flush()
                if kw_text:
                    current_parts.append(kw_text)
            else:
                current_parts.append(val)
        elif typ == "math":
            current_parts.append(val)
        else:  # text
            # Split on semicolons to get statement boundaries
            parts = val.split(";")
            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    current_parts.append(part)
                if i < len(parts) - 1:
                    # Semicolon = end of statement
                    if current_parts:
                        current_parts[-1] = current_parts[-1].rstrip() + ";"
                    flush()
                    # After flushing a statement inside a block, indent for body
                    if in_block and indent < 2:
                        indent = 2

    flush()

    # Post-process: trim trailing empty lines, collapse excessive blanks
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dedup_camelcase(text: str) -> str:
    """Fix LaTeXML's doubled table headers like 'Mainmain' -> 'Main'.

    LaTeXML sometimes renders \\textbf{Main} as 'Mainmain' where the
    styled and unstyled versions are concatenated.
    """
    # Pattern: CapitalizedWord immediately followed by same word lowercase
    # e.g. "Mainmain" "Verifierverifier" "Speculatorspeculator"
    return re.sub(r"([A-Z][a-z]+)(\1)", lambda m: m.group(1), text, flags=re.IGNORECASE)
