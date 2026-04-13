r"""LaTeX table parser — extracts tabular/longtable to Markdown tables.

Parses raw LaTeX directly, handling:
- \makecell{line1\\line2}  → "line1 line2"
- \multicolumn{n}{spec}{text}
- \multirow{n}{width}{text}
- \toprule, \midrule, \bottomrule, \hline, \cline (stripped)
- \textasciitilde → ~
- Basic formatting: \textbf, \textit, \emph, \textrm
"""

from __future__ import annotations

import re


def extract_tables_from_tex(tex: str) -> list[tuple[str, list[list[str]]]]:
    r"""Extract tables from LaTeX source.

    Returns list of (caption, rows) where each row is a list of cell strings.
    Rows[0] is the header row.
    """
    results = []

    # Match \begin{table}...\end{table} wrappers (may contain tabular inside)
    # Also match bare tabular environments
    for env in ("table", "table*"):
        for m in re.finditer(
            rf"\\begin\{{{env}\}}(?:\[[^\]]*\])?(.*?)\\end\{{{env}\}}",
            tex, re.DOTALL,
        ):
            body = m.group(1)
            caption = _extract_caption(body)
            tab = _find_tabular(body)
            if tab:
                rows = _parse_tabular(tab)
                if rows and len(rows) > 1:
                    results_entry = (caption, rows)
                    results.append(results_entry)

    return results


def _extract_caption(body: str) -> str:
    """Extract caption text from table environment."""
    m = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", body)
    if m:
        return _clean_cell(m.group(1))
    return ""


def _find_tabular(body: str) -> str | None:
    """Find the tabular/longtable body, skipping the column spec."""
    for env in ("tabular", "tabular*", "longtable"):
        # Find \begin{env}
        start_pat = rf"\\begin\{{{env}\}}"
        m = re.search(start_pat, body)
        if not m:
            continue
        pos = m.end()
        # Skip optional [pos]
        if pos < len(body) and body[pos] == '[':
            close = body.index(']', pos)
            pos = close + 1
        # Skip {column spec} — handle nested braces
        if pos < len(body) and body[pos] == '{':
            depth = 0
            for i in range(pos, len(body)):
                if body[i] == '{':
                    depth += 1
                elif body[i] == '}':
                    depth -= 1
                    if depth == 0:
                        pos = i + 1
                        break
        # Find \end{env}
        end_pat = rf"\\end\{{{env}\}}"
        end_m = re.search(end_pat, body[pos:])
        if end_m:
            return body[pos:pos + end_m.start()]
    return None


def _parse_tabular(body: str) -> list[list[str]]:
    """Parse tabular body into rows of cells."""
    # Remove rules
    body = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|cline\{[^}]*\})\s*", "", body)
    # Remove \rowcolor, \cellcolor
    body = re.sub(r"\\(?:rowcolor|cellcolor)(?:\[[^\]]*\])?\{[^}]*\}", "", body)
    # Remove \\[Xpt] spacing hints
    body = re.sub(r"\\\\\[[\d.]+(?:pt|em|ex|mm|cm)?\]", r"\\\\", body)

    # Split on \\ (row delimiter)
    raw_rows = re.split(r"\\\\", body)

    rows: list[list[str]] = []
    for raw_row in raw_rows:
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        # Skip rows that are just rules
        if re.match(r"^\\(?:toprule|midrule|bottomrule|hline)\s*$", raw_row):
            continue

        cells = _split_row(raw_row)
        if cells:
            rows.append(cells)

    if not rows:
        return []

    # Handle multirow: fill forward
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")

    # Fill multirow gaps
    for col in range(max_cols):
        for row_idx in range(1, len(rows)):
            if not rows[row_idx][col].strip() and rows[row_idx - 1][col].strip():
                # Check if previous cell was a multirow
                rows[row_idx][col] = rows[row_idx - 1][col]

    return rows


def _split_row(row: str) -> list[str]:
    """Split a row by & respecting brace nesting and \\& escapes."""
    cells = []
    depth = 0
    current: list[str] = []
    i = 0

    while i < len(row):
        ch = row[i]
        if ch == '\\' and i + 1 < len(row) and row[i + 1] == '&':
            # Escaped \& — keep as literal &
            current.append('&')
            i += 2
            continue
        elif ch == '{':
            depth += 1
            current.append(ch)
        elif ch == '}':
            depth -= 1
            current.append(ch)
        elif ch == '&' and depth == 0:
            cells.append(_clean_cell("".join(current)))
            current = []
        else:
            current.append(ch)
        i += 1

    cells.append(_clean_cell("".join(current)))
    return cells


def _clean_cell(text: str) -> str:
    """Clean a single cell's LaTeX content to readable text."""
    text = text.strip()

    # Handle \makecell[align]{line1\\line2} → "line1 line2"
    text = re.sub(
        r"\\makecell(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}",
        lambda m: re.sub(r"\\\\", " ", m.group(1)).strip(),
        text,
    )

    # Handle \multicolumn{n}{spec}{text} → just text
    text = re.sub(
        r"\\multicolumn\{\d+\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}",
        r"\1",
        text,
    )

    # Handle \multirow{n}{width}{text} → just text
    text = re.sub(
        r"\\multirow\{\d+\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}",
        r"\1",
        text,
    )

    # \textasciitilde → ~
    text = text.replace("\\textasciitilde", "~")

    # \textbf{...}, \textit{...}, \emph{...}, \textrm{...} → content
    text = re.sub(r"\\(?:textbf|textit|emph|textrm|textsc|textsf|texttt)\{([^}]*)\}", r"\1", text)

    # \textsuperscript{...} → content
    text = re.sub(r"\\textsuperscript\{([^}]*)\}", r"^\1", text)

    # \text{...} → content
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)

    # $...$ math — keep as-is for downstream rendering
    # But clean simple cases: ${}^{*}$ → *
    text = re.sub(r"\$\{\}\^\{([^}]+)\}\$", r"^\1", text)

    # \textsuperscript{x} → ^x
    text = re.sub(r"\\textsuperscript\{([^}]*)\}", r"^\1", text)

    # \% → %
    text = text.replace("\\%", "%")
    # \& → &
    text = text.replace("\\&", "&")
    # \textasciitilde → ~
    text = re.sub(r"\\textasciitilde\s*", "~", text)

    # \makecell leftovers
    text = re.sub(r"\\makecell(?:\[[^\]]*\])?\{", "", text)

    # Strip formatting/sizing commands
    text = re.sub(r"\\(?:small|footnotesize|scriptsize|tiny|normalsize|large|Large|LARGE|huge|Huge)\b\s*", "", text)
    text = re.sub(r"\\(?:centering|arraybackslash|raggedright|raggedleft)\b\s*", "", text)
    text = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|cline\{[^}]*\})\s*", "", text)
    # \vspace, \hspace, \newline
    text = re.sub(r"\\(?:vspace|hspace)\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\newline\b", " ", text)
    # \cite{...}, \citep{...}, \citet{...} → remove
    text = re.sub(r"~?\\(?:cite[pt]?)\{[^}]*\}", "", text)
    # \textcolor{color}{text} → text
    text = re.sub(r"\\textcolor\{[^}]*\}\{([^}]*)\}", r"\1", text)
    # \shortstack{...} → content (replace \\ with space)
    text = re.sub(r"\\shortstack(?:\[[^\]]*\])?\{([^}]*)\}", lambda m: m.group(1).replace("\\\\", " "), text)
    # \includegraphics → [image]
    text = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", "[image]", text)

    # \multicolumn without braces pattern (leftover)
    text = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", text)

    # Remove stray backslash commands that are just formatting
    text = re.sub(r"\\(?:cellsep|rowsep|cmidrule)\b(?:\{[^}]*\})*", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tables_to_markdown(tables: list[tuple[str, list[list[str]]]]) -> list[str]:
    """Convert parsed tables to Markdown strings."""
    results = []
    for caption, rows in tables:
        if not rows:
            continue
        md_lines = []
        # Header
        md_lines.append(" | ".join(rows[0]))
        md_lines.append(" | ".join(["---"] * len(rows[0])))
        # Data
        for row in rows[1:]:
            # Pad or trim to header width
            padded = row[:len(rows[0])]
            while len(padded) < len(rows[0]):
                padded.append("")
            md_lines.append(" | ".join(padded))
        results.append("\n".join(md_lines))
    return results
