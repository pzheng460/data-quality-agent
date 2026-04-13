r"""algorithm2e LaTeX parser — extracts pseudocode with proper indentation.

Parses raw LaTeX algorithm blocks directly, bypassing LaTeXML's broken
algorithm2e support.  Handles nested ``\ForEach{cond}{body}``,
``\If{cond}{body}``, ``\eIf{cond}{then}{else}``, etc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── algorithm2e keyword taxonomy ──────────────────────────────────

# "io"    = input/output line (no body)
# "block1"= keyword{condition}{body}          e.g. \ForEach, \While, \If
# "block2"= keyword{condition}{then}{else}    e.g. \eIf
# "deindent" = \Else (no condition, has body)
# "stmt"  = single-line statement keyword
# "io1"   = io keyword with one brace arg     e.g. \KwIn{...}
# "skip"  = config macro, discard

_KW: dict[str, tuple[str, str]] = {
    # (type, display_text)
    # --- I/O ---
    "KwIn":     ("io1",   "Input:"),
    "KwOut":    ("io1",   "Output:"),
    "KwData":   ("io1",   "Data:"),
    "KwResult": ("io1",   "Result:"),
    # --- block with condition + body ---
    "ForEach":  ("block1", "for each"),
    "ForAll":   ("block1", "for all"),
    "For":      ("block1", "for"),
    "While":    ("block1", "while"),
    "If":       ("block1", "if"),
    "ElseIf":   ("block1", "else if"),
    "Case":     ("block1", "case"),
    "Switch":   ("block1", "switch"),
    "Repeat":   ("block1", "repeat"),
    # --- block2: condition + then + else ---
    "eIf":      ("block2", "if"),
    # --- deindent: no condition, has body ---
    "Else":     ("deindent", "else"),
    "Until":    ("deindent", "until"),
    # --- standalone statement ---
    "Return":   ("stmt",  "return"),
    "BlankLine": ("blank", ""),
    # --- TCP comments ---
    "tcp":      ("comment", "//"),
    "tcc":      ("comment", "//"),
    # --- config macros (skip) ---
    "SetKwInOut":      ("skip", ""),
    "SetKwInput":      ("skip", ""),
    "SetKwFunction":   ("skip", ""),
    "SetKwData":       ("skip", ""),
    "SetKwComment":    ("skip", ""),
    "SetAlgoLined":    ("skip", ""),
    "DontPrintSemicolon": ("skip", ""),
    "SetAlgoNoLine":   ("skip", ""),
    "SetAlgoNoEnd":    ("skip", ""),
    "SetKwProg":       ("skip", ""),
    "SetKw":           ("skip", ""),
    "SetKwBlock":      ("skip", ""),
    "SetKwFor":        ("skip", ""),
    "SetKwRepeat":     ("skip", ""),
    "SetKwIF":         ("skip", ""),
    "SetKwSwitch":     ("skip", ""),
    "SetNoFillComment": ("skip", ""),
    "LinesNumbered":   ("skip", ""),
    "LinesNotNumbered": ("skip", ""),
    "NoCaptionOfAlgo":  ("skip", ""),
}

INDENT = "  "


def extract_algorithms_from_tex(tex: str) -> list[tuple[str, str, str]]:
    r"""Extract all algorithm environments from raw LaTeX source.

    Returns list of (caption, label, pseudocode_text) tuples.
    """
    results = []
    for m in re.finditer(
        r"\\begin\{algorithm\}(?:\[[^\]]*\])?\s*\n?(.*?)\\end\{algorithm\}",
        tex, re.DOTALL,
    ):
        body = m.group(1)

        # Skip commented-out algorithms
        lines = body.strip().split("\n")
        non_comment = [l for l in lines if not l.strip().startswith("%")]
        if not non_comment:
            continue

        # Extract caption and label
        caption = ""
        cap_m = re.search(r"\\caption\{([^}]*)\}", body)
        if cap_m:
            caption = cap_m.group(1).strip()

        label = ""
        lab_m = re.search(r"\\label\{([^}]*)\}", body)
        if lab_m:
            label = lab_m.group(1).strip()

        # Remove caption, label, and comment lines
        clean = re.sub(r"\\caption\{[^}]*\}", "", body)
        clean = re.sub(r"\\label\{[^}]*\}", "", clean)
        clean = re.sub(r"(?m)^\s*%.*$", "", clean)
        clean = clean.strip()

        if not clean:
            continue

        pseudocode = _parse_algorithm_body(clean)
        if pseudocode:
            results.append((caption, label, pseudocode))

    return results


def _parse_algorithm_body(body: str) -> str:
    """Parse algorithm2e body into indented pseudocode."""
    tokens = _tokenize(body)
    lines = _tokens_to_lines(tokens, indent_level=0)
    return "\n".join(lines)


# ── Tokenizer: LaTeX → token stream ──────────────────────────────

@dataclass
class Token:
    kind: str  # "cmd", "text", "math", "brace_open", "brace_close", "comment"
    value: str = ""
    cmd_name: str = ""  # for kind="cmd"


def _tokenize(tex: str) -> list[Token]:
    """Tokenize algorithm body into a stream of tokens."""
    tokens: list[Token] = []
    i = 0
    n = len(tex)

    while i < n:
        c = tex[i]

        # Skip whitespace (but track newlines for plain text separation)
        if c in (" ", "\t", "\n", "\r"):
            i += 1
            continue

        # Comment: % to end of line
        if c == "%" and (i == 0 or tex[i - 1] != "\\"):
            end = tex.find("\n", i)
            if end == -1:
                end = n
            # Check if it's a //{...} style comment content
            tokens.append(Token("comment", tex[i + 1:end].strip()))
            i = end + 1
            continue

        # Math: $...$
        if c == "$":
            end = tex.find("$", i + 1)
            if end == -1:
                end = n
            tokens.append(Token("math", tex[i + 1:end]))
            i = end + 1
            continue

        # Braces
        if c == "{":
            tokens.append(Token("brace_open"))
            i += 1
            continue
        if c == "}":
            tokens.append(Token("brace_close"))
            i += 1
            continue

        # Command: \name
        if c == "\\":
            # Check for \\ (line break) — skip
            if i + 1 < n and tex[i + 1] == "\\":
                i += 2
                continue
            # Read command name
            j = i + 1
            while j < n and tex[j].isalpha():
                j += 1
            if j == i + 1:
                # Single-char command like \, \; \space
                if j < n:
                    tokens.append(Token("text", tex[i:j + 1]))
                    i = j + 1
                else:
                    i = j
                continue
            cmd_name = tex[i + 1:j]
            tokens.append(Token("cmd", "\\" + cmd_name, cmd_name))
            i = j
            continue

        # Plain text: read until next special char
        j = i
        while j < n and tex[j] not in ("\\", "$", "{", "}", "%", "\n", "\r"):
            j += 1
        text = tex[i:j].strip()
        if text:
            tokens.append(Token("text", text))
        i = j if j > i else i + 1

    return tokens


# ── Token stream → indented lines ────────────────────────────────

def _tokens_to_lines(tokens: list[Token], indent_level: int) -> list[str]:
    """Convert token stream to indented pseudocode lines."""
    lines: list[str] = []
    i = 0
    current_line_parts: list[str] = []
    prefix = INDENT * indent_level

    def flush():
        nonlocal current_line_parts
        text = " ".join(current_line_parts).strip()
        # Clean up multiple spaces
        text = re.sub(r"  +", " ", text)
        if text:
            lines.append(prefix + text)
        current_line_parts = []

    while i < len(tokens):
        tok = tokens[i]

        if tok.kind == "cmd":
            kw_info = _KW.get(tok.cmd_name)
            if kw_info is None:
                # Unknown command — include as-is (e.g. \textbf)
                # Try to consume {arg}
                if i + 1 < len(tokens) and tokens[i + 1].kind == "brace_open":
                    arg, end = _read_brace_group(tokens, i + 1)
                    arg_text = _tokens_to_text(arg)
                    current_line_parts.append(arg_text)
                    i = end + 1
                else:
                    i += 1
                continue

            kw_type, kw_display = kw_info

            if kw_type == "skip":
                # Skip config macros and their arguments
                i += 1
                while i < len(tokens) and tokens[i].kind == "brace_open":
                    _, end = _read_brace_group(tokens, i)
                    i = end + 1
                continue

            if kw_type == "blank":
                flush()
                i += 1
                continue

            if kw_type == "io1":
                # \KwIn{text}
                flush()
                i += 1
                if i < len(tokens) and tokens[i].kind == "brace_open":
                    arg, end = _read_brace_group(tokens, i)
                    arg_text = _tokens_to_text(arg)
                    lines.append(prefix + f"{kw_display} {arg_text}")
                    i = end + 1
                else:
                    current_line_parts.append(kw_display)
                continue

            if kw_type == "block1":
                # \ForEach{condition}{body}
                flush()
                i += 1
                cond_text = ""
                if i < len(tokens) and tokens[i].kind == "brace_open":
                    cond_tokens, end = _read_brace_group(tokens, i)
                    cond_text = _tokens_to_text(cond_tokens)
                    i = end + 1

                lines.append(prefix + f"{kw_display} {cond_text}:")

                if i < len(tokens) and tokens[i].kind == "brace_open":
                    body_tokens, end = _read_brace_group(tokens, i)
                    body_lines = _tokens_to_lines(body_tokens, indent_level + 1)
                    lines.extend(body_lines)
                    i = end + 1
                continue

            if kw_type == "block2":
                # \eIf{condition}{then}{else}
                flush()
                i += 1
                cond_text = ""
                if i < len(tokens) and tokens[i].kind == "brace_open":
                    cond_tokens, end = _read_brace_group(tokens, i)
                    cond_text = _tokens_to_text(cond_tokens)
                    i = end + 1

                lines.append(prefix + f"{kw_display} {cond_text}:")

                # then block
                if i < len(tokens) and tokens[i].kind == "brace_open":
                    then_tokens, end = _read_brace_group(tokens, i)
                    then_lines = _tokens_to_lines(then_tokens, indent_level + 1)
                    lines.extend(then_lines)
                    i = end + 1

                # else block
                if i < len(tokens) and tokens[i].kind == "brace_open":
                    lines.append(prefix + "else:")
                    else_tokens, end = _read_brace_group(tokens, i)
                    else_lines = _tokens_to_lines(else_tokens, indent_level + 1)
                    lines.extend(else_lines)
                    i = end + 1
                continue

            if kw_type == "deindent":
                # \Else{body}
                flush()
                i += 1
                lines.append(prefix + f"{kw_display}:")
                if i < len(tokens) and tokens[i].kind == "brace_open":
                    body_tokens, end = _read_brace_group(tokens, i)
                    body_lines = _tokens_to_lines(body_tokens, indent_level + 1)
                    lines.extend(body_lines)
                    i = end + 1
                continue

            if kw_type == "stmt":
                flush()
                i += 1
                # Collect rest of the line
                current_line_parts.append(kw_display)
                continue

            if kw_type == "comment":
                # \tcp{...} or \tcc{...}
                i += 1
                if i < len(tokens) and tokens[i].kind == "brace_open":
                    arg, end = _read_brace_group(tokens, i)
                    comment_text = _tokens_to_text(arg)
                    current_line_parts.append(f"// {comment_text}")
                    i = end + 1
                continue

        elif tok.kind == "math":
            current_line_parts.append(f"${tok.value}$")
            i += 1

        elif tok.kind == "text":
            # Split on semicolons for statement separation
            parts = tok.value.split(";")
            for j, part in enumerate(parts):
                part = part.strip()
                if part:
                    current_line_parts.append(part)
                if j < len(parts) - 1 and current_line_parts:
                    current_line_parts[-1] = current_line_parts[-1] + ";"
                    flush()
            i += 1

        elif tok.kind == "comment":
            if current_line_parts:
                current_line_parts.append(f"// {tok.value}")
            else:
                current_line_parts.append(f"// {tok.value}")
            flush()
            i += 1

        else:
            i += 1

    flush()
    return lines


def _read_brace_group(tokens: list[Token], start: int) -> tuple[list[Token], int]:
    """Read tokens from { to matching }, handling nesting.

    Returns (inner_tokens, index_of_closing_brace).
    """
    assert tokens[start].kind == "brace_open"
    depth = 1
    i = start + 1
    inner: list[Token] = []
    while i < len(tokens) and depth > 0:
        if tokens[i].kind == "brace_open":
            depth += 1
            if depth > 1:
                inner.append(tokens[i])
        elif tokens[i].kind == "brace_close":
            depth -= 1
            if depth > 0:
                inner.append(tokens[i])
        else:
            inner.append(tokens[i])
        i += 1
    return inner, i - 1


def _tokens_to_text(tokens: list[Token]) -> str:
    """Convert a flat list of tokens to inline text (no structure parsing)."""
    parts: list[str] = []
    for tok in tokens:
        if tok.kind == "text":
            parts.append(tok.value)
        elif tok.kind == "math":
            parts.append(f"${tok.value}$")
        elif tok.kind == "cmd":
            # Unknown command in text context — try to extract arg
            pass
        elif tok.kind == "comment":
            parts.append(f"// {tok.value}")
    return " ".join(parts).strip()
