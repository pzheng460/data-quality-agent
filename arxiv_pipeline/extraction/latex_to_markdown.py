#!/usr/bin/env python3
"""Stage 2: Convert raw LaTeX to clean markdown.

Reads raw LaTeX from ingestion output, converts via pandoc,
cleans up pandoc artifacts, writes standardized markdown.

Input:  raw/*.jsonl  (source_format: latex, text is raw LaTeX)
Output: extracted/*.jsonl.zst  (text is clean markdown)

Usage:
    python -m arxiv_pipeline.extraction.latex_to_markdown \
        --input raw/batch1.jsonl \
        --output extracted/batch1.jsonl.zst
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def latex_to_markdown(tex: str) -> str:
    """Convert LaTeX body to markdown using pandoc.

    Strips \\begin{document}/\\end{document} wrapper,
    runs pandoc, then cleans pandoc artifacts.
    """
    # Extract document body
    m = re.search(r"\\begin\{document\}", tex)
    if m:
        tex = tex[m.end():]
    tex = re.sub(r"\\end\{document\}.*", "", tex, flags=re.DOTALL)

    # Primary: pandoc
    try:
        import pypandoc
        md = pypandoc.convert_text(
            tex, "markdown", format="latex",
            extra_args=["--wrap=none", "--strip-comments"],
        )
    except Exception as e:
        logger.warning("pandoc failed, using pylatexenc fallback: %s", e)
        md = _pylatexenc_fallback(tex)

    # Clean pandoc artifacts
    md = _clean_pandoc_output(md)
    return md


def _pylatexenc_fallback(tex: str) -> str:
    """Fallback if pandoc is unavailable."""
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        l2t = LatexNodes2Text(math_mode="verbatim", strict_latex_spaces=False)
        return l2t.latex_to_text(tex)
    except Exception:
        return tex


def _clean_pandoc_output(md: str) -> str:
    """Remove pandoc-specific artifacts from markdown."""
    import re

    # Reference link attributes: {reference-type="ref" reference="fig1"}
    md = re.sub(r"\{[^}]*reference-type[^}]*\}", "", md)
    # Pandoc citation brackets: [@ref1; @ref2]
    md = re.sub(r"\[@[^\]]*\]", "", md)
    # Pandoc div markers: ::: {.theorem}
    md = re.sub(r"^:::\s*.*$", "", md, flags=re.MULTILINE)
    # Image references (no use for pretraining)
    md = re.sub(r"!\[.*?\]\(.*?\)(?:\{[^}]*\})?", "", md)
    # pandoc footnote markers: [^1]
    # (keep the footnote text at bottom, just clean inline markers)

    # LaTeX table junk that pandoc sometimes leaves
    md = re.sub(r"\b(?:toprule|midrule|bottomrule|hline)\b", "", md)
    md = re.sub(r"@\{[^}]*\}", "", md)

    # Clean up whitespace
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"^[ \t]+$", "", md, flags=re.MULTILINE)

    return md.strip()


def process_file(input_path: Path, output_path: Path) -> int:
    """Process a JSONL of raw LaTeX → JSONL of markdown."""
    from dq.utils.io import write_jsonl_zst

    output_path.parent.mkdir(parents=True, exist_ok=True)
    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = __import__("json").loads(line)
            if raw.get("source_format") != "latex":
                logger.warning("Skip %s: source_format=%s", raw.get("id"), raw.get("source_format"))
                continue

            title = raw.get("metadata", {}).get("title", "Untitled")
            md = latex_to_markdown(raw["text"])
            raw["raw_text"] = raw["text"][:500]  # keep preview of raw for comparison
            raw["text"] = f"# {title}\n\n{md}" if not md.startswith("#") else md
            raw["source_format"] = "markdown"
            raw["extraction_method"] = "pandoc"
            raw["text_length"] = len(raw["text"])
            docs.append(raw)
            logger.info("Extracted %s: %d chars", raw.get("id"), raw["text_length"])

    count = write_jsonl_zst(iter(docs), str(output_path))
    logger.info("Extraction done: %d docs → %s", count, output_path)
    return count


def _pylatexenc_fallback(tex: str) -> str:
    """Convert using pylatexenc when pandoc is unavailable."""
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        l2t = LatexNodes2Text(math_mode="verbatim", strict_latex_spaces=False)
        return l2t.latex_to_text(tex)
    except Exception:
        return tex


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: LaTeX → Markdown extraction")
    parser.add_argument("--input", required=True, help="Input JSONL (raw LaTeX)")
    parser.add_argument("--output", required=True, help="Output JSONL.zst (markdown)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    process_file(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
