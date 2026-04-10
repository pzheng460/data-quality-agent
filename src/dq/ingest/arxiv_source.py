"""Fetch papers directly from arxiv e-print (LaTeX source → text).

Usage:
    # By ID list
    src = ArxivSource(ids=["2310.06825", "2307.09288"])
    for doc in src.fetch(limit=10):
        print(doc["id"], len(doc["text"]))

    # By date range (OAI-PMH)
    src = ArxivSource(from_date="2025-01-01", to_date="2025-01-07", categories=["cs.CL"])
    for doc in src.fetch(limit=100):
        ...
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import re
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from dq.ingest.base import IngestSource
from dq.ingest.registry import register_source

logger = logging.getLogger(__name__)

_EPRINT_URL = "https://arxiv.org/e-print"
_API_URL = "http://export.arxiv.org/api/query"
_OAI_URL = "http://export.arxiv.org/oai2"


@register_source("arxiv_latexml")
class ArxivSource(IngestSource):
    """Ingest papers from arxiv via LaTeX source + LaTeXML conversion."""

    name = "arxiv_latexml"
    domain = "arxiv"
    priority = 300

    @classmethod
    def params_schema(cls):
        return {
            "ids": {"type": "list", "label": "Arxiv IDs", "required": False},
            "from_date": {"type": "string", "label": "From date", "required": False},
            "to_date": {"type": "string", "label": "To date", "required": False},
            "categories": {"type": "list", "label": "Categories", "required": False},
            "delay": {"type": "number", "label": "Delay (s)", "default": 3.0},
        }

    def __init__(
        self,
        ids: list[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        categories: list[str] | None = None,
        delay: float = 3.0,
        **_kwargs,
    ) -> None:
        self.ids = ids
        self.from_date = from_date
        self.to_date = to_date
        self.categories = set(categories) if categories else None
        self.delay = delay

    def fetch(self, limit: int = 0) -> Iterator[dict]:
        if self.ids:
            yield from self._fetch_by_ids(self.ids, limit)
        elif self.from_date:
            yield from self._fetch_by_date(limit)
        else:
            raise ValueError("Provide either ids or from_date")

    def _fetch_by_ids(self, ids: list[str], limit: int) -> Iterator[dict]:
        meta = _batch_metadata(ids)
        count = 0
        for aid in ids:
            if limit and count >= limit:
                break
            try:
                tex = _download_latex(aid)
                if not tex or len(tex) < 200:
                    logger.warning("Skip %s: no/short source", aid)
                    continue
                title = meta.get(aid, {}).get("title", "Untitled")
                text = _tex_to_text(tex, title)
                if len(text) < 200:
                    continue
                yield {
                    "id": f"arxiv_{aid}",
                    "text": text,
                    "source": "arxiv",
                    "metadata": meta.get(aid, {"arxiv_id": aid, "title": aid}),
                }
                count += 1
                logger.info("Fetched %s: %s (%d chars)", aid, title[:50], len(text))
            except Exception as e:
                logger.warning("Failed %s: %s", aid, e)
            time.sleep(self.delay)

    def _fetch_by_date(self, limit: int) -> Iterator[dict]:
        """Discover papers via OAI-PMH, then download their source."""
        ids = _oai_list_ids(self.from_date, self.to_date, self.categories, max_results=limit or 1000)
        logger.info("OAI-PMH returned %d IDs (from=%s to=%s)", len(ids), self.from_date, self.to_date)
        yield from self._fetch_by_ids(ids[:limit] if limit else ids, limit)


# ── LaTeX download & conversion ──


def _download_latex(arxiv_id: str) -> str | None:
    """Download LaTeX source from arxiv, merge \\input{} files."""
    url = f"{_EPRINT_URL}/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "dq-pipeline/1.0"})
    try:
        data = urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        return None

    # Try as tar.gz
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tex_files: dict[str, str] = {}
            for m in tar.getmembers():
                if m.name.endswith(".tex"):
                    f = tar.extractfile(m)
                    if f:
                        tex_files[m.name] = f.read().decode("utf-8", errors="replace")

            if not tex_files:
                return None

            # Find main file (with \begin{document})
            main = None
            for name, content in tex_files.items():
                if r"\begin{document}" in content:
                    main = name
                    break
            if not main:
                main = max(tex_files, key=lambda k: len(tex_files[k]))

            # Merge \input{} references (up to 3 levels)
            content = tex_files[main]
            for _ in range(3):
                def _resolve(m: re.Match) -> str:
                    fname = m.group(1).strip()
                    if not fname.endswith('.tex'):
                        fname += '.tex'
                    for path in tex_files:
                        if path == fname or path.endswith('/' + fname):
                            return tex_files[path]
                    return ""
                content = re.sub(r"\\input\{([^}]+)\}", _resolve, content)
            return content
    except tarfile.TarError:
        pass

    # Single gzipped file
    try:
        return gzip.decompress(data).decode("utf-8", errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")



def _tex_to_text(tex: str, title: str) -> str:
    """Convert LaTeX to clean text via LaTeXML → HTML → text extraction.

    LaTeXML faithfully expands macros (\\newcommand, \\def) and resolves
    cross-references, producing high-quality HTML. We then extract readable
    text with BeautifulSoup while preserving math as LaTeX source.

    Falls back to lightweight regex cleaning if LaTeXML is unavailable.
    """
    text = _latexml_convert(tex)
    if text is None:
        text = _fallback_tex_to_text(tex)
        return f"# {title}\n\n{text.strip()}"

    # LaTeXML output already includes the title as <h1>, no need to prepend
    return text.strip()


def _latexml_convert(tex: str) -> str | None:
    """LaTeXML pipeline: .tex → .xml → .html → clean text."""
    try:
        with __import__("tempfile").TemporaryDirectory() as tmpdir:
            tex_path = f"{tmpdir}/paper.tex"
            xml_path = f"{tmpdir}/paper.xml"
            html_path = f"{tmpdir}/paper.html"

            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex)

            # Step 1: LaTeX → XML
            r1 = subprocess.run(
                ["latexml", "--dest", xml_path, tex_path],
                capture_output=True, timeout=120,
            )
            if r1.returncode != 0 or not Path(xml_path).exists():
                logger.warning("latexml failed: %s", r1.stderr[:500])
                return None

            # Step 2: XML → HTML
            r2 = subprocess.run(
                ["latexmlpost", "--dest", html_path, "--format=html5",
                 "--nocrossref", "--nodefaultresources", xml_path],
                capture_output=True, timeout=60,
            )
            if r2.returncode != 0 or not Path(html_path).exists():
                logger.warning("latexmlpost failed: %s", r2.stderr[:500])
                return None

            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()

            return _html_to_text(html)
    except FileNotFoundError:
        logger.warning("latexml not installed, using fallback")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("latexml timed out")
        return None
    except Exception as e:
        logger.warning("latexml error: %s", e)
        return None


def _html_to_math_latex(el) -> str:
    """Extract LaTeX source from a LaTeXML <math> element."""
    # LaTeXML stores original LaTeX in alttext or tex attribute
    alt = el.get("alttext", "")
    if alt:
        return alt
    tex_attr = el.get("tex", "")
    if tex_attr:
        return tex_attr
    # Fallback: get text content
    return el.get_text()


def _html_to_text(html: str) -> str:
    """Extract clean text from LaTeXML HTML output."""
    from bs4 import BeautifulSoup, NavigableString

    soup = BeautifulSoup(html, "lxml")

    # Remove elements that don't contribute to readable text
    for tag in soup.find_all(["style", "script", "nav", "header", "footer"]):
        tag.decompose()

    # Replace <math> elements with their LaTeX source
    for math_el in soup.find_all("math"):
        latex_src = _html_to_math_latex(math_el)
        if latex_src:
            # Check if display or inline math
            display = math_el.get("display", "inline")
            if display == "block":
                math_el.replace_with(f"\n$${latex_src}$$\n")
            else:
                math_el.replace_with(f"${latex_src}$")
        else:
            math_el.replace_with(math_el.get_text())

    # Remove citations (LaTeXML renders as <cite> tags)
    for cite_el in soup.find_all("cite"):
        cite_el.decompose()

    # Remove footnote markers (repeated superscript numbers)
    for note in soup.find_all(class_=re.compile(r"ltx_note_mark|ltx_tag_note")):
        note.decompose()

    # Remove figure images but keep captions
    for fig in soup.find_all("figure"):
        for img in fig.find_all(["img", "embed", "object", "picture", "svg"]):
            img.decompose()

    # Remove table formatting but keep content as simple text
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows_text = " | ".join(cells)
                rows_text = rows_text.strip()
                if rows_text:
                    rows.append(rows_text)
        table.replace_with("\n".join(rows) + "\n")

    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6",
                              "p", "li", "figcaption", "blockquote",
                              "div"]):
        # Skip nested block elements to avoid duplication
        if el.find_parent(["p", "li", "figcaption", "blockquote"]):
            continue
        # Skip divs that contain block children (they're just wrappers)
        if el.name == "div" and el.find(["h1","h2","h3","h4","h5","h6",
                                          "p","li","figcaption","blockquote"]):
            continue

        text = el.get_text(separator=" ", strip=True)
        if not text:
            continue

        tag = el.name
        if tag in ("h1", "h2"):
            level = "#" if tag == "h1" else "##"
            # Strip leading section numbers (e.g. "1 Introduction" → "Introduction")
            text = re.sub(r"^\d+(\.\d+)*\s+", "", text)
            lines.append(f"\n{level} {text}\n")
        elif tag == "h3":
            text = re.sub(r"^\d+(\.\d+)*\s+", "", text)
            lines.append(f"\n### {text}\n")
        elif tag in ("h4", "h5", "h6"):
            text = re.sub(r"^\d+(\.\d+)*\s+", "", text)
            lines.append(f"\n#### {text}\n")
        elif tag == "figcaption":
            lines.append(f"\n[Caption: {text}]\n")
        elif tag == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    result = "\n\n".join(lines)

    # Minimal normalization only — data cleaning is done by ArxivFilter
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def _fallback_tex_to_text(tex: str) -> str:
    """Lightweight regex fallback when LaTeXML is unavailable."""
    # Extract document body
    m = re.search(r"\\begin\{document\}", tex)
    if m:
        tex = tex[m.end():]
    tex = re.sub(r"\\end\{document\}.*", "", tex, flags=re.DOTALL)

    # Strip comments
    tex = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.MULTILINE)

    # Remove layout commands
    tex = re.sub(r"\\(?:maketitle|tableofcontents|newpage|clearpage|pagebreak)\b", "", tex)

    # Collapse blank lines
    tex = re.sub(r"\n{3,}", "\n\n", tex)

    return tex.strip()


# ── Metadata & OAI-PMH ──


def _batch_metadata(ids: list[str]) -> dict[str, dict]:
    """Fetch title/categories/abstract via arxiv API."""
    import html as html_mod
    result: dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        batch = ids[i:i+50]
        url = f"{_API_URL}?id_list={','.join(batch)}&max_results={len(batch)}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml = resp.read().decode()
            for entry in re.finditer(r"<entry>(.*?)</entry>", xml, re.DOTALL):
                block = entry.group(1)
                aid_m = re.search(r"<id>.*?/abs/([^<]+)</id>", block)
                title_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL)
                abs_m = re.search(r"<summary>(.*?)</summary>", block, re.DOTALL)
                cats = re.findall(r'<category[^>]*term="([^"]*)"', block)
                if aid_m:
                    aid = re.sub(r"v\d+$", "", aid_m.group(1).strip())
                    result[aid] = {
                        "arxiv_id": aid,
                        "title": html_mod.unescape(title_m.group(1).strip()) if title_m else "",
                        "abstract": html_mod.unescape(abs_m.group(1).strip()) if abs_m else "",
                        "categories": cats,
                        "primary_category": cats[0] if cats else "",
                        "version": "v1",
                    }
        except Exception as e:
            logger.warning("Metadata fetch failed: %s", e)
        time.sleep(0.5)
    return result


def _oai_list_ids(
    from_date: str,
    to_date: str | None = None,
    categories: set[str] | None = None,
    max_results: int = 1000,
) -> list[str]:
    """List arxiv IDs via OAI-PMH (for discovering new papers by date)."""
    ids: list[str] = []
    params = f"verb=ListIdentifiers&metadataPrefix=arXiv&from={from_date}"
    if to_date:
        params += f"&until={to_date}"
    url = f"{_OAI_URL}?{params}"

    while url and len(ids) < max_results:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml_text = resp.read().decode()
            root = ET.fromstring(xml_text)
            ns = {"oai": "http://www.openarchives.org/OAI/2.0/"}

            for header in root.findall(".//oai:header", ns):
                identifier = header.findtext("oai:identifier", "", ns)
                if identifier:
                    # oai:arXiv.org:2310.06825 → 2310.06825
                    aid = identifier.split(":")[-1]
                    # Category filter
                    if categories:
                        setspecs = [s.text or "" for s in header.findall("oai:setSpec", ns)]
                        if not any(c in " ".join(setspecs) for c in categories):
                            continue
                    ids.append(aid)

            # Resumption token for pagination
            token_el = root.find(".//oai:resumptionToken", ns)
            if token_el is not None and token_el.text:
                url = f"{_OAI_URL}?verb=ListIdentifiers&resumptionToken={token_el.text}"
                time.sleep(1)
            else:
                break
        except Exception as e:
            logger.warning("OAI-PMH error: %s", e)
            break

    return ids[:max_results]
