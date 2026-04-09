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
import re
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from typing import Iterator

from dq.ingest.base import IngestSource

logger = logging.getLogger(__name__)

_EPRINT_URL = "https://arxiv.org/e-print"
_API_URL = "http://export.arxiv.org/api/query"
_OAI_URL = "http://export.arxiv.org/oai2"


class ArxivSource(IngestSource):
    """Ingest papers from arxiv by ID list or date range."""

    name = "arxiv"

    def __init__(
        self,
        ids: list[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        categories: list[str] | None = None,
        delay: float = 3.0,
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
    """Convert LaTeX to text. Leaves some residuals for ArxivFilter to clean."""
    tex = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.MULTILINE)  # comments
    m = re.search(r"\\begin\{document\}", tex)
    if m:
        tex = tex[m.end():]
    tex = re.sub(r"\\end\{document\}.*", "", tex, flags=re.DOTALL)

    # Sections → markdown headings
    tex = re.sub(r"\\title\{[^}]*\}", "", tex)
    tex = re.sub(r"\\section\*?\{([^}]+)\}", r"\n## \1\n", tex)
    tex = re.sub(r"\\subsection\*?\{([^}]+)\}", r"\n### \1\n", tex)
    tex = re.sub(r"\\subsubsection\*?\{([^}]+)\}", r"\n#### \1\n", tex)
    tex = re.sub(r"\\paragraph\*?\{([^}]+)\}", r"\n**\1**\n", tex)
    tex = re.sub(r"\\begin\{abstract\}", "\n## Abstract\n", tex)
    tex = re.sub(r"\\end\{abstract\}", "\n", tex)

    # Remove author/date metadata
    tex = re.sub(r"\\(?:author|date|affiliation|institute|email)\b[^{]*(?:\{[^}]*\})?", "", tex)
    tex = re.sub(r"\\maketitle", "", tex)

    # Figures/tables → placeholders (ArxivFilter will strip [Figure] and preserve captions)
    tex = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "[Figure]", tex, flags=re.DOTALL)
    tex = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "[Table]", tex, flags=re.DOTALL)

    # Math environments → $$
    for env in ["equation", "align", "alignat", "gather", "multline", "eqnarray"]:
        tex = re.sub(rf"\\begin\{{{env}\*?\}}", "$$", tex)
        tex = re.sub(rf"\\end\{{{env}\*?\}}", "$$", tex)

    # List environments → strip markers, keep \item
    for env in ["itemize", "enumerate", "description"]:
        tex = re.sub(rf"\\begin\{{{env}\*?\}}(?:\[[^\]]*\])?", "", tex)
        tex = re.sub(rf"\\end\{{{env}\*?\}}", "", tex)
    tex = re.sub(r"\\item\b\s*(?:\[[^\]]*\])?\s*", "\n- ", tex)

    # Cleanup
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return f"# {title}\n\n{tex.strip()}"


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
