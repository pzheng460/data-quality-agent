"""Fetch raw LaTeX source from arxiv e-print.

Yields raw LaTeX — extraction (LaTeXML conversion) happens in the
extraction stage, not here.

Usage:
    src = ArxivSource(ids=["2310.06825", "2307.09288"])
    for doc in src.fetch(limit=10):
        print(doc["id"], len(doc["text"]))  # text = raw LaTeX
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import re
import tarfile
import time
import urllib.request
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Iterator

from dq.stages.ingestion.base import IngestSource
from dq.stages.ingestion.registry import register_source

logger = logging.getLogger(__name__)

_EPRINT_URL = "https://arxiv.org/e-print"
_API_URL = "http://export.arxiv.org/api/query"
_OAI_URL = "http://export.arxiv.org/oai2"


@register_source("arxiv_latexml")
class ArxivSource(IngestSource):
    """Fetch raw LaTeX from arxiv. Extraction happens in extraction stage."""

    name = "arxiv_latexml"
    domain = "arxiv"
    priority = 100
    output_format = "latex"

    @classmethod
    def params_schema(cls):
        return {
            "ids": {"type": "list", "label": "Arxiv IDs", "required": False},
            "from_date": {"type": "string", "label": "From date", "required": False},
            "to_date": {"type": "string", "label": "To date", "required": False},
            "categories": {"type": "list", "label": "Categories", "required": False},
            "delay": {"type": "number", "label": "Delay (s)", "default": 3.0},
            "save_figures": {"type": "bool", "label": "Save figures to disk", "default": True},
            "image_dir": {"type": "string", "label": "Figures directory",
                          "default": "/tmp/arxiv_images"},
        }

    def __init__(
        self,
        ids: list[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        categories: list[str] | None = None,
        delay: float = 3.0,
        save_figures: bool = True,
        image_dir: str = "/tmp/arxiv_images",
        **_kwargs,
    ) -> None:
        self.ids = ids
        self.from_date = from_date
        self.to_date = to_date
        self.categories = set(categories) if categories else None
        self.delay = delay
        self.save_figures = bool(save_figures)
        self.image_dir = image_dir

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
                save_dir = None
                if self.save_figures:
                    save_dir = Path(self.image_dir) / f"arxiv_{aid}".replace("/", "_")
                tex, figures = _download_latex_with_figures(aid, save_dir)
                if not tex or len(tex) < 200:
                    logger.warning("Skip %s: no/short source", aid)
                    continue
                doc_meta = meta.get(aid, {"arxiv_id": aid, "title": aid})
                if figures:
                    doc_meta = dict(doc_meta)
                    doc_meta["figures"] = figures
                yield {
                    "id": f"arxiv_{aid}",
                    "text": tex,  # raw LaTeX — extraction stage converts to text
                    "source": "arxiv",
                    "metadata": doc_meta,
                }
                count += 1
                title = meta.get(aid, {}).get("title", "")
                logger.info("Fetched %s: %s (%d chars, %d figures)",
                            aid, title[:50], len(tex), len(figures))
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


_FIGURE_EXTS = (".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg", ".gif")


def _download_latex_with_figures(
    arxiv_id: str,
    save_dir: Path | None = None,
) -> tuple[str | None, list[dict]]:
    """Download LaTeX source AND extract/save figure files.

    Returns (merged_tex, [{name, path}]). When save_dir is None, figures are
    not written to disk and the list is empty — same behavior as _download_latex.
    """
    url = f"{_EPRINT_URL}/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "dq-pipeline/1.0"})
    try:
        data = urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        return None, []

    figures: list[dict] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tex_files: dict[str, str] = {}
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                if name.endswith(".tex"):
                    f = tar.extractfile(m)
                    if f:
                        tex_files[name] = f.read().decode("utf-8", errors="replace")
                elif save_dir is not None and name.lower().endswith(_FIGURE_EXTS):
                    f = tar.extractfile(m)
                    if f is None:
                        continue
                    blob = f.read()
                    if not blob:
                        continue
                    save_dir.mkdir(parents=True, exist_ok=True)
                    fname = os.path.basename(name)
                    dest = save_dir / fname
                    # Avoid name collisions from subdirs
                    if dest.exists():
                        stem, ext = os.path.splitext(fname)
                        safe = "_".join(name.replace("/", "_").rsplit(".", 1))
                        dest = save_dir / safe
                    with open(dest, "wb") as fp:
                        fp.write(blob)
                    figures.append({"name": fname, "path": str(dest)})

            if not tex_files:
                return None, figures

            # Merge \input{} references, same as _download_latex
            main = None
            for n, content in tex_files.items():
                if r"\begin{document}" in content:
                    main = n
                    break
            if not main:
                main = max(tex_files, key=lambda k: len(tex_files[k]))
            content = tex_files[main]
            for _ in range(3):
                def _resolve(m: re.Match) -> str:
                    fname = m.group(1).strip()
                    if not fname.endswith('.tex'):
                        fname += '.tex'
                    for path, body in tex_files.items():
                        if path == fname or path.endswith('/' + fname):
                            return body
                    return ""
                content = re.sub(r"\\input\{([^}]+)\}", _resolve, content)
            return content, figures
    except tarfile.TarError:
        pass

    # Single gzipped .tex submission (no figures inside)
    try:
        return gzip.decompress(data).decode("utf-8", errors="replace"), []
    except OSError:
        return data.decode("utf-8", errors="replace"), []


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
