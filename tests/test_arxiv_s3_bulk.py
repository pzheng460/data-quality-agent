"""Tests for the arxiv S3 bulk ingestion source.

No network access — we only exercise the parsing + extraction helpers.
"""

from __future__ import annotations

import gzip
import io
import tarfile

import pytest

from dq.stages.ingestion.arxiv_s3_bulk import (
    ArxivS3BulkSource,
    _extract_tex_blob,
    _merge_tex_files,
    _parse_arxiv_id,
)


@pytest.mark.parametrize("name,expected", [
    ("2310.12345.gz", "2310.12345"),
    ("2310.12345v3", "2310.12345"),
    ("arXiv_src_2310/2310.00001.gz", "2310.00001"),
    ("hep-th/0001001", "hep-th/0001001"),
    ("hep-th0001001", "hep-th/0001001"),
    ("0704.0001", "0704.0001"),
    ("not_an_id.txt", None),
    ("dir/2310.12345.tar.gz", "2310.12345"),
])
def test_parse_arxiv_id(name, expected):
    assert _parse_arxiv_id(name) == expected


def test_merge_tex_picks_main_doc():
    tex = {
        "helper.tex": "bla",
        "main.tex": r"\begin{document} This is main. \end{document}",
    }
    assert "This is main" in _merge_tex_files(tex)


def test_merge_tex_inlines_input_directives():
    tex = {
        "main.tex": r"\begin{document} Intro \input{chapter1} end \end{document}",
        "chapter1.tex": "CHAPTER ONE BODY",
    }
    merged = _merge_tex_files(tex)
    assert "CHAPTER ONE BODY" in merged


def test_extract_tex_blob_handles_plain_gzip():
    # Single-file submission: gzip of a .tex
    import gzip as gz
    body = r"\documentclass{article} \begin{document} Hello \end{document}"
    blob = gz.compress(body.encode())
    out = _extract_tex_blob(blob)
    assert out and "Hello" in out


def test_extract_tex_from_tarball():
    import gzip as gz
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = r"\begin{document} OK \end{document}".encode()
        info = tarfile.TarInfo(name="paper.tex")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    out = _extract_tex_blob(buf.getvalue())
    assert out and "OK" in out


def test_source_registers():
    from dq.stages.ingestion import ensure_sources_registered, list_sources
    ensure_sources_registered()
    srcs = list_sources()
    arxiv = [s for s in srcs.get("arxiv", []) if s.get("name") == "arxiv_s3_bulk"]
    assert arxiv, "arxiv_s3_bulk should be registered"


def test_source_requires_boto3(monkeypatch):
    # Simulate missing boto3 → source should raise RuntimeError on fetch
    import sys
    monkeypatch.setitem(sys.modules, "boto3", None)
    src = ArxivS3BulkSource()
    gen = src.fetch(limit=1)
    import pytest
    with pytest.raises(RuntimeError, match="boto3"):
        next(gen)


def test_figures_extracted_to_disk(tmp_path):
    """save_figures=True writes image files + doc.metadata.figures has paths."""
    from dq.stages.ingestion.arxiv_s3_bulk import _extract_tex_and_figures
    from pathlib import Path

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in [
            ("paper.tex", rb"\begin{document} body \end{document}"),
            ("fig1.png", b"\x89PNG\r\n\x00" + b"x" * 50),
            ("chart.pdf", b"%PDF-1.4\n" + b"y" * 50),
            ("notes.bib", b"@article{foo,...}"),  # non-figure file, should be ignored
        ]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    out_dir = tmp_path / "arxiv_2310.12345"
    tex, figs = _extract_tex_and_figures(buf.getvalue(), save_figures=True, out_dir=out_dir)
    assert tex and r"\begin{document}" in tex
    names = sorted(f["name"] for f in figs)
    assert names == ["chart.pdf", "fig1.png"]
    for f in figs:
        assert Path(f["path"]).exists()
        assert "bytes" not in f


def test_figures_bytes_mode_no_disk_writes(tmp_path):
    """save_figures=False → figures returned with 'bytes', nothing on disk."""
    import io, tarfile
    from dq.stages.ingestion.arxiv_s3_bulk import _extract_tex_and_figures
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in [
            ("paper.tex", rb"\begin{document} body \end{document}"),
            ("pic.jpg", b"\xff\xd8\xff img"),
        ]:
            info = tarfile.TarInfo(name=name); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    tex, figs = _extract_tex_and_figures(buf.getvalue(), save_figures=False)
    assert tex
    assert len(figs) == 1
    assert figs[0]["name"] == "pic.jpg"
    assert figs[0]["bytes"] == b"\xff\xd8\xff img"
    assert not any(tmp_path.iterdir())  # nothing written


def test_extract_tex_blob_still_works_without_figures():
    """Back-compat: _extract_tex_blob returns just the LaTeX text."""
    import io
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = rb"\begin{document} hi \end{document}"
        info = tarfile.TarInfo(name="main.tex"); info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    tex = _extract_tex_blob(buf.getvalue())
    assert tex and "hi" in tex


