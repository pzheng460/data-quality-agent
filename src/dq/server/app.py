"""FastAPI backend for controlling the pipeline and browsing results."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="dq Pipeline Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Global state ──

_state: dict[str, Any] = {
    "status": "idle",          # idle | running | finished | error
    "current_phase": None,
    "progress": [],            # list of phase result dicts
    "error": None,
    "config_path": None,
    "input_path": None,
    "output_dir": None,
}
_lock = threading.Lock()
_events: list[dict] = []      # SSE event log


def _push_event(event: dict) -> None:
    with _lock:
        _events.append(event)


# ── Models ──

class RunRequest(BaseModel):
    input_path: str
    output_dir: str
    config_path: str
    workers: int | None = None
    num_samples: int = 0
    resume: bool = True


class PhaseRunRequest(BaseModel):
    input_path: str
    output_dir: str
    config_path: str
    phase: int
    workers: int | None = None
    num_samples: int = 0


# ── Pipeline execution in background thread ──

def _run_pipeline(req: RunRequest) -> None:
    """Run 4-stage pipeline in background thread."""
    from dq.runner.engine import PhaseEngine
    from dq.runner.stages import stage_ingest, stage_extract, stage_curate, stage_package
    from dq.shared.stats import save_overview

    with _lock:
        _state["status"] = "running"
        _state["current_phase"] = None
        _state["progress"] = []
        _state["error"] = None
        _state["config_path"] = req.config_path
        _state["input_path"] = req.input_path
        _state["output_dir"] = req.output_dir

    try:
        engine = PhaseEngine(
            config_path=req.config_path,
            input_path=req.input_path,
            output_dir=req.output_dir,
            workers=req.workers,
            num_samples=req.num_samples,
        )
        engine.output_dir.mkdir(parents=True, exist_ok=True)

        stage_list = [
            ("ingestion", stage_ingest),
            ("extraction", stage_extract),
            ("curation", stage_curate),
            ("packaging", stage_package),
        ]

        all_stats = []
        for name, func in stage_list:
            if req.resume and engine.is_stage_done(name):
                _push_event({"type": "phase_skip", "phase": name})
                with _lock:
                    _state["progress"].append({"phase": name, "skipped": True})
                continue

            with _lock:
                _state["current_phase"] = name
            _push_event({"type": "phase_start", "phase": name})

            stats = func(engine)
            engine.mark_stage_done(name)
            all_stats.append(stats)

            stats_dir = engine.output_dir / "stats" / engine.version
            stats_dir.mkdir(parents=True, exist_ok=True)
            stats.save(stats_dir / f"{name}.json")

            result = stats.to_dict()
            with _lock:
                _state["progress"].append(result)
            _push_event({"type": "phase_done", "phase": name, "stats": result})

        if all_stats:
            stats_dir = engine.output_dir / "stats" / engine.version
            save_overview(stats_dir, all_stats, engine.version, config_hash=engine.config_hash)

        with _lock:
            _state["status"] = "finished"
            _state["current_phase"] = None
        _invalidate_cache(req.output_dir)
        _push_event({"type": "pipeline_done"})

    except Exception as e:
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(e)
        _push_event({"type": "error", "message": str(e)})
        logger.exception("Pipeline failed")


# ── API endpoints ──

@app.get("/api/status")
def get_status():
    """Current pipeline status."""
    with _lock:
        return dict(_state)


@app.post("/api/run")
def start_pipeline(req: RunRequest):
    """Start the full pipeline in background."""
    with _lock:
        if _state.get("status") == "running":
            raise HTTPException(400, "Pipeline already running")
        _events.clear()

    import threading
    t = threading.Thread(target=_run_pipeline, args=(req,), daemon=True)
    t.start()
    return {"status": "started"}


@app.post("/api/run-phase")
def start_phase(req: PhaseRunRequest):
    """Run a single phase."""
    with _lock:
        if _state.get("status") == "running":
            raise HTTPException(400, "Pipeline already running")
        _events.clear()

    import threading

    def _run():
        from dq.runner.engine import PhaseEngine
        from dq.runner.stages import stage_ingest, stage_extract, stage_curate, stage_package

        stage_map = {
            1: ("ingestion", stage_ingest),
            2: ("extraction", stage_extract),
            3: ("curation", stage_curate),
            4: ("packaging", stage_package),
        }

        try:
            with _lock:
                _state["status"] = "running"

            engine = PhaseEngine(
                config_path=req.config_path,
                input_path=req.input_path,
                output_dir=req.output_dir,
                workers=req.workers,
                num_samples=req.num_samples,
            )
            engine.output_dir.mkdir(parents=True, exist_ok=True)
            name, func = stage_map[req.phase]
            with _lock:
                _state["current_phase"] = name
            _push_event({"type": "phase_start", "phase": name})

            stats = func(engine)
            engine.mark_stage_done(name)

            stats_dir = engine.output_dir / "stats" / engine.version
            stats_dir.mkdir(parents=True, exist_ok=True)
            stats.save(stats_dir / f"{name}.json")

            result = stats.to_dict()
            with _lock:
                _state["progress"].append(result)
                _state["status"] = "finished"
                _state["current_phase"] = None
            _push_event({"type": "phase_done", "phase": name, "stats": result})

        except Exception as e:
            with _lock:
                _state["status"] = "error"
                _state["error"] = str(e)
            _push_event({"type": "error", "message": str(e)})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "phase": req.phase}


@app.get("/api/events")
async def event_stream():
    """SSE stream of pipeline progress events."""
    async def generate():
        last_idx = 0
        while True:
            with _lock:
                new_events = _events[last_idx:]
                last_idx = len(_events)
                status = _state.get("status")
            for ev in new_events:
                yield f"data: {__import__('json').dumps(ev)}\n\n"
            if status in ("finished", "error", "idle") and not new_events:
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/config")
def get_config(path: str):
    """Read a YAML config file."""
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"Config not found: {path}")
    with open(p) as f:
        return __import__("yaml").safe_load(f)


@app.put("/api/config")
def save_config(path: str, body: dict):
    """Write updated config to YAML."""
    p = Path(path)
    with open(p, "w") as f:
        yaml.dump(body, f, default_flow_style=False, allow_unicode=True)
    return {"status": "saved", "path": str(p)}


@app.get("/api/phases")
def list_phases(output_dir: str):
    """List pipeline stages and their completion status."""
    out = Path(output_dir)
    stages = []
    for num, name in [(1, "ingestion"), (2, "extraction"), (3, "curation"), (4, "packaging")]:
        done = (out / f".{name}_SUCCESS").exists()
        stats_file = None
        for stats_dir in out.glob("stats/*/"):
            sf = stats_dir / f"{name}.json"
            if sf.exists():
                stats_file = str(sf)
                break
        stages.append({"phase": num, "name": name, "done": done, "stats_file": stats_file})
    return stages


@app.get("/api/stages/all")
def get_all_stages(output_dir: str):
    """Get all stage statuses and stats in one call."""
    out = Path(output_dir)
    result = []
    for num, name in [(1, "ingestion"), (2, "extraction"), (3, "curation"), (4, "packaging")]:
        entry: dict = {"phase": num, "name": name, "done": (out / f".{name}_SUCCESS").exists(), "stats": None}
        for stats_dir in out.glob("stats/*/"):
            sf = stats_dir / f"{name}.json"
            if sf.exists():
                with open(sf) as f:
                    entry["stats"] = json.load(f)
                break
        result.append(entry)
    return result


@app.get("/api/phase-stats/{phase_name}")
def get_phase_stats(phase_name: str, output_dir: str):
    """Get stats for a specific phase."""
    out = Path(output_dir)
    for stats_dir in out.glob("stats/*/"):
        sf = stats_dir / f"{phase_name}.json"
        if sf.exists():
            with open(sf) as f:
                return json.load(f)
    raise HTTPException(404, f"Stats not found for {phase_name}")


_docs_cache: dict[str, tuple[str, list[dict]]] = {}  # path -> (fingerprint, docs)


def _stage_fingerprint(stage_path: Path) -> str:
    """Compute fingerprint from shard files: count + total size + max mtime."""
    shards = sorted(stage_path.glob("*.jsonl.zst")) or sorted(stage_path.glob("*.jsonl"))
    if not shards:
        return "empty"
    total_size = sum(f.stat().st_size for f in shards)
    max_mtime = max(f.stat().st_mtime for f in shards)
    return f"{len(shards)}:{total_size}:{max_mtime}"


def _load_stage_docs(stage_path: Path) -> list[dict]:
    """Load and cache all docs from a stage directory."""
    from dq.shared.shard import read_shards
    key = str(stage_path)
    fp = _stage_fingerprint(stage_path)
    if key in _docs_cache and _docs_cache[key][0] == fp:
        return _docs_cache[key][1]
    docs = list(read_shards(stage_path))
    _docs_cache[key] = (fp, docs)
    return docs


def _invalidate_cache(output_dir: str | None = None):
    """Clear docs cache. Called when pipeline finishes."""
    if output_dir:
        prefix = str(output_dir)
        to_remove = [k for k in _docs_cache if k.startswith(prefix)]
        for k in to_remove:
            del _docs_cache[k]
    else:
        _docs_cache.clear()


@app.get("/api/docs/{stage}")
@app.get("/api/docs/{stage}/{sub}")
def list_docs(stage: str, output_dir: str, sub: str = "", offset: int = 0, limit: int = 20):
    """Browse documents from a pipeline stage (kept/rejected)."""
    stage_path = Path(output_dir) / stage / sub if sub else Path(output_dir) / stage
    if not stage_path.exists():
        raise HTTPException(404, f"Stage not found: {stage}/{sub}")

    all_docs = _load_stage_docs(stage_path)
    page = all_docs[offset:offset + limit]

    docs = []
    for doc in page:
        summary = dict(doc)
        text = summary.get("text", "")
        summary["text_preview"] = text[:300]
        summary["text_length"] = len(text)
        docs.append(summary)

    return {"docs": docs, "offset": offset, "limit": limit, "total": len(all_docs), "has_more": offset + limit < len(all_docs)}


@app.get("/api/raw-input")
def list_raw_input(input_path: str, offset: int = 0, limit: int = 20):
    """Read raw input file (JSONL) before any pipeline processing."""
    from dq.utils.io import read_docs

    p = Path(input_path)
    if not p.exists():
        raise HTTPException(404, f"Input file not found: {input_path}")

    docs = []
    count = 0
    for doc in read_docs(p):
        if count < offset:
            count += 1
            continue
        if len(docs) >= limit:
            break
        summary = dict(doc)
        text = summary.get("text", "")
        summary["text_preview"] = text[:300]
        summary["text_length"] = len(text)
        docs.append(summary)
        count += 1

    return {"docs": docs, "offset": offset, "limit": limit}


@app.get("/api/raw-input/doc")
def get_raw_input_doc(input_path: str, doc_id: str):
    """Get a single raw input document by ID."""
    from dq.utils.io import read_docs

    p = Path(input_path)
    if not p.exists():
        raise HTTPException(404, f"Input file not found: {input_path}")

    for doc in read_docs(p):
        if doc.get("id") == doc_id:
            return doc
    raise HTTPException(404, f"Doc {doc_id} not found in {input_path}")


@app.post("/api/cache/clear")
def clear_cache(output_dir: str = ""):
    """Clear the docs cache for a specific output dir (or all)."""
    _invalidate_cache(output_dir or None)
    return {"status": "cleared"}


@app.get("/api/doc")
def get_full_doc(output_dir: str, stage: str, doc_id: str, sub: str = ""):
    """Get a single document by ID with full text."""
    stage_path = Path(output_dir) / stage / sub if sub else Path(output_dir) / stage
    if not stage_path.exists():
        raise HTTPException(404, f"Stage not found: {stage}/{sub}")
    for doc in _load_stage_docs(stage_path):
        if doc.get("id") == doc_id:
            return doc
    raise HTTPException(404, f"Doc not found: {doc_id}")


@app.get("/api/overview")
def get_overview(output_dir: str):
    """Get pipeline overview stats."""
    out = Path(output_dir)
    # Find the overview file
    for stats_dir in sorted(out.glob("stats/*/")):
        ov = stats_dir / "overview.json"
        if ov.exists():
            with open(ov) as f:
                return json.load(f)
    # Fallback: build from individual phase stats
    phases = {}
    for stats_dir in out.glob("stats/*/"):
        for sf in sorted(stats_dir.glob("phase*.json")):
            with open(sf) as f:
                data = json.load(f)
            phases[data["phase"]] = {
                "input": data["input_count"],
                "output": data["output_count"],
                "keep_rate": data["keep_rate"],
                "reject_reasons": data.get("reject_reasons", {}),
            }
    if phases:
        return {"version": "unknown", "phases": phases}
    raise HTTPException(404, "No stats found")


# ── Benchmark (dq bench) ──

_bench_state: dict[str, Any] = {
    "status": "idle",  # idle | running | done | error
    "result": None,
    "error": None,
}


class BenchRequest(BaseModel):
    input_path: str
    config_path: str = ""
    num_samples: int = 100
    data_type: str = "auto"
    workers: int = 4
    with_llm_scoring: bool = False
    llm_samples: int = 50


@app.post("/api/bench")
def start_bench(req: BenchRequest):
    """Run dq bench in background. Poll /api/bench/status for results."""
    with _lock:
        if _bench_state.get("status") == "running":
            raise HTTPException(400, "Benchmark already running")
        _bench_state.update(status="running", result=None, error=None)

    def _run():
        try:
            from dq.benchmark.runner import run_benchmark
            from dq.benchmark_report import benchmark_to_json
            from dq.utils.io import read_docs

            docs = list(read_docs(Path(req.input_path)))
            if req.num_samples > 0 and len(docs) > req.num_samples:
                import random
                docs = random.sample(docs, req.num_samples)

            report = run_benchmark(
                config_path=req.config_path or None,
                datasets={"input": docs},
                n=0,
                data_type=req.data_type,
                workers=req.workers,
                sft_samples=req.llm_samples if req.with_llm_scoring else 0,
                save_rejected=False,
            )

            result_json = json.loads(benchmark_to_json(report))
            with _lock:
                _bench_state["status"] = "done"
                _bench_state["result"] = result_json
        except Exception as e:
            logger.exception("Benchmark failed")
            with _lock:
                _bench_state["status"] = "error"
                _bench_state["error"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@app.get("/api/bench/status")
def bench_status():
    """Poll benchmark status and results."""
    with _lock:
        return dict(_bench_state)


# ── Ingestion state ──

_ingest_state: dict[str, Any] = {
    "status": "idle",      # idle | downloading | done | error
    "total": 0,
    "downloaded": 0,
    "papers": [],          # list of {arxiv_id, title, chars, source_method}
    "error": None,
    "output_path": None,
}


# ── Unified ingestion endpoints ──


@app.get("/api/sources")
def get_sources():
    """List available data sources grouped by domain."""
    from dq.stages.ingestion import ensure_sources_registered, list_sources
    ensure_sources_registered()
    return list_sources()


class IngestRequest(BaseModel):
    source: str
    params: dict[str, Any] = {}
    output_path: str
    limit: int = 0


@app.post("/api/ingest")
def start_ingest(req: IngestRequest):
    """Start ingestion from any registered source."""
    from dq.stages.ingestion import ensure_sources_registered
    from dq.stages.ingestion.registry import get_source_class
    ensure_sources_registered()

    with _lock:
        if _ingest_state.get("status") == "downloading":
            raise HTTPException(400, "Download already in progress")

    try:
        src_cls = get_source_class(req.source)
    except ValueError as e:
        raise HTTPException(400, str(e))

    src = src_cls(**req.params)

    # Validate required params early — fail fast instead of hanging in background
    params = req.params or {}
    if hasattr(src, "domain") and src.domain == "arxiv":
        if not params.get("ids") and not params.get("from_date"):
            raise HTTPException(400, "Provide either arxiv IDs or a date range")

    with _lock:
        _ingest_state.update(
            status="downloading", total=0, downloaded=0,
            papers=[], error=None, output_path=req.output_path,
        )
        _events.clear()

    def _run():
        _run_ingest(src.fetch(limit=req.limit), req.output_path)

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "source": req.source}


@app.get("/api/ingest/status")
def ingest_status():
    with _lock:
        return dict(_ingest_state)


def _run_ingest(doc_iter, output_path: str):
    """Shared ingestion worker: iterate docs, write to JSONL."""
    try:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for doc in doc_iter:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                meta = doc.get("metadata", {})
                paper_info = {
                    "arxiv_id": meta.get("arxiv_id", doc.get("id", "")),
                    "title": meta.get("title", ""),
                    "categories": meta.get("categories", []),
                    "primary_category": meta.get("primary_category", ""),
                    "abstract": (meta.get("abstract", "") or "")[:200],
                    "chars": len(doc.get("text", "")),
                    "source_method": doc.get("source", "unknown"),
                }
                with _lock:
                    _ingest_state["downloaded"] += 1
                    _ingest_state["papers"].append(paper_info)
                _push_event({"type": "paper_downloaded", **paper_info})
        with _lock:
            _ingest_state["status"] = "done"
        _push_event({"type": "ingest_done", "count": _ingest_state["downloaded"]})
    except Exception as e:
        with _lock:
            _ingest_state["status"] = "error"
            _ingest_state["error"] = str(e)
        _push_event({"type": "error", "message": str(e)})


