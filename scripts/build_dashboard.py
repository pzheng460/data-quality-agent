#!/usr/bin/env python3
"""Build dashboard JSON data from pipeline stats and samples.

Reads from:
  {data_dir}/stats/{version}/*.json
  {data_dir}/samples/{version}/*.jsonl

Writes to:
  dashboard/public/data/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def build(data_dir: Path, version: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_dir = data_dir / "stats" / version
    samples_dir = data_dir / "samples" / version

    # 1. Copy overview.json
    overview_path = stats_dir / "overview.json"
    if overview_path.exists():
        _copy(overview_path, output_dir / "overview.json")
    else:
        # Build from individual phase stats
        phases = {}
        for f in sorted(stats_dir.glob("phase*.json")):
            with open(f) as fh:
                phase_data = json.load(fh)
                phases[phase_data["phase"]] = {
                    "input": phase_data.get("input_count", 0),
                    "output": phase_data.get("output_count", 0),
                    "keep_rate": phase_data.get("keep_rate", 0),
                    "reject_reasons": phase_data.get("reject_reasons", {}),
                    "duration_seconds": phase_data.get("duration_seconds", 0),
                }
        overview = {"version": f"arxiv-{version}", "phases": phases, "config_sha256": ""}
        _write_json(overview, output_dir / "overview.json")

    # 2. Copy per-phase stats
    phase_stats = {}
    for f in sorted(stats_dir.glob("phase*.json")):
        with open(f) as fh:
            phase_stats[f.stem] = json.load(fh)
    if phase_stats:
        _write_json(phase_stats, output_dir / "phase_stats.json")

    # 3. Copy signal histograms if available
    hist_path = stats_dir / "signals_histograms.json"
    if hist_path.exists():
        _copy(hist_path, output_dir / "signals_histograms.json")

    # 4. Copy contamination data
    contam_path = stats_dir / "contamination.json"
    if contam_path.exists():
        _copy(contam_path, output_dir / "contamination.json")

    # 5. Copy golden results
    golden_path = stats_dir / "golden_results.json"
    if golden_path.exists():
        _copy(golden_path, output_dir / "golden_results.json")

    # 6. Build samples index
    samples_out = output_dir / "samples"
    samples_out.mkdir(exist_ok=True)
    index_entries = []
    if samples_dir.exists():
        for f in sorted(samples_dir.glob("*.jsonl")):
            with open(f) as fh:
                docs = [json.loads(line) for line in fh if line.strip()]
            out_name = f.stem + ".json"
            _write_json(docs, samples_out / out_name)
            index_entries.append({"name": f.stem, "path": out_name, "count": len(docs)})
    _write_json({"files": index_entries}, samples_out / "index.json")

    print(f"Dashboard data written to {output_dir}")


def _copy(src, dst):
    shutil.copy2(src, dst)


def _write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build dashboard JSON data")
    parser.add_argument("--data-dir", required=True, help="Pipeline output directory")
    parser.add_argument("--version", required=True, help="Pipeline version (e.g. v2025-04)")
    parser.add_argument("--output", default="dashboard/public/data", help="Output dir for JSON files")
    args = parser.parse_args()
    build(Path(args.data_dir), args.version, Path(args.output))


if __name__ == "__main__":
    main()
