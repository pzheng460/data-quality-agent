"""Quality report generation (JSON + Markdown)."""

import json
from pathlib import Path
from typing import Any

from dq.pipeline import PipelineStats


def stats_to_json(stats: PipelineStats, path: str | Path | None = None) -> str:
    """Convert pipeline stats to JSON string. Optionally write to file."""
    data = stats.to_dict()
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    if path:
        Path(path).write_text(json_str, encoding="utf-8")
    return json_str


def stats_to_markdown(stats: PipelineStats, path: str | Path | None = None) -> str:
    """Convert pipeline stats to Markdown report. Optionally write to file."""
    lines: list[str] = []
    lines.append("# Data Quality Report")
    lines.append("")
    lines.append("## Overall Statistics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total documents in | {stats.total_in} |")
    lines.append(f"| Total documents out | {stats.total_out} |")
    lines.append(f"| Total dropped | {stats.total_dropped} |")
    lines.append(f"| Overall drop rate | {stats.overall_drop_rate:.2%} |")
    lines.append("")

    if stats.filter_stats:
        lines.append("## Per-Filter Statistics")
        lines.append("")
        lines.append("| Filter | Docs In | Docs Out | Dropped | Drop Rate |")
        lines.append("|--------|---------|----------|---------|-----------|")
        for fs in stats.filter_stats:
            lines.append(
                f"| {fs.name} | {fs.docs_in} | {fs.docs_out} | "
                f"{fs.docs_dropped} | {fs.drop_rate:.2%} |"
            )
        lines.append("")

        # Sample drops
        has_samples = any(fs.sample_drops for fs in stats.filter_stats)
        if has_samples:
            lines.append("## Sample Dropped Documents")
            lines.append("")
            for fs in stats.filter_stats:
                if not fs.sample_drops:
                    continue
                lines.append(f"### {fs.name}")
                lines.append("")
                for i, sample in enumerate(fs.sample_drops[:3], 1):
                    reason = sample.get("reason", {})
                    preview = sample.get("text_preview", "")[:100]
                    lines.append(f"**Sample {i}**: {reason}")
                    lines.append(f"> {preview}...")
                    lines.append("")

    md = "\n".join(lines)
    if path:
        Path(path).write_text(md, encoding="utf-8")
    return md


def generate_report(
    stats: PipelineStats,
    dedup_stats: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Generate both JSON and Markdown reports.

    Returns dict with 'json' and 'markdown' content strings.
    """
    result: dict[str, str] = {}

    json_path = Path(output_dir) / "report.json" if output_dir else None
    md_path = Path(output_dir) / "report.md" if output_dir else None

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    result["json"] = stats_to_json(stats, json_path)
    result["markdown"] = stats_to_markdown(stats, md_path)

    # Append dedup stats to JSON if present
    if dedup_stats:
        data = json.loads(result["json"])
        data["dedup"] = dedup_stats
        result["json"] = json.dumps(data, indent=2, ensure_ascii=False)
        if json_path:
            json_path.write_text(result["json"], encoding="utf-8")

    return result
