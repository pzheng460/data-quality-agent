"""Benchmark report generation: rich console table, JSON, and Markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from dq.benchmark import BenchmarkReport

# Thresholds for verdict classification
STRONG_THRESHOLD = 0.05   # >5% = discriminates
WEAK_THRESHOLD = 0.02     # 2-5% = weak signal
# <2% = no signal


def _verdict(delta: float) -> tuple[str, str]:
    """Return (label, style) for a discrimination delta."""
    if delta >= STRONG_THRESHOLD:
        return "✅ Discriminates", "green"
    elif delta >= WEAK_THRESHOLD:
        return "⚠️ Weak signal", "yellow"
    else:
        return "— No signal", "dim"


def _verdict_md(delta: float) -> str:
    """Return markdown verdict string."""
    if delta >= STRONG_THRESHOLD:
        return "✅ Discriminates"
    elif delta >= WEAK_THRESHOLD:
        return "⚠️ Weak signal"
    else:
        return "— No signal"


def print_benchmark_report(report: BenchmarkReport, console: Console | None = None) -> None:
    """Print a rich comparison table to the console."""
    console = console or Console()

    ds_names = list(report.datasets.keys())
    if len(ds_names) < 2:
        console.print("[yellow]Need at least 2 datasets for comparison.[/yellow]")
        return

    # Gather all filter names (preserving order from first dataset)
    all_filters: list[str] = []
    seen: set[str] = set()
    for dr in report.datasets.values():
        for f in dr.per_filter_pass_rate:
            if f not in seen:
                all_filters.append(f)
                seen.add(f)

    discrimination = report.discrimination_scores()

    # Title
    console.print()
    console.print(
        f"[bold]📊 Data Quality Benchmark: {' vs '.join(ds_names)}[/bold]",
        highlight=False,
    )
    console.print(f"[dim]Config: {report.config_path or 'default'} | "
                  f"Samples: {report.num_samples or 'all'}[/dim]")
    console.print()

    # Build the table
    table = Table(padding=(0, 1), show_edge=True)
    table.add_column("Filter", style="bold", min_width=20)
    for name in ds_names:
        table.add_column(name, justify="right", min_width=12)
    table.add_column("Δ", justify="right", min_width=10)
    table.add_column("Verdict", min_width=18)

    # Per-filter rows
    for f in all_filters:
        row: list[str | Text] = [f]
        rates = []
        for name in ds_names:
            rate = report.datasets[name].per_filter_pass_rate.get(f, 0.0)
            rates.append(rate)
            row.append(f"{rate:.1%}")

        delta = discrimination.get(f, 0.0)
        row.append(f"+{delta:.1%}")
        label, style = _verdict(delta)
        row.append(Text(label, style=style))

        table.add_row(*row)

    # Overall pass rate row
    table.add_section()
    overall_row: list[str | Text] = ["Overall pipeline"]
    overall_rates = []
    for name in ds_names:
        rate = report.datasets[name].overall_pass_rate
        overall_rates.append(rate)
        overall_row.append(f"{rate:.1%}")

    overall_delta = max(overall_rates) - min(overall_rates) if overall_rates else 0.0
    overall_row.append(f"+{overall_delta:.1%}")
    label, style = _verdict(overall_delta)
    overall_row.append(Text(f"{'✅ Validated' if overall_delta >= STRONG_THRESHOLD else label}", style=style))

    table.add_row(*overall_row, style="bold")

    console.print(table)
    console.print()

    # Legend
    console.print("[green]✅ = Cleaned version passes significantly more (>5%) → filter catches real issues[/green]")
    console.print("[yellow]⚠️ = Small difference (2-5%) → filter has weak signal for this data type[/yellow]")
    console.print("[dim]— = No meaningful difference (<2%) → filter not relevant for SFT data[/dim]")
    console.print()

    # Show per-filter examples of docs that FAIL in original but PASS in cleaned
    _print_sample_failures(report, ds_names, all_filters, discrimination, console)


def _print_sample_failures(
    report: BenchmarkReport,
    ds_names: list[str],
    all_filters: list[str],
    discrimination: dict[str, float],
    console: Console,
) -> None:
    """Print example docs that fail in original dataset (shows what cleaning fixed)."""
    if len(ds_names) < 2:
        return

    # Identify the "original" dataset (first one) to show its failures
    orig_name = ds_names[0]
    orig_result = report.datasets[orig_name]

    discriminating_filters = [f for f in all_filters if discrimination.get(f, 0.0) >= WEAK_THRESHOLD]
    if not discriminating_filters:
        return

    console.print("[bold]Sample failures from original dataset (issues the cleaning addressed):[/bold]")
    console.print()

    for f in discriminating_filters:
        fr = orig_result.per_filter.get(f)
        if not fr or not fr.sample_failed:
            continue

        console.print(f"  [bold]{f}[/bold] — failed {fr.failed}/{fr.total} docs:")
        for i, sample in enumerate(fr.sample_failed[:3]):
            preview = sample.get("text_preview", "")[:120].replace("\n", " ")
            reason = sample.get("reason", {})
            reason_str = ", ".join(f"{k}={v}" for k, v in reason.items()) if isinstance(reason, dict) else str(reason)
            console.print(f"    [{i+1}] [dim]{preview}...[/dim]")
            console.print(f"        Reason: {reason_str}")
        console.print()


def benchmark_to_json(report: BenchmarkReport, path: str | Path | None = None) -> str:
    """Serialize benchmark report to JSON."""
    discrimination = report.discrimination_scores()

    data = {
        "config_path": report.config_path,
        "num_samples": report.num_samples,
        "datasets": {},
        "discrimination": {},
    }

    for f, delta in discrimination.items():
        data["discrimination"][f] = {
            "delta": round(delta, 4),
            "verdict": _verdict_md(delta),
        }

    for name, dr in report.datasets.items():
        ds_data: dict = {
            "num_docs": dr.num_docs,
            "overall_pass_rate": round(dr.overall_pass_rate, 4),
            "per_filter": {},
        }
        for fname, fr in dr.per_filter.items():
            ds_data["per_filter"][fname] = {
                "total": fr.total,
                "passed": fr.passed,
                "failed": fr.failed,
                "pass_rate": round(fr.pass_rate, 4),
                "sample_failed": fr.sample_failed[:3],
            }
        # Also include flat pass rate dict for backward compat
        ds_data["per_filter_pass_rate"] = {
            k: round(v, 4) for k, v in dr.per_filter_pass_rate.items()
        }
        ds_data["pipeline_stats"] = dr.stats.to_dict()
        data["datasets"][name] = ds_data

    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json_str, encoding="utf-8")
    return json_str


def benchmark_to_markdown(report: BenchmarkReport, path: str | Path | None = None) -> str:
    """Generate a Markdown comparison table from the benchmark report."""
    ds_names = list(report.datasets.keys())
    discrimination = report.discrimination_scores()

    all_filters: list[str] = []
    seen: set[str] = set()
    for dr in report.datasets.values():
        for f in dr.per_filter_pass_rate:
            if f not in seen:
                all_filters.append(f)
                seen.add(f)

    lines: list[str] = []
    lines.append(f"# Data Quality Benchmark: {' vs '.join(ds_names)}")
    lines.append("")
    lines.append(f"**Samples per dataset**: {report.num_samples or 'all'}")
    if report.config_path:
        lines.append(f"**Config**: `{report.config_path}`")
    lines.append("")

    # Header
    header = "| Filter |"
    separator = "|--------|"
    for name in ds_names:
        header += f" {name} |"
        separator += "--------|"
    header += " Δ | Verdict |"
    separator += "--------|--------|"

    lines.append(header)
    lines.append(separator)

    # Per-filter rows
    for f in all_filters:
        row = f"| {f} |"
        for name in ds_names:
            rate = report.datasets[name].per_filter_pass_rate.get(f, 0.0)
            row += f" {rate:.1%} |"
        delta = discrimination.get(f, 0.0)
        row += f" +{delta:.1%} | {_verdict_md(delta)} |"
        lines.append(row)

    # Overall row
    row = "| **Overall pipeline** |"
    overall_rates = []
    for name in ds_names:
        rate = report.datasets[name].overall_pass_rate
        overall_rates.append(rate)
        row += f" **{rate:.1%}** |"
    overall_delta = max(overall_rates) - min(overall_rates) if overall_rates else 0.0
    verdict = "✅ Validated" if overall_delta >= STRONG_THRESHOLD else _verdict_md(overall_delta)
    row += f" **+{overall_delta:.1%}** | **{verdict}** |"
    lines.append(row)

    lines.append("")
    lines.append("### Legend")
    lines.append("")
    lines.append("- ✅ = Cleaned version passes significantly more (>5%) → filter catches real issues")
    lines.append("- ⚠️ = Small difference (2-5%) → filter has weak signal for this data type")
    lines.append("- — = No meaningful difference (<2%) → filter not relevant for SFT data")
    lines.append("")

    # Sample failures section
    if len(ds_names) >= 2:
        orig_name = ds_names[0]
        orig_result = report.datasets[orig_name]
        discriminating = [f for f in all_filters if discrimination.get(f, 0.0) >= WEAK_THRESHOLD]

        if discriminating:
            lines.append(f"### Sample failures from {orig_name}")
            lines.append("")
            lines.append("Documents that **fail** in the original dataset (issues the cleaning addressed):")
            lines.append("")

            for f in discriminating:
                fr = orig_result.per_filter.get(f)
                if not fr or not fr.sample_failed:
                    continue
                lines.append(f"#### {f} — failed {fr.failed}/{fr.total}")
                lines.append("")
                for i, sample in enumerate(fr.sample_failed[:3]):
                    preview = sample.get("text_preview", "")[:150].replace("\n", " ")
                    reason = sample.get("reason", {})
                    if isinstance(reason, dict):
                        reason_str = ", ".join(f"`{k}={v}`" for k, v in reason.items())
                    else:
                        reason_str = f"`{reason}`"
                    lines.append(f"{i+1}. `{preview}...`")
                    lines.append(f"   - Reason: {reason_str}")
                lines.append("")

    md = "\n".join(lines)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(md, encoding="utf-8")
    return md
