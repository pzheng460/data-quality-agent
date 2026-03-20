"""Benchmark report generation: rich console table, JSON, and Markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from dq.benchmark import BenchmarkReport, RuleStats

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


def _collect_rule_names(report: BenchmarkReport, filter_name: str, ds_names: list[str]) -> list[str]:
    """Collect all rule names for a filter across all datasets, preserving order."""
    seen: set[str] = set()
    rules: list[str] = []
    for name in ds_names:
        for rule in report.rule_stats.get(name, {}).get(filter_name, {}):
            if rule not in seen:
                rules.append(rule)
                seen.add(rule)
    return rules


def _verdict_md(delta: float) -> str:
    """Return markdown verdict string."""
    if delta >= STRONG_THRESHOLD:
        return "✅ Discriminates"
    elif delta >= WEAK_THRESHOLD:
        return "⚠️ Weak signal"
    else:
        return "— No signal"


def _add_filter_row(
    table: Table,
    report: BenchmarkReport,
    f: str,
    ds_names: list[str],
    discrimination: dict[str, float],
) -> None:
    """Add a filter row with per-rule sub-rows to a table."""
    row: list[str | Text] = [Text(f, style="bold")]
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

    # Add per-rule sub-rows if rule_stats available
    if report.rule_stats:
        rule_names = _collect_rule_names(report, f, ds_names)
        for i, rule in enumerate(rule_names):
            is_last = i == len(rule_names) - 1
            prefix = "└─" if is_last else "├─"
            rule_row: list[str | Text] = [Text(f"  {prefix} {rule}", style="dim")]
            rule_rates = []
            for name in ds_names:
                rs = report.rule_stats.get(name, {}).get(f, {}).get(rule)
                rate = rs.pass_rate if rs else 1.0
                rule_rates.append(rate)
                rule_row.append(f"{rate:.1%}")

            rule_delta = max(rule_rates) - min(rule_rates) if len(rule_rates) >= 2 else 0.0
            rule_row.append(f"+{rule_delta:.1%}")
            rl, rs_style = _verdict(rule_delta)
            rule_row.append(Text(rl, style=rs_style))

            table.add_row(*rule_row)


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

    # Separate filters into layers
    heuristic_filters = [f for f in all_filters if f != "sft_rules"]
    sft_rule_filters = [f for f in all_filters if f == "sft_rules"]

    # Build the table
    table = Table(padding=(0, 1), show_edge=True)
    table.add_column("Filter", style="bold", min_width=20)
    for name in ds_names:
        table.add_column(name, justify="right", min_width=12)
    table.add_column("Δ", justify="right", min_width=10)
    table.add_column("Verdict", min_width=18)

    # Layer 1: Heuristic filters
    if heuristic_filters:
        num_cols = len(ds_names) + 3  # Filter + datasets + Δ + Verdict
        table.add_row(*([Text("Layer 1: Pre-training Rules", style="bold cyan")] + [""] * (num_cols - 1)))

    for f in heuristic_filters:
        _add_filter_row(table, report, f, ds_names, discrimination)

    # Layer 1: SFT Rules
    if sft_rule_filters:
        table.add_section()
        num_cols = len(ds_names) + 3
        table.add_row(*([Text("Layer 1: SFT Rules", style="bold cyan")] + [""] * (num_cols - 1)))

        for f in sft_rule_filters:
            _add_filter_row(table, report, f, ds_names, discrimination)

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

    # Layer 2: LLM Binary Judge
    if report.llm_scoring_enabled:
        _print_llm_scores(report, ds_names, console)


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


def _print_llm_scores(
    report: BenchmarkReport,
    ds_names: list[str],
    console: Console,
) -> None:
    """Print Layer 2 LLM quality scoring results."""
    # Check if any dataset has LLM scores
    has_scores = any(
        report.datasets[name].llm_scores is not None
        for name in ds_names
    )
    if not has_scores:
        return

    console.print()
    console.print(
        f"[bold]🤖 Layer 2: LLM Binary Judge ({report.llm_samples} samples)[/bold]",
        highlight=False,
    )
    console.print()

    # Determine data type from first dataset with scores
    first_scores = None
    for name in ds_names:
        if report.datasets[name].llm_scores:
            first_scores = report.datasets[name].llm_scores
            break

    if first_scores is None:
        return

    data_type = first_scores.get("type", "pretrain")

    # Build table
    table = Table(padding=(0, 1), show_edge=True)
    table.add_column("Metric", style="bold", min_width=25)
    for name in ds_names:
        table.add_column(name, justify="right", min_width=12)

    if data_type == "sft":
        _add_sft_score_rows(table, report, ds_names)
    else:
        _add_pretrain_score_rows(table, report, ds_names)

    # Scoring stats row
    table.add_section()
    stats_row: list[str] = ["Samples scored"]
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores:
            stats_row.append(str(scores.get("num_scored", 0)))
        else:
            stats_row.append("—")
    table.add_row(*stats_row, style="dim")

    errors_row: list[str] = ["Scoring errors"]
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores:
            errors_row.append(str(scores.get("scoring_errors", 0)))
        else:
            errors_row.append("—")
    table.add_row(*errors_row, style="dim")

    console.print(table)
    console.print()

    # Note: score distributions removed — LLM Binary Judge uses HIGH/LOW, not 1-6 scale


def _add_sft_score_rows(
    table: Table,
    report: BenchmarkReport,
    ds_names: list[str],
) -> None:
    """Add SFT LLM Binary Judge rows."""
    # HIGH rate
    row: list[str] = ["HIGH quality rate"]
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores and scores.get("num_scored", 0) > 0:
            row.append(f"{scores.get('high_rate', 0):.1%}")
        else:
            row.append("—")
    table.add_row(*row)

    # Per-rule failure counts
    all_rules: set[str] = set()
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores:
            all_rules.update(scores.get("rule_fail_counts", {}).keys())

    for rule in sorted(all_rules):
        row = [f"  ✗ {rule}"]
        for name in ds_names:
            scores = report.datasets[name].llm_scores
            if scores:
                count = scores.get("rule_fail_counts", {}).get(rule, 0)
                total = scores.get("num_scored", 0)
                row.append(f"{count}/{total}" if count > 0 else "—")
            else:
                row.append("—")
        table.add_row(*row, style="dim")


def _add_pretrain_score_rows(
    table: Table,
    report: BenchmarkReport,
    ds_names: list[str],
) -> None:
    """Add pre-training LLM Binary Judge rows."""
    # HIGH rate
    row: list[str] = ["HIGH quality rate"]
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores and scores.get("num_scored", 0) > 0:
            row.append(f"{scores.get('high_rate', 0):.1%}")
        else:
            row.append("—")
    table.add_row(*row)

    # Per-rule failure counts
    all_rules: set[str] = set()
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores:
            all_rules.update(scores.get("rule_fail_counts", {}).keys())

    for rule in sorted(all_rules):
        row = [f"  ✗ {rule}"]
        for name in ds_names:
            scores = report.datasets[name].llm_scores
            if scores:
                count = scores.get("rule_fail_counts", {}).get(rule, 0)
                total = scores.get("num_scored", 0)
                row.append(f"{count}/{total}" if count > 0 else "—")
            else:
                row.append("—")
        table.add_row(*row, style="dim")


def _print_score_distributions(
    report: BenchmarkReport,
    ds_names: list[str],
    data_type: str,
    console: Console,
) -> None:
    """Print score distribution tables."""
    if data_type == "sft":
        dist_keys = [
            ("complexity_distribution", "Complexity Score Distribution"),
            ("quality_distribution", "Quality Score Distribution"),
        ]
    else:
        dist_keys = [
            ("educational_distribution", "Educational Value Distribution"),
            ("writing_distribution", "Writing Quality Distribution"),
        ]

    for dist_key, title in dist_keys:
        table = Table(title=title, padding=(0, 1), show_edge=True)
        table.add_column("Score", style="bold", justify="center")
        for name in ds_names:
            table.add_column(name, justify="right", min_width=12)

        for score_val in range(1, 7):
            row: list[str] = [str(score_val)]
            for name in ds_names:
                scores = report.datasets[name].llm_scores
                if scores:
                    dist = scores.get(dist_key, {})
                    count = dist.get(score_val, dist.get(str(score_val), 0))
                    row.append(str(count))
                else:
                    row.append("—")
            table.add_row(*row)

        console.print(table)
        console.print()


def benchmark_to_json(report: BenchmarkReport, path: str | Path | None = None) -> str:
    """Serialize benchmark report to JSON."""
    discrimination = report.discrimination_scores()

    data: dict = {
        "config_path": report.config_path,
        "num_samples": report.num_samples,
        "llm_scoring_enabled": report.llm_scoring_enabled,
        "llm_samples": report.llm_samples,
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
            filter_data: dict = {
                "total": fr.total,
                "passed": fr.passed,
                "failed": fr.failed,
                "pass_rate": round(fr.pass_rate, 4),
                "sample_failed": fr.sample_failed[:3],
            }
            # Add per-rule breakdown if available
            if name in report.rule_stats and fname in report.rule_stats[name]:
                rules_data: dict = {}
                for rule_name, rs in report.rule_stats[name][fname].items():
                    rules_data[rule_name] = {
                        "total": rs.total,
                        "passed": rs.passed,
                        "failed": rs.failed,
                        "pass_rate": round(rs.pass_rate, 4),
                    }
                filter_data["rules"] = rules_data
            ds_data["per_filter"][fname] = filter_data
        # Also include flat pass rate dict for backward compat
        ds_data["per_filter_pass_rate"] = {
            k: round(v, 4) for k, v in dr.per_filter_pass_rate.items()
        }
        if dr.stats is not None:
            ds_data["pipeline_stats"] = dr.stats.to_dict()
        ds_data["data_type"] = dr.data_type
        if dr.llm_scores is not None:
            ds_data["llm_scores"] = dr.llm_scores
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

    # Per-filter rows with rule breakdown
    for f in all_filters:
        row = f"| **{f}** |"
        for name in ds_names:
            rate = report.datasets[name].per_filter_pass_rate.get(f, 0.0)
            row += f" {rate:.1%} |"
        delta = discrimination.get(f, 0.0)
        row += f" +{delta:.1%} | {_verdict_md(delta)} |"
        lines.append(row)

        # Add per-rule sub-rows
        if report.rule_stats:
            rule_names = _collect_rule_names(report, f, ds_names)
            for i, rule in enumerate(rule_names):
                is_last = i == len(rule_names) - 1
                prefix = "└─" if is_last else "├─"
                rule_row = f"|   {prefix} {rule} |"
                rule_rates = []
                for name in ds_names:
                    rs = report.rule_stats.get(name, {}).get(f, {}).get(rule)
                    rate = rs.pass_rate if rs else 1.0
                    rule_rates.append(rate)
                    rule_row += f" {rate:.1%} |"
                rule_delta = max(rule_rates) - min(rule_rates) if len(rule_rates) >= 2 else 0.0
                rule_row += f" +{rule_delta:.1%} | {_verdict_md(rule_delta)} |"
                lines.append(rule_row)

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

    # Layer 2: LLM Scoring section
    if report.llm_scoring_enabled:
        _append_llm_scores_markdown(lines, report, ds_names)

    md = "\n".join(lines)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(md, encoding="utf-8")
    return md


def _append_llm_scores_markdown(
    lines: list[str],
    report: BenchmarkReport,
    ds_names: list[str],
) -> None:
    """Append Layer 2 LLM scoring section to markdown lines."""
    has_scores = any(
        report.datasets[name].llm_scores is not None
        for name in ds_names
    )
    if not has_scores:
        return

    lines.append(f"## Layer 2: LLM Binary Judge ({report.llm_samples} samples)")
    lines.append("")

    # Determine data type
    first_scores = None
    for name in ds_names:
        if report.datasets[name].llm_scores:
            first_scores = report.datasets[name].llm_scores
            break

    if first_scores is None:
        return

    data_type = first_scores.get("type", "pretrain")

    # Build header
    header = "| Metric |"
    separator = "|--------|"
    for name in ds_names:
        header += f" {name} |"
        separator += "--------|"
    lines.append(header)
    lines.append(separator)

    # HIGH quality rate row
    row = "| HIGH quality rate |"
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores and scores.get("num_scored", 0) > 0:
            row += f" {scores.get('high_rate', 0):.1%} |"
        else:
            row += " — |"
    lines.append(row)

    # Per-rule failure counts
    all_rules: set[str] = set()
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores:
            all_rules.update(scores.get("rule_fail_counts", {}).keys())

    for rule in sorted(all_rules):
        row = f"| ✗ {rule} |"
        for name in ds_names:
            scores = report.datasets[name].llm_scores
            if scores:
                count = scores.get("rule_fail_counts", {}).get(rule, 0)
                total = scores.get("num_scored", 0)
                row += f" {count}/{total} |" if count > 0 else " — |"
            else:
                row += " — |"
        lines.append(row)

    # Stats rows
    row = "| Samples scored |"
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores:
            row += f" {scores.get('num_scored', 0)} |"
        else:
            row += " — |"
    lines.append(row)

    row = "| Scoring errors |"
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores:
            row += f" {scores.get('scoring_errors', 0)} |"
        else:
            row += " — |"
    lines.append(row)

    lines.append("")
