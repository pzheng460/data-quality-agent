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

# Thresholds for verdict classification (multi-dataset comparison only)
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


def _is_single(report: BenchmarkReport) -> bool:
    """Check if this is a single-dataset report."""
    return len(report.datasets) < 2


# ---------------------------------------------------------------------------
# Rich console output
# ---------------------------------------------------------------------------

def print_benchmark_report(report: BenchmarkReport, console: Console | None = None) -> None:
    """Print benchmark results to the console."""
    console = console or Console()
    if _is_single(report):
        _print_single_dataset(report, console)
    else:
        _print_comparison(report, console)


def _print_single_dataset(report: BenchmarkReport, console: Console) -> None:
    """Print quality report for a single dataset."""
    ds_name = list(report.datasets.keys())[0]
    result = report.datasets[ds_name]

    console.print()
    console.print(f"[bold]Data Quality Report: {ds_name}[/bold]", highlight=False)
    console.print(f"[dim]Samples: {result.num_docs} | Config: {report.config_path or 'default'} | Type: {result.data_type}[/dim]")
    console.print()

    # Dataset stats
    ds = result.dataset_stats
    if ds:
        table = Table(title="Dataset Statistics", padding=(0, 1), show_edge=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Avg word count", f"{ds.avg_word_count:.1f}")
        table.add_row("Min word count", str(ds.min_word_count))
        table.add_row("Max word count", str(ds.max_word_count))
        table.add_row("Avg word length", f"{ds.avg_word_length:.2f}")
        table.add_row("Fields", ", ".join(ds.fields) if ds.fields else "N/A")
        if ds.dedup:
            dup_style = "red" if ds.dedup.duplicate_rate > 0.05 else "dim"
            table.add_row("Exact duplicates", Text(f"{ds.dedup.exact_duplicates} ({ds.dedup.duplicate_rate:.1%})", style=dup_style))
        console.print(table)
        console.print()

    # Per-filter table
    table = Table(padding=(0, 1), show_edge=True)
    table.add_column("Filter", style="bold", min_width=25)
    table.add_column("Pass Rate", justify="right", min_width=10)
    table.add_column("Failed", justify="right", min_width=8)
    table.add_column("Total", justify="right", min_width=8)

    all_filters = list(result.per_filter_pass_rate.keys())

    for f in all_filters:
        fr = result.per_filter.get(f)
        rate = result.per_filter_pass_rate.get(f, 0.0)
        failed = fr.failed if fr else 0
        total = fr.total if fr else 0
        table.add_row(f, f"{rate:.1%}", str(failed), str(total))

        # Per-rule sub-rows
        if report.rule_stats:
            rule_names = _collect_rule_names(report, f, [ds_name])
            # Sort by failed count descending
            rule_stats_map = report.rule_stats.get(ds_name, {}).get(f, {})
            rule_names.sort(key=lambda r: rule_stats_map.get(r, _empty_rs()).failed, reverse=True)
            for i, rule in enumerate(rule_names):
                rs = rule_stats_map.get(rule)
                if not rs or rs.failed == 0:
                    continue
                is_last = i == len(rule_names) - 1 or all(
                    rule_stats_map.get(rule_names[j], _empty_rs()).failed == 0
                    for j in range(i + 1, len(rule_names))
                )
                prefix = "└─" if is_last else "├─"
                table.add_row(
                    Text(f"  {prefix} {rule}", style="dim"),
                    f"{rs.pass_rate:.1%}",
                    str(rs.failed),
                    str(rs.total),
                )

    # Overall row
    table.add_section()
    table.add_row(
        "Overall",
        f"{result.overall_pass_rate:.1%}",
        str(result.num_docs - int(result.overall_pass_rate * result.num_docs)),
        str(result.num_docs),
        style="bold",
    )

    console.print(table)
    console.print()

    # Sample failures
    _print_sample_failures_single(report, ds_name, all_filters, console)

    # Layer 2
    if report.llm_scoring_enabled:
        _print_llm_scores(report, [ds_name], console)


def _print_sample_failures_single(
    report: BenchmarkReport,
    ds_name: str,
    all_filters: list[str],
    console: Console,
) -> None:
    """Print sample failures for single dataset."""
    result = report.datasets[ds_name]
    has_samples = any(
        result.per_filter.get(f) and result.per_filter[f].sample_failed
        for f in all_filters
    )
    if not has_samples:
        return

    console.print("[bold]Sample Dropped Documents[/bold]")
    console.print()

    for f in all_filters:
        fr = result.per_filter.get(f)
        if not fr or not fr.sample_failed:
            continue
        console.print(f"  [bold]{f}[/bold] — failed {fr.failed}/{fr.total} docs:")
        for i, sample in enumerate(fr.sample_failed[:3]):
            preview = sample.get("text_preview", "")[:120].replace("\n", " ")
            reason = sample.get("reason", {})
            if isinstance(reason, dict):
                reason_str = reason.get("reason", "unknown")
                value = reason.get("value", "")
                if isinstance(value, float):
                    value = round(value, 4)
                reason_str = f"{reason_str} (value: {value})"
            else:
                reason_str = str(reason)
            console.print(f"    [{i+1}] [dim]{preview}...[/dim]")
            console.print(f"        Reason: {reason_str}")
        console.print()


def _print_comparison(report: BenchmarkReport, console: Console) -> None:
    """Print comparison table for multiple datasets."""
    ds_names = list(report.datasets.keys())

    all_filters: list[str] = []
    seen: set[str] = set()
    for dr in report.datasets.values():
        for f in dr.per_filter_pass_rate:
            if f not in seen:
                all_filters.append(f)
                seen.add(f)

    discrimination = report.discrimination_scores()

    console.print()
    console.print(
        f"[bold]Data Quality Benchmark: {' vs '.join(ds_names)}[/bold]",
        highlight=False,
    )
    console.print(f"[dim]Config: {report.config_path or 'default'} | "
                  f"Samples: {report.num_samples or 'all'}[/dim]")
    console.print()

    heuristic_filters = [f for f in all_filters if f != "sft_rules"]
    sft_rule_filters = [f for f in all_filters if f == "sft_rules"]

    table = Table(padding=(0, 1), show_edge=True)
    table.add_column("Filter", style="bold", min_width=20)
    for name in ds_names:
        table.add_column(name, justify="right", min_width=12)
    table.add_column("Δ", justify="right", min_width=10)
    table.add_column("Verdict", min_width=18)

    if heuristic_filters:
        num_cols = len(ds_names) + 3
        table.add_row(*([Text("Layer 1: Pre-training Rules", style="bold cyan")] + [""] * (num_cols - 1)))

    for f in heuristic_filters:
        _add_comparison_filter_row(table, report, f, ds_names, discrimination)

    if sft_rule_filters:
        table.add_section()
        num_cols = len(ds_names) + 3
        table.add_row(*([Text("Layer 1: SFT Rules", style="bold cyan")] + [""] * (num_cols - 1)))
        for f in sft_rule_filters:
            _add_comparison_filter_row(table, report, f, ds_names, discrimination)

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

    console.print("[green]✅ = Passes significantly more (>5%) → filter catches real issues[/green]")
    console.print("[yellow]⚠️ = Small difference (2-5%) → weak signal[/yellow]")
    console.print("[dim]— = No meaningful difference (<2%)[/dim]")
    console.print()

    _print_sample_failures_comparison(report, ds_names, all_filters, discrimination, console)

    if report.llm_scoring_enabled:
        _print_llm_scores(report, ds_names, console)


def _add_comparison_filter_row(
    table: Table,
    report: BenchmarkReport,
    f: str,
    ds_names: list[str],
    discrimination: dict[str, float],
) -> None:
    """Add a filter row with per-rule sub-rows for comparison mode."""
    row: list[str | Text] = [Text(f, style="bold")]
    for name in ds_names:
        rate = report.datasets[name].per_filter_pass_rate.get(f, 0.0)
        row.append(f"{rate:.1%}")

    delta = discrimination.get(f, 0.0)
    row.append(f"+{delta:.1%}")
    label, style = _verdict(delta)
    row.append(Text(label, style=style))
    table.add_row(*row)

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


def _print_sample_failures_comparison(
    report: BenchmarkReport,
    ds_names: list[str],
    all_filters: list[str],
    discrimination: dict[str, float],
    console: Console,
) -> None:
    """Print sample failures for comparison mode."""
    if len(ds_names) < 2:
        return

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
    has_scores = any(
        report.datasets[name].llm_scores is not None
        for name in ds_names
    )
    if not has_scores:
        return

    console.print()
    console.print(
        f"[bold]Layer 2: LLM Binary Judge ({report.llm_samples} samples)[/bold]",
        highlight=False,
    )
    console.print()

    first_scores = None
    for name in ds_names:
        if report.datasets[name].llm_scores:
            first_scores = report.datasets[name].llm_scores
            break
    if first_scores is None:
        return

    table = Table(padding=(0, 1), show_edge=True)
    table.add_column("Metric", style="bold", min_width=25)
    for name in ds_names:
        table.add_column(name, justify="right", min_width=12)

    # HIGH rate
    row: list[str] = ["HIGH quality rate"]
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores and scores.get("num_scored", 0) > 0:
            row.append(f"{scores.get('high_rate', 0):.1%}")
        else:
            row.append("—")
    table.add_row(*row)

    # Per-rule failures
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

    table.add_section()
    stats_row: list[str] = ["Samples scored"]
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        stats_row.append(str(scores.get("num_scored", 0)) if scores else "—")
    table.add_row(*stats_row, style="dim")

    console.print(table)
    console.print()


def _empty_rs():
    """Return a placeholder RuleStats-like object with failed=0."""
    class _Fake:
        failed = 0
        pass_rate = 1.0
    return _Fake()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def benchmark_to_json(report: BenchmarkReport, path: str | Path | None = None) -> str:
    """Serialize benchmark report to JSON."""
    discrimination = report.discrimination_scores()

    data: dict = {
        "config_path": report.config_path,
        "num_samples": report.num_samples,
        "llm_scoring_enabled": report.llm_scoring_enabled,
        "llm_samples": report.llm_samples,
        "datasets": {},
    }

    if discrimination:
        data["discrimination"] = {}
        for f, delta in discrimination.items():
            data["discrimination"][f] = {
                "delta": round(delta, 4),
                "verdict": _verdict_md(delta),
            }

    for name, dr in report.datasets.items():
        ds_data: dict = {
            "num_docs": dr.num_docs,
            "overall_pass_rate": round(dr.overall_pass_rate, 4),
            "data_type": dr.data_type,
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

        ds_data["per_filter_pass_rate"] = {
            k: round(v, 4) for k, v in dr.per_filter_pass_rate.items()
        }
        if dr.dataset_stats:
            s = dr.dataset_stats
            stats_dict: dict = {
                "avg_word_count": round(s.avg_word_count, 1),
                "min_word_count": s.min_word_count,
                "max_word_count": s.max_word_count,
                "avg_word_length": round(s.avg_word_length, 2),
                "fields": s.fields,
            }
            if s.dedup:
                stats_dict["exact_duplicates"] = s.dedup.exact_duplicates
                stats_dict["duplicate_rate"] = round(s.dedup.duplicate_rate, 4)
            ds_data["dataset_stats"] = stats_dict
        if dr.llm_scores is not None:
            ds_data["llm_scores"] = dr.llm_scores
        data["datasets"][name] = ds_data

    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json_str, encoding="utf-8")
    return json_str


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def benchmark_to_markdown(report: BenchmarkReport, path: str | Path | None = None) -> str:
    """Generate Markdown report."""
    if _is_single(report):
        md = _markdown_single(report)
    else:
        md = _markdown_comparison(report)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(md, encoding="utf-8")
    return md


def _format_reason(reason) -> str:
    """Format a drop reason dict into a readable string."""
    if isinstance(reason, dict):
        reason_str = reason.get("reason", "unknown")
        value = reason.get("value", "")
        if isinstance(value, float):
            value = round(value, 4)
        return f"`{reason_str}` (value: {value})"
    return f"`{reason}`"


def _markdown_single(report: BenchmarkReport) -> str:
    """Generate Markdown for single dataset quality report."""
    ds_name = list(report.datasets.keys())[0]
    result = report.datasets[ds_name]
    all_filters = list(result.per_filter_pass_rate.keys())

    lines: list[str] = []
    lines.append(f"# Data Quality Report: {ds_name}")
    lines.append("")
    lines.append(f"**Samples**: {result.num_docs}")
    if report.config_path:
        lines.append(f"**Config**: `{report.config_path}`")
    lines.append("")

    # Dataset stats
    ds = result.dataset_stats
    if ds:
        lines.append("## Dataset Statistics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total documents | {result.num_docs} |")
        lines.append(f"| Data type | {result.data_type} |")
        lines.append(f"| Avg word count | {ds.avg_word_count:.1f} |")
        lines.append(f"| Min word count | {ds.min_word_count} |")
        lines.append(f"| Max word count | {ds.max_word_count} |")
        lines.append(f"| Avg word length | {ds.avg_word_length:.2f} |")
        lines.append(f"| Fields | {', '.join(ds.fields) if ds.fields else 'N/A'} |")
        if ds.dedup:
            lines.append(f"| Exact duplicates | {ds.dedup.exact_duplicates} ({ds.dedup.duplicate_rate:.1%}) |")
        lines.append("")

    # Overall stats
    lines.append("## Quality Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total documents | {result.num_docs} |")
    total_passed = int(result.overall_pass_rate * result.num_docs)
    total_failed = result.num_docs - total_passed
    lines.append(f"| Passed | {total_passed} |")
    lines.append(f"| Failed | {total_failed} |")
    lines.append(f"| Overall pass rate | {result.overall_pass_rate:.1%} |")
    lines.append("")

    # Per-filter table
    lines.append("## Per-Filter Statistics")
    lines.append("")
    lines.append("| Filter | Pass Rate | Failed | Total |")
    lines.append("|--------|-----------|--------|-------|")
    for f in all_filters:
        fr = result.per_filter.get(f)
        rate = result.per_filter_pass_rate.get(f, 0.0)
        failed = fr.failed if fr else 0
        total = fr.total if fr else 0
        lines.append(f"| **{f}** | {rate:.1%} | {failed} | {total} |")
    lines.append("")

    # Per-rule breakdown
    if report.rule_stats and ds_name in report.rule_stats:
        lines.append("## Per-Rule Breakdown")
        lines.append("")
        for f in all_filters:
            rules = report.rule_stats.get(ds_name, {}).get(f, {})
            if not rules:
                continue
            # Filter to only rules with failures, sorted by failed count
            failed_rules = [(r, rs) for r, rs in rules.items() if rs.failed > 0]
            if not failed_rules:
                continue
            failed_rules.sort(key=lambda x: x[1].failed, reverse=True)

            lines.append(f"### {f}")
            lines.append("")
            lines.append("| Rule | Failed | Total | Fail Rate |")
            lines.append("|------|--------|-------|-----------|")
            for rule_name, rs in failed_rules:
                fail_rate = rs.failed / rs.total if rs.total > 0 else 0.0
                lines.append(f"| {rule_name} | {rs.failed} | {rs.total} | {fail_rate:.2%} |")
            lines.append("")

    # Sample failures
    has_samples = any(
        result.per_filter.get(f) and result.per_filter[f].sample_failed
        for f in all_filters
    )
    if has_samples:
        lines.append("## Sample Dropped Documents")
        lines.append("")
        for f in all_filters:
            fr = result.per_filter.get(f)
            if not fr or not fr.sample_failed:
                continue
            lines.append(f"### {f}")
            lines.append("")
            for i, sample in enumerate(fr.sample_failed[:3], 1):
                reason = sample.get("reason", {})
                preview = sample.get("text_preview", "")[:100].replace("\n", " ")
                lines.append(f"**Sample {i}**: {_format_reason(reason)}")
                lines.append(f"> {preview}...")
                lines.append("")

    # Layer 2
    if report.llm_scoring_enabled:
        _append_llm_scores_markdown(lines, report, [ds_name])

    return "\n".join(lines)


def _markdown_comparison(report: BenchmarkReport) -> str:
    """Generate Markdown comparison table for multiple datasets."""
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

    for f in all_filters:
        row = f"| **{f}** |"
        for name in ds_names:
            rate = report.datasets[name].per_filter_pass_rate.get(f, 0.0)
            row += f" {rate:.1%} |"
        delta = discrimination.get(f, 0.0)
        row += f" +{delta:.1%} | {_verdict_md(delta)} |"
        lines.append(row)

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
    lines.append("- ✅ = Passes significantly more (>5%) → filter catches real issues")
    lines.append("- ⚠️ = Small difference (2-5%) → weak signal")
    lines.append("- — = No meaningful difference (<2%)")
    lines.append("")

    # Sample failures
    if len(ds_names) >= 2:
        orig_name = ds_names[0]
        orig_result = report.datasets[orig_name]
        discriminating = [f for f in all_filters if discrimination.get(f, 0.0) >= WEAK_THRESHOLD]

        if discriminating:
            lines.append(f"### Sample failures from {orig_name}")
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
                    lines.append(f"{i+1}. `{preview}...`")
                    lines.append(f"   - Reason: {_format_reason(reason)}")
                lines.append("")

    if report.llm_scoring_enabled:
        _append_llm_scores_markdown(lines, report, ds_names)

    return "\n".join(lines)


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

    header = "| Metric |"
    separator = "|--------|"
    for name in ds_names:
        header += f" {name} |"
        separator += "--------|"
    lines.append(header)
    lines.append(separator)

    row = "| HIGH quality rate |"
    for name in ds_names:
        scores = report.datasets[name].llm_scores
        if scores and scores.get("num_scored", 0) > 0:
            row += f" {scores.get('high_rate', 0):.1%} |"
        else:
            row += " — |"
    lines.append(row)

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

    lines.append("")
