"""CLI entry point for the data quality agent."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from dq.config import PipelineConfig
from dq.utils.io import read_docs, write_docs
from dq.utils.stats import avg_word_length, word_count

console = Console()


def _load_config(config_path: str | None) -> PipelineConfig:
    """Load config from file or return default."""
    if config_path:
        return PipelineConfig.from_yaml(config_path)
    return PipelineConfig.default()


@click.group()
@click.version_option(package_name="dq")
def main():
    """dq — Training data quality detection agent."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), help="Pipeline config YAML")
@click.option("-o", "--output", "output_path", required=True, type=click.Path(), help="Output file path")
@click.option("--report-dir", type=click.Path(), help="Directory for report files")
@click.option("--progress/--no-progress", default=True, help="Show progress bar")
def run(input_path: str, config_path: str | None, output_path: str, report_dir: str | None, progress: bool):
    """Run full quality pipeline on input data."""
    # Import here to trigger filter registration
    import dq.filters  # noqa: F401
    from dq.dedup import ExactDedup, MinHashDedup
    from dq.pipeline import Pipeline
    from dq.report import generate_report

    config = _load_config(config_path)
    pipeline = Pipeline(config)

    console.print(f"[bold]Running pipeline on[/bold] {input_path}")
    console.print(f"[dim]Filters: {[f.name for f in pipeline.filters]}[/dim]")

    # Phase 1: filter
    docs = list(read_docs(input_path))
    filtered = list(pipeline.run(iter(docs), progress=progress))

    # Phase 2: dedup
    dedup_stats = {}
    if config.dedup.exact:
        console.print("[bold]Running exact dedup...[/bold]")
        exact = ExactDedup(text_field=config.text_field)
        filtered = list(exact.dedup(filtered))
        dedup_stats["exact"] = exact.stats()

    mh_config = config.dedup.minhash
    if mh_config.get("enabled", False):
        console.print("[bold]Running MinHash LSH dedup...[/bold]")
        mh = MinHashDedup(
            text_field=config.text_field,
            num_perm=mh_config.get("num_perm", 112),
            bands=mh_config.get("bands", 14),
            rows=mh_config.get("rows", 8),
            ngram_size=mh_config.get("ngram_size", 5),
        )
        filtered = list(mh.dedup(filtered))
        dedup_stats["minhash"] = mh.stats()

    # Write output
    count = write_docs(iter(filtered), output_path)
    console.print(f"[green]Wrote {count} documents to {output_path}[/green]")

    # Print summary
    _print_stats_table(pipeline.stats)

    # Generate report
    if report_dir:
        generate_report(pipeline.stats, dedup_stats, report_dir)
        console.print(f"[green]Report saved to {report_dir}/[/green]")


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--text-field", default="text", help="Field name containing text")
def stats(input_path: str, text_field: str):
    """Show dataset statistics without filtering."""
    docs = list(read_docs(input_path))

    total = len(docs)
    if total == 0:
        console.print("[yellow]No documents found.[/yellow]")
        return

    word_counts = [word_count(d.get(text_field, "")) for d in docs]
    avg_words = sum(word_counts) / len(word_counts)
    avg_wl = sum(avg_word_length(d.get(text_field, "")) for d in docs) / total

    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Total documents", str(total))
    table.add_row("Avg word count", f"{avg_words:.1f}")
    table.add_row("Min word count", str(min(word_counts)))
    table.add_row("Max word count", str(max(word_counts)))
    table.add_row("Avg word length", f"{avg_wl:.2f}")
    table.add_row("Fields", ", ".join(docs[0].keys()) if docs else "N/A")

    console.print(table)


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), help="Pipeline config YAML")
@click.option("--report-dir", type=click.Path(), help="Directory for report files")
@click.option("--progress/--no-progress", default=True, help="Show progress bar")
def report(input_path: str, config_path: str | None, report_dir: str | None, progress: bool):
    """Dry-run: generate quality report without writing output."""
    import dq.filters  # noqa: F401
    from dq.pipeline import Pipeline
    from dq.report import generate_report, stats_to_markdown

    config = _load_config(config_path)
    pipeline = Pipeline(config)

    console.print(f"[bold]Dry-run report on[/bold] {input_path}")

    docs = read_docs(input_path)
    pipeline_stats = pipeline.dry_run(docs, progress=progress)

    _print_stats_table(pipeline_stats)

    if report_dir:
        generate_report(pipeline_stats, output_dir=report_dir)
        console.print(f"[green]Report saved to {report_dir}/[/green]")
    else:
        console.print(stats_to_markdown(pipeline_stats))


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", required=True, type=click.Path(), help="Output file path")
@click.option("--method", type=click.Choice(["exact", "minhash", "both"]), default="both", help="Dedup method")
@click.option("--text-field", default="text", help="Field name containing text")
def dedup(input_path: str, output_path: str, method: str, text_field: str):
    """Run deduplication only."""
    from dq.dedup import ExactDedup, MinHashDedup

    docs = list(read_docs(input_path))
    console.print(f"[bold]Deduplicating {len(docs)} documents...[/bold]")

    result = docs

    if method in ("exact", "both"):
        exact = ExactDedup(text_field=text_field)
        result = list(exact.dedup(result))
        console.print(f"  Exact dedup: {exact.stats()['duplicate_docs']} duplicates removed")

    if method in ("minhash", "both"):
        mh = MinHashDedup(text_field=text_field)
        result = list(mh.dedup(result))
        console.print(f"  MinHash dedup: {mh.stats()['duplicate_docs']} near-duplicates removed")

    count = write_docs(iter(result), output_path)
    console.print(f"[green]Wrote {count} documents to {output_path}[/green]")


@main.command()
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), help="Pipeline config YAML")
@click.option("-n", "--num-samples", default=0, type=int, help="Samples per dataset (0 = all)")
@click.option("--no-dedup", is_flag=True, default=True, help="Skip dedup (default: skip)")
@click.option("--dedup", "no_dedup", flag_value=False, help="Enable dedup in benchmark")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
@click.option("-o", "--output", "output_dir", type=click.Path(), help="Directory to save reports")
@click.option("--report-dir", type=click.Path(), help="Alias for -o (deprecated)")
def bench(config_path: str | None, num_samples: int, no_dedup: bool, seed: int,
          output_dir: str | None, report_dir: str | None):
    """Run benchmark: compare Alpaca original vs cleaned filter pass rates."""
    from dq.benchmark import run_benchmark
    from dq.benchmark_report import (
        benchmark_to_json,
        benchmark_to_markdown,
        print_benchmark_report,
    )

    # -o takes precedence over --report-dir
    save_dir = output_dir or report_dir

    n = num_samples if num_samples > 0 else None
    samples_label = str(num_samples) if num_samples > 0 else "all"
    console.print(f"[bold]Running benchmark with {samples_label} samples per dataset...[/bold]")
    console.print(f"[dim]Config: {config_path or 'default'} | Dedup: {'off' if no_dedup else 'on'} | Seed: {seed}[/dim]")

    try:
        report = run_benchmark(
            config_path=config_path,
            n=n,
            no_dedup=no_dedup,
            seed=seed,
        )
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        raise SystemExit(1)

    print_benchmark_report(report, console=console)

    if save_dir:
        benchmark_to_json(report, path=Path(save_dir) / "benchmark.json")
        benchmark_to_markdown(report, path=Path(save_dir) / "benchmark.md")
        console.print(f"[green]Reports saved to {save_dir}/[/green]")


def _print_stats_table(stats) -> None:
    """Print pipeline stats as a rich table."""
    table = Table(title="Pipeline Results")
    table.add_column("Filter", style="bold")
    table.add_column("Docs In", justify="right")
    table.add_column("Docs Out", justify="right")
    table.add_column("Dropped", justify="right")
    table.add_column("Drop Rate", justify="right")

    for fs in stats.filter_stats:
        table.add_row(
            fs.name,
            str(fs.docs_in),
            str(fs.docs_out),
            str(fs.docs_dropped),
            f"{fs.drop_rate:.2%}",
        )

    table.add_row(
        "[bold]TOTAL[/bold]",
        str(stats.total_in),
        str(stats.total_out),
        str(stats.total_dropped),
        f"{stats.overall_drop_rate:.2%}",
        style="bold",
    )
    console.print(table)
