"""CLI entry point for the data quality agent."""

from pathlib import Path

import click
from rich.console import Console

from dq.config import PipelineConfig
from dq.utils.io import read_docs, sample_docs

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


def _load_input_datasets(
    input_path: str,
    num_samples: int,
    seed: int,
    split: str,
    text_field: str,
    hf_config: str | None,
) -> dict[str, list[dict]]:
    """Load dataset from input_path (local file or HF dataset).

    Returns:
        Dict of {dataset_name: docs}.
    """
    local_path = Path(input_path)
    is_hf = "/" in input_path and not local_path.exists()

    if is_hf:
        from dq.benchmark.datasets import load_hf_dataset
        n = num_samples if num_samples > 0 else 1000
        console.print(f"[bold]Loading HuggingFace dataset[/bold] {input_path} ({n} samples, streaming)")
        docs = load_hf_dataset(
            input_path, n=n, split=split, text_field=text_field, seed=seed, config=hf_config,
        )
        console.print(f"[dim]Loaded {len(docs)} samples[/dim]")
        ds_name = input_path.split("/")[-1]
        return {ds_name: docs}
    else:
        if not local_path.exists():
            raise click.BadParameter(f"File not found: {input_path}", param_hint="INPUT")
        if num_samples > 0:
            console.print(f"[dim]Sampling {num_samples} docs (seed={seed})...[/dim]")
            docs = sample_docs(input_path, n=num_samples, seed=seed)
        else:
            docs = list(read_docs(input_path))
        ds_name = local_path.stem
        return {ds_name: docs}


@main.command()
@click.argument("input_path")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), help="Pipeline config YAML")
@click.option("-n", "--num-samples", default=0, type=int, help="Samples per dataset (0 = all, default 1000 for HF)")
@click.option("--no-dedup", is_flag=True, default=True, help="Skip dedup (default: skip)")
@click.option("--dedup", "no_dedup", flag_value=False, help="Enable dedup in benchmark")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
@click.option("-o", "--output", "output_dir", type=click.Path(), default="reports", help="Directory to save reports (default: reports/)")
@click.option("--with-model-filters", is_flag=True, default=False, help="Enable model-based filters")
@click.option("--check-contamination", default=None, help="Benchmark contamination check (e.g. 'mmlu,hellaswag' or 'all')")
@click.option("--with-llm-scoring", is_flag=True, default=False, help="Enable Layer 2 LLM binary judge")
@click.option("--llm-samples", default=50, type=int, help="Docs to score per dataset for LLM scoring (default 50)")
@click.option("--data-type", type=click.Choice(["sft", "pretrain", "auto"]), default="auto",
              help="Data type (auto-detect by default)")
@click.option("--api-url", envvar="DQ_API_BASE_URL", default=None, help="LLM API base URL")
@click.option("--api-key", envvar="DQ_API_KEY", default=None, help="LLM API key")
@click.option("--model", "llm_model", envvar="DQ_MODEL", default=None, help="LLM model name for scoring")
@click.option("--split", default="train", help="HF dataset split (default: train)")
@click.option("--text-field", default="text", help="Field name containing text (default: text)")
@click.option("--hf-config", default=None, help="HF dataset config/subset name")
@click.option("-w", "--workers", default=None, type=int, help="Parallel workers (default: auto)")
@click.option("--save-rejected", "save_rejected_path", default=None, type=click.Path(),
              help="Save all rejected docs with reasons to a JSONL file")
def bench(input_path: str, config_path: str | None, num_samples: int, no_dedup: bool, seed: int,
          output_dir: str, with_model_filters: bool,
          check_contamination: str | None, with_llm_scoring: bool, llm_samples: int,
          data_type: str, api_url: str | None, api_key: str | None, llm_model: str | None,
          split: str, text_field: str, hf_config: str | None, workers: int | None,
          save_rejected_path: str | None):
    """Run quality benchmark and generate report.

    INPUT_PATH can be a local file (jsonl/csv/parquet) or a HuggingFace dataset ID
    (e.g. 'allenai/dolma3_mix-6T'). HF datasets are loaded via streaming — only the
    requested samples are downloaded.

    \b
    Examples:
      dq bench data.jsonl -n 1000                              # Local file
      dq bench allenai/dolma3_mix-6T -n 10000                  # HuggingFace dataset
      dq bench data.jsonl --with-llm-scoring                   # + Layer 2 LLM judge
      dq bench data.jsonl --check-contamination mmlu,hellaswag # + contamination check
      dq bench data.jsonl --check-contamination all            # all benchmarks
    """
    if with_model_filters:
        import dq.model_filters  # noqa: F401
    from dq.benchmark import run_benchmark
    from dq.benchmark_report import (
        benchmark_to_json,
        benchmark_to_markdown,
        print_benchmark_report,
    )

    n = num_samples if num_samples > 0 else None
    samples_label = str(num_samples) if num_samples > 0 else "all"
    console.print(f"[bold]Running benchmark on[/bold] {input_path} [dim]({samples_label} samples)[/dim]")
    console.print(f"[dim]Config: {config_path or 'default'} | Dedup: {'off' if no_dedup else 'on'} | Seed: {seed}[/dim]")
    if with_llm_scoring:
        console.print(f"[dim]LLM scoring: enabled ({llm_samples} samples/dataset, type={data_type})[/dim]")

    try:
        # Load config for LLM settings
        pipeline_config = _load_config(config_path)

        # Apply YAML LLM config to shared client (CLI args override)
        from dq.llm_client import set_config_from_yaml
        set_config_from_yaml(pipeline_config.llm)

        datasets = _load_input_datasets(
            input_path, num_samples, seed, split, text_field, hf_config,
        )
        report = run_benchmark(
            config_path=config_path,
            datasets=datasets,
            n=n,
            no_dedup=no_dedup,
            seed=seed,
            data_type=data_type,
            workers=workers,
            save_rejected=bool(save_rejected_path),
        )
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        raise SystemExit(1)

    # Layer 2: LLM scoring
    # Use YAML llm.samples as default, CLI --llm-samples overrides
    effective_llm_samples = llm_samples if llm_samples != 50 else pipeline_config.llm.samples
    if with_llm_scoring:
        from dq.benchmark import run_llm_scoring

        console.print()
        console.print("[bold]Running Layer 2: LLM Quality Scoring...[/bold]")
        data_type_override = data_type if data_type != "auto" else None
        try:
            run_llm_scoring(
                report=report,
                datasets=datasets,
                llm_samples=effective_llm_samples,
                data_type_override=data_type_override,
                seed=seed,
                api_url=api_url,        # CLI override (or None → falls back to YAML/env)
                api_key=api_key,
                model=llm_model,
            )
        except Exception as e:
            console.print(f"[yellow]LLM scoring failed: {e}[/yellow]")

    print_benchmark_report(report, console=console)

    # Optional contamination check
    if check_contamination:
        from dq.stages.curation.contamination.ngram import NgramContaminationDetector, load_benchmark as load_bm, _BENCHMARK_CONFIGS

        # Resolve benchmark list
        if check_contamination.lower() == "all":
            bm_names = list(_BENCHMARK_CONFIGS.keys())
        else:
            bm_names = [b.strip() for b in check_contamination.split(",") if b.strip()]

        console.print(f"[bold]Running contamination check (n-gram) against {', '.join(bm_names)}...[/bold]")
        bm_texts: dict[str, list[str]] = {}
        for bm_name in bm_names:
            try:
                bm_texts[bm_name] = load_bm(bm_name)
                console.print(f"  Loaded benchmark '{bm_name}': {len(bm_texts[bm_name])} texts")
            except (ImportError, Exception) as e:
                console.print(f"  [yellow]Skipping '{bm_name}': {e}[/yellow]")

        if bm_texts:
            detector = NgramContaminationDetector(n=13, threshold=0.8)
            for ds_name, ds_docs in datasets.items():
                try:
                    cr = detector.scan_dataset(ds_docs, benchmarks=bm_texts, dataset_name=ds_name)
                    cr.print_rich(console=console)
                    # Save contamination report alongside benchmark report
                    cr.to_json(str(Path(output_dir) / "contamination.json"))
                    cr.to_markdown(str(Path(output_dir) / "contamination.md"))
                except Exception as e:
                    console.print(f"  [yellow]Contamination check failed for {ds_name}: {e}[/yellow]")

    # Save reports
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    benchmark_to_json(report, path=Path(output_dir) / "benchmark.json")
    benchmark_to_markdown(report, path=Path(output_dir) / "benchmark.md")
    console.print(f"[green]Reports saved to {output_dir}/[/green]")

    # Save rejected docs to JSONL
    if save_rejected_path and report.rejected_docs:
        import json
        rejected_path = Path(save_rejected_path)
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        total_rejected = 0
        with open(rejected_path, "w", encoding="utf-8") as f:
            for ds_name, rejected_list in report.rejected_docs.items():
                for doc in rejected_list:
                    doc["__dq_dataset"] = ds_name
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    total_rejected += 1
        console.print(f"[green]Rejected docs saved to {save_rejected_path} ({total_rejected} docs)[/green]")


@main.command()
@click.argument("input_path")
@click.option("-o", "--output-dir", required=True, type=click.Path(), help="Output base directory")
@click.option("-c", "--config", "config_path", required=True, type=click.Path(exists=True),
              help="Pipeline config YAML (e.g. configs/arxiv.yaml)")
@click.option("--stage", default=None, type=int, help="Run specific stage (1=ingest, 2=extract, 3=curate, 4=package)")
@click.option("--phase", default=None, type=int, hidden=True, help="Deprecated alias for --stage")
@click.option("--resume/--no-resume", default=True, help="Resume from last completed phase")
@click.option("-w", "--workers", default=None, type=int, help="Parallel workers (default: auto)")
@click.option("-n", "--num-docs", default=0, type=int, help="Limit input docs (0=all, for testing)")
@click.option("--dry-run", is_flag=True, default=False, help="Show plan without executing")
def run(input_path: str, output_dir: str, config_path: str, stage: int | None,
        phase: int | None, resume: bool, workers: int | None, num_docs: int, dry_run: bool):
    """Run production data cleaning pipeline.

    INPUT_PATH can be a directory of shards, a single JSONL file, or a .jsonl.zst file.

    \b
    Examples:
      dq run /data/raw -o /data/arxiv -c configs/arxiv.yaml
      dq run /data/raw -o /data/arxiv -c configs/arxiv.yaml --stage 3
      dq run /data/raw -o /data/arxiv -c configs/arxiv.yaml --no-resume
      dq run /data/raw -o /data/arxiv -c configs/arxiv.yaml -n 1000 --dry-run
    """
    from dq.runner.engine import PhaseEngine

    engine = PhaseEngine(
        config_path=config_path,
        input_path=input_path,
        output_dir=output_dir,
        workers=workers,
        num_samples=num_docs,
    )

    if dry_run:
        engine.show_plan()
        return

    target = stage or phase  # --phase is deprecated alias for --stage
    if target is not None:
        engine.run_stage(target)
    else:
        engine.run_all(resume=resume)
