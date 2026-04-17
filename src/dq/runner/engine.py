"""PhaseEngine — orchestrates the 4-stage production pipeline.

Stages:
  1. ingestion  — fetch raw data
  2. extraction — convert raw format to text
  3. curation   — filter + dedup + contamination
  4. packaging  — sort, shard, manifest
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterator

import yaml
from rich.console import Console

from dq.config import PipelineConfig

logger = logging.getLogger(__name__)
console = Console()

_STAGE_NAMES = {
    1: "ingestion",
    2: "extraction",
    3: "curation",
    4: "packaging",
}

# Backward compat: old phase numbers → new stage numbers
_PHASE_TO_STAGE = {1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4}


class PhaseEngine:
    """Orchestrates the 4-stage pipeline with _SUCCESS markers for resumability."""

    def __init__(
        self,
        config_path: str,
        input_path: str,
        output_dir: str,
        workers: int | None = None,
        num_samples: int = 0,
    ) -> None:
        self.config_path = config_path

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self.config = PipelineConfig.from_dict(raw, config_dir=Path(config_path).parent)

        # Extra config sections (arxiv-specific, quality scoring, etc.)
        self.extra_config = {k: v for k, v in raw.items() if k != "pipeline"}
        # Legacy alias
        self.arxiv_config = raw.get("arxiv", {})

        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples

        self.version = self.extra_config.get("arxiv", {}).get("version", "unknown")

        # Workers
        parallelism = self.extra_config.get("arxiv", {}).get("parallelism", {})
        if workers is not None:
            self.workers = workers
        elif parallelism.get("num_workers"):
            self.workers = parallelism["num_workers"]
        else:
            self.workers = max(1, min((os.cpu_count() or 1) // 4, 32))

        # Compute backend — default LocalBackend, can be overridden from config
        from dq.runner.backend import make_backend
        compute_cfg = self.extra_config.get("compute") or {}
        backend_name = compute_cfg.get("backend", "local")
        backend_kwargs = {k: v for k, v in compute_cfg.items() if k != "backend"}
        if backend_name == "local" and "cpu_workers" not in backend_kwargs:
            backend_kwargs["cpu_workers"] = self.workers
        self.backend = make_backend(backend_name, **backend_kwargs)
        logger.info("Compute backend: %s (cpu workers=%s)", backend_name, self.workers)

        # Shard target
        self.shard_target_bytes = self.extra_config.get("arxiv", {}).get(
            "phase5", {}
        ).get("shard_target_bytes", 1_073_741_824)

        # Config hash
        with open(config_path, "rb") as f:
            self._config_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        return self._config_hash

    def stage_dir(self, stage_name: str, sub: str | None = None) -> Path:
        base = self.output_dir / stage_name
        return base / sub if sub else base

    def _success_path(self, stage_name: str) -> Path:
        return self.output_dir / f".{stage_name}_SUCCESS"

    def is_stage_done(self, stage_name: str) -> bool:
        return self._success_path(stage_name).exists()

    def mark_stage_done(self, stage_name: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._success_path(stage_name).touch()

    # Backward compat aliases
    def is_phase_done(self, name: str) -> bool:
        return self.is_stage_done(name)

    def mark_phase_done(self, name: str) -> None:
        self.mark_stage_done(name)

    def iter_input(self) -> Iterator[dict]:
        from dq.shared.shard import read_shards
        from dq.utils.io import read_docs

        source = self.input_path
        if source.is_dir():
            gen = read_shards(source)
        else:
            gen = read_docs(source)

        count = 0
        for doc in gen:
            yield doc
            count += 1
            if self.num_samples > 0 and count >= self.num_samples:
                break

    def run_all(self, resume: bool = True) -> None:
        """Run all 4 stages."""
        from dq.runner.stages import stage_ingest, stage_extract, stage_curate, stage_package
        from dq.shared.stats import save_overview

        self.output_dir.mkdir(parents=True, exist_ok=True)

        stage_list = [
            ("ingestion", stage_ingest),
            ("extraction", stage_extract),
            ("curation", stage_curate),
            ("packaging", stage_package),
        ]

        all_stats = []
        for name, func in stage_list:
            if resume and self.is_stage_done(name):
                console.print(f"[dim]  Skip {name} (done)[/dim]")
                continue

            console.print(f"[bold]  {name}...[/bold]")
            stats = func(self)
            self.mark_stage_done(name)
            all_stats.append(stats)

            stats_dir = self.output_dir / "stats" / self.version
            stats_dir.mkdir(parents=True, exist_ok=True)
            stats.save(stats_dir / f"{name}.json")

            console.print(
                f"    {stats.input_count} in → {stats.output_count} kept, "
                f"{stats.rejected_count} rejected ({stats.duration_seconds:.1f}s)"
            )

        if all_stats:
            stats_dir = self.output_dir / "stats" / self.version
            save_overview(stats_dir, all_stats, self.version, config_hash=self._config_hash)

        # ── Auto-benchmark final output ──
        final_dir = self.output_dir / "stage4_final"
        if final_dir.exists():
            try:
                self._run_bench(final_dir)
            except Exception as e:
                logger.warning("Auto-benchmark failed: %s", e)

    def run_stage(self, stage_num: int) -> None:
        """Run a single stage by number (1-4)."""
        from dq.runner import stages as sm

        stage_map = {
            1: ("ingestion", sm.stage_ingest),
            2: ("extraction", sm.stage_extract),
            3: ("curation", sm.stage_curate),
            4: ("packaging", sm.stage_package),
        }
        if stage_num not in stage_map:
            raise ValueError(f"Invalid stage: {stage_num}. Must be 1-4.")

        name, func = stage_map[stage_num]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Running {name}...[/bold]")
        stats = func(self)
        self.mark_stage_done(name)
        console.print(f"  {stats.input_count} in → {stats.output_count} kept")

    def _run_bench(self, final_dir: Path) -> None:
        """Run quality benchmark on final output and print results."""
        from dq.shared.shard import read_shards
        from dq.benchmark.runner import run_benchmark

        docs = list(read_shards(final_dir))
        if not docs:
            return

        console.print(f"\n[bold]  Quality benchmark on {len(docs)} final docs[/bold]")
        report = run_benchmark(
            config_path=str(self.config_path) if hasattr(self, "config_path") and self.config_path else None,
            datasets={"final_output": docs},
            workers=1,
        )
        for ds_report in report.datasets.values():
            console.print(f"    Overall pass rate: {ds_report.overall_pass_rate:.1%}")
            for name, rate in ds_report.per_filter_pass_rate.items():
                if rate < 1.0:
                    console.print(f"    [yellow]{name}: {rate:.1%}[/yellow]")
            if ds_report.overall_pass_rate == 1.0:
                console.print("    [green]All filters passed![/green]")

    # Backward compat
    def run_phase(self, phase_num: int) -> None:
        """Deprecated: use run_stage(). Maps old phase numbers to stages."""
        stage_num = _PHASE_TO_STAGE.get(phase_num, phase_num)
        self.run_stage(stage_num)

    def show_plan(self) -> None:
        console.print(f"[bold]Pipeline Plan[/bold] — config: {self.config_path}")
        console.print(f"  Input:   {self.input_path}")
        console.print(f"  Output:  {self.output_dir}")
        console.print(f"  Workers: {self.workers}")
        filters = [fc.name for fc in self.config.filters if fc.enabled]
        console.print(f"  Filters: {filters}")
        console.print()
        for num in sorted(_STAGE_FUNCS):
            name = _STAGE_FUNCS[num]
            done = self.is_stage_done(name)
            status = "[green]done[/green]" if done else "[yellow]pending[/yellow]"
            console.print(f"  Stage {num}: {name} — {status}")


_STAGE_FUNCS = {1: "ingestion", 2: "extraction", 3: "curation", 4: "packaging"}
