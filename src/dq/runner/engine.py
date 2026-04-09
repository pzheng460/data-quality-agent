"""PhaseEngine — orchestrates the multi-phase production cleaning pipeline."""

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

_PHASE_FUNCS = {
    1: "phase1_parse",
    2: "phase2_filter",
    3: "phase3_dedup",
    4: "phase4_contamination",
    5: "phase5_package",
}


class PhaseEngine:
    """Orchestrates multi-phase pipeline with _SUCCESS markers for resumability."""

    def __init__(
        self,
        config_path: str,
        input_path: str,
        output_dir: str,
        workers: int | None = None,
        num_samples: int = 0,
    ) -> None:
        self.config_path = config_path

        # Load YAML for both PipelineConfig and arxiv-specific section
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self.config = PipelineConfig.from_dict(raw, config_dir=Path(config_path).parent)
        self.arxiv_config: dict = raw.get("arxiv", {})

        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.version = self.arxiv_config.get("version", "unknown")

        # Worker count
        parallelism = self.arxiv_config.get("parallelism", {})
        if workers is not None:
            self.workers = workers
        elif parallelism.get("num_workers"):
            self.workers = parallelism["num_workers"]
        else:
            self.workers = max(1, min((os.cpu_count() or 1) // 4, 32))

        # Shard target size
        phase5_cfg = self.arxiv_config.get("phase5", {})
        self.shard_target_bytes = phase5_cfg.get("shard_target_bytes", 1_073_741_824)

        # Config hash for manifest
        with open(config_path, "rb") as f:
            self._config_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        return self._config_hash

    def stage_dir(self, stage_name: str, sub: str | None = None) -> Path:
        """Path for a pipeline stage: output_dir/stage_name[/sub]."""
        base = self.output_dir / stage_name
        return base / sub if sub else base

    def _success_path(self, phase_name: str) -> Path:
        return self.output_dir / f".{phase_name}_SUCCESS"

    def is_phase_done(self, phase_name: str) -> bool:
        return self._success_path(phase_name).exists()

    def mark_phase_done(self, phase_name: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._success_path(phase_name).touch()

    def iter_input(self) -> Iterator[dict]:
        """Iterate input documents (with optional num_samples limit)."""
        from dq.runner.shard import read_shards
        from dq.utils.io import read_docs

        if self.input_path.is_dir():
            source = read_shards(self.input_path)
        else:
            source = read_docs(self.input_path)

        count = 0
        for doc in source:
            yield doc
            count += 1
            if self.num_samples > 0 and count >= self.num_samples:
                break

    def run_all(self, resume: bool = True) -> None:
        """Run all 5 phases, skipping completed ones if resume=True."""
        from dq.runner.phases import (
            phase1_parse,
            phase2_filter,
            phase3_dedup,
            phase4_contamination,
            phase5_package,
        )
        from dq.runner.stats import save_overview

        self.output_dir.mkdir(parents=True, exist_ok=True)

        phase_list = [
            ("phase1_parse", phase1_parse),
            ("phase2_filter", phase2_filter),
            ("phase3_dedup", phase3_dedup),
            ("phase4_contamination", phase4_contamination),
            ("phase5_package", phase5_package),
        ]

        all_stats = []
        for name, func in phase_list:
            if resume and self.is_phase_done(name):
                console.print(f"[dim]Skipping {name} (already done)[/dim]")
                continue

            console.print(f"[bold]Running {name}...[/bold]")
            stats = func(self)
            self.mark_phase_done(name)
            all_stats.append(stats)

            # Save per-phase stats
            stats_dir = self.output_dir / "stats" / self.version
            stats_dir.mkdir(parents=True, exist_ok=True)
            stats.save(stats_dir / f"{name}.json")

            console.print(
                f"  {stats.input_count} in -> {stats.output_count} kept, "
                f"{stats.rejected_count} rejected ({stats.duration_seconds:.1f}s)"
            )

        if all_stats:
            stats_dir = self.output_dir / "stats" / self.version
            save_overview(stats_dir, all_stats, self.version, config_hash=self._config_hash)

    def run_phase(self, phase_num: int) -> None:
        """Run a specific phase by number (1-5)."""
        from dq.runner import phases as pm

        phase_map = {
            1: ("phase1_parse", pm.phase1_parse),
            2: ("phase2_filter", pm.phase2_filter),
            3: ("phase3_dedup", pm.phase3_dedup),
            4: ("phase4_contamination", pm.phase4_contamination),
            5: ("phase5_package", pm.phase5_package),
        }
        if phase_num not in phase_map:
            raise ValueError(f"Invalid phase: {phase_num}. Must be 1-5.")

        name, func = phase_map[phase_num]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Running {name}...[/bold]")
        stats = func(self)
        self.mark_phase_done(name)
        console.print(
            f"  {stats.input_count} in -> {stats.output_count} kept, "
            f"{stats.rejected_count} rejected ({stats.duration_seconds:.1f}s)"
        )

    def show_plan(self) -> None:
        """Print what phases would execute (dry-run)."""
        console.print(f"[bold]Pipeline Plan[/bold] -- config: {self.config_path}")
        console.print(f"  Version: {self.version}")
        console.print(f"  Input:   {self.input_path}")
        console.print(f"  Output:  {self.output_dir}")
        console.print(f"  Workers: {self.workers}")
        filters = [fc.name for fc in self.config.filters if fc.enabled]
        console.print(f"  Filters: {filters}")
        console.print()
        for num in sorted(_PHASE_FUNCS):
            name = _PHASE_FUNCS[num]
            done = self.is_phase_done(name)
            status = "[green]done[/green]" if done else "[yellow]pending[/yellow]"
            console.print(f"  Phase {num}: {name} -- {status}")
