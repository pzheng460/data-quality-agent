"""Contamination detection report dataclasses and output formatters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContaminationResult:
    """Result of checking a single document for contamination."""

    is_contaminated: bool
    overlap_ratio: float = 0.0
    matched_ngrams: int = 0
    total_ngrams: int = 0
    matched_benchmark: str = ""
    score: float = 0.0  # generic score (Min-K% Prob, TS-Guessing accuracy, etc.)
    method: str = ""


@dataclass
class BenchmarkContamination:
    """Contamination stats for a single benchmark."""

    benchmark_name: str
    total_docs: int = 0
    contaminated_docs: int = 0
    contamination_rate: float = 0.0
    avg_overlap: float = 0.0
    sample_contaminated: list[dict] = field(default_factory=list)


@dataclass
class ContaminationReport:
    """Full contamination scan report across multiple benchmarks."""

    dataset_name: str = ""
    total_docs: int = 0
    contaminated_docs: int = 0
    contamination_rate: float = 0.0
    per_benchmark: dict[str, BenchmarkContamination] = field(default_factory=dict)
    sample_contaminated: list[dict] = field(default_factory=list)
    method: str = "ngram"

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "total_docs": self.total_docs,
            "contaminated_docs": self.contaminated_docs,
            "contamination_rate": round(self.contamination_rate, 4),
            "method": self.method,
            "per_benchmark": {
                name: {
                    "benchmark_name": bc.benchmark_name,
                    "total_docs": bc.total_docs,
                    "contaminated_docs": bc.contaminated_docs,
                    "contamination_rate": round(bc.contamination_rate, 4),
                    "avg_overlap": round(bc.avg_overlap, 4),
                    "sample_contaminated": bc.sample_contaminated[:10],
                }
                for name, bc in self.per_benchmark.items()
            },
            "sample_contaminated": self.sample_contaminated[:10],
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Export report as JSON."""
        json_str = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Generate markdown report."""
        lines: list[str] = []
        lines.append(f"# Contamination Report: {self.dataset_name}")
        lines.append("")
        lines.append(f"**Method**: {self.method}")
        lines.append(f"**Total documents**: {self.total_docs}")
        lines.append(f"**Contaminated documents**: {self.contaminated_docs}")
        lines.append(f"**Contamination rate**: {self.contamination_rate:.2%}")
        lines.append("")

        if self.per_benchmark:
            lines.append("## Per-Benchmark Results")
            lines.append("")
            lines.append("| Benchmark | Docs Checked | Contaminated | Rate | Avg Overlap |")
            lines.append("|-----------|-------------|--------------|------|-------------|")
            for name, bc in self.per_benchmark.items():
                lines.append(
                    f"| {name} | {bc.total_docs} | {bc.contaminated_docs} "
                    f"| {bc.contamination_rate:.2%} | {bc.avg_overlap:.4f} |"
                )
            lines.append("")

        if self.sample_contaminated:
            lines.append("## Sample Contaminated Documents")
            lines.append("")
            for i, sample in enumerate(self.sample_contaminated[:10]):
                preview = sample.get("text", "")[:150].replace("\n", " ")
                overlap = sample.get("overlap_ratio", 0.0)
                benchmark = sample.get("matched_benchmark", "unknown")
                lines.append(f"{i + 1}. **{benchmark}** (overlap={overlap:.2%})")
                lines.append(f"   > {preview}...")
                lines.append("")

        md = "\n".join(lines)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(md, encoding="utf-8")
        return md

    def print_rich(self, console=None) -> None:
        """Print report as a rich table."""
        from rich.console import Console
        from rich.table import Table

        console = console or Console()

        console.print(f"\n[bold]Contamination Report: {self.dataset_name}[/bold]")
        console.print(f"[dim]Method: {self.method} | "
                       f"Total: {self.total_docs} | "
                       f"Contaminated: {self.contaminated_docs} | "
                       f"Rate: {self.contamination_rate:.2%}[/dim]\n")

        if self.per_benchmark:
            table = Table(title="Per-Benchmark Contamination")
            table.add_column("Benchmark", style="bold")
            table.add_column("Docs", justify="right")
            table.add_column("Contaminated", justify="right")
            table.add_column("Rate", justify="right")
            table.add_column("Avg Overlap", justify="right")

            for name, bc in self.per_benchmark.items():
                table.add_row(
                    name,
                    str(bc.total_docs),
                    str(bc.contaminated_docs),
                    f"{bc.contamination_rate:.2%}",
                    f"{bc.avg_overlap:.4f}",
                )
            console.print(table)

        if self.sample_contaminated:
            console.print("\n[bold]Sample Contaminated Documents:[/bold]")
            for i, sample in enumerate(self.sample_contaminated[:5]):
                preview = sample.get("text", "")[:120].replace("\n", " ")
                overlap = sample.get("overlap_ratio", 0.0)
                benchmark = sample.get("matched_benchmark", "unknown")
                console.print(f"  [{i + 1}] [dim]{preview}...[/dim]")
                console.print(f"      Benchmark: {benchmark} | Overlap: {overlap:.2%}")
