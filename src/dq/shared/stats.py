"""Phase statistics collection and persistence."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PhaseStats:
    """Statistics for a single pipeline phase."""

    phase: str
    input_count: int = 0
    output_count: int = 0
    rejected_count: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def keep_rate(self) -> float:
        return self.output_count / self.input_count if self.input_count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "rejected_count": self.rejected_count,
            "keep_rate": round(self.keep_rate, 4),
            "reject_reasons": self.reject_reasons,
            "duration_seconds": round(self.duration_seconds, 2),
            **self.extra,
        }

    def save(self, path: "Path") -> None:
        import json as _json
        from pathlib import Path as _Path

        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            _json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class PhaseTimer:
    """Context manager to time a phase and update stats."""

    def __init__(self, stats: PhaseStats) -> None:
        self.stats = stats
        self._start = 0.0

    def __enter__(self) -> "PhaseTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args) -> None:
        self.stats.duration_seconds = time.monotonic() - self._start


def save_overview(
    stats_dir: "Path",
    phase_stats: list[PhaseStats],
    version: str,
    config_hash: str = "",
) -> None:
    """Write overview.json with the pipeline funnel data."""
    from pathlib import Path as _Path

    phases = {}
    for ps in phase_stats:
        phases[ps.phase] = {
            "input": ps.input_count,
            "output": ps.output_count,
            "rejected": ps.rejected_count,
            "keep_rate": round(ps.keep_rate, 4),
        }

    overview = {
        "version": "arxiv-" + version,
        "phases": phases,
        "config_sha256": config_hash,
    }

    path = Path(stats_dir) / "overview.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)
