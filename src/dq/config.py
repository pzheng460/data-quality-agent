"""YAML-based pipeline configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FilterConfig:
    """Configuration for a single filter."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class LLMConfig:
    """Configuration for LLM API (Layer 2 judge)."""

    api_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    samples: int = 50


@dataclass
class DedupConfig:
    """Configuration for deduplication."""

    exact: bool = True
    minhash: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "num_perm": 112,
        "bands": 14,
        "rows": 8,
        "ngram_size": 5,
    })


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    text_field: str = "text"
    filters: list[FilterConfig] = field(default_factory=list)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load pipeline config from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "PipelineConfig":
        """Build config from a dict (e.g. parsed YAML)."""
        pipeline = raw.get("pipeline", raw)

        text_field = pipeline.get("text_field", "text")

        filters = []
        for fc in pipeline.get("filters", []):
            # Handle pii filter which has 'mode' at top level
            params = dict(fc.get("params", {}))
            if "mode" in fc:
                params["mode"] = fc["mode"]
            filters.append(FilterConfig(
                name=fc["name"],
                params=params,
                enabled=fc.get("enabled", True),
            ))

        dedup_raw = pipeline.get("dedup", {})
        dedup = DedupConfig(
            exact=dedup_raw.get("exact", True),
            minhash=dedup_raw.get("minhash", DedupConfig().minhash),
        )

        llm_raw = pipeline.get("llm", {})
        llm = LLMConfig(
            api_url=llm_raw.get("api_url"),
            api_key=llm_raw.get("api_key"),
            model=llm_raw.get("model"),
            samples=llm_raw.get("samples", 50),
        )

        return cls(text_field=text_field, filters=filters, dedup=dedup, llm=llm)

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Return default pipeline config with all standard filters."""
        return cls(
            text_field="text",
            filters=[
                FilterConfig("gopher_quality"),
                FilterConfig("gopher_repetition"),
                FilterConfig("c4"),
                FilterConfig("fineweb"),
                FilterConfig("pii", params={"mode": "redact"}),
            ],
            dedup=DedupConfig(),
        )
