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


_LLM_CONFIG_FILENAME = "llm.yaml"


@dataclass
class LLMConfig:
    """Configuration for LLM API (Layer 2 judge).

    Loaded from configs/llm.yaml (gitignored) by default.
    Supports two backends: "anthropic" (default) and "openai".
    """

    api_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    samples: int = 50
    backend: str = "anthropic"  # "anthropic" or "openai"

    @classmethod
    def from_file(cls, config_dir: Path | None = None) -> "LLMConfig":
        """Load LLM config from llm.yaml in the config directory.

        Args:
            config_dir: Directory containing llm.yaml. Defaults to configs/ in project root.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "configs"
        llm_path = config_dir / _LLM_CONFIG_FILENAME
        if not llm_path.exists():
            return cls()
        with open(llm_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            api_url=raw.get("api_url"),
            api_key=raw.get("api_key"),
            model=raw.get("model"),
            samples=raw.get("samples", 50),
            backend=raw.get("backend", "anthropic"),
        )


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
        """Load pipeline config from a YAML file.

        LLM config is loaded from configs/llm.yaml (same directory as the
        pipeline config file). Pipeline YAML can override llm settings inline,
        but the separate llm.yaml is the recommended approach for credentials.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw, config_dir=path.parent)

    @classmethod
    def from_dict(cls, raw: dict, config_dir: Path | None = None) -> "PipelineConfig":
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

        # LLM config: load from llm.yaml, then override with inline pipeline.llm
        llm = LLMConfig.from_file(config_dir)
        llm_raw = pipeline.get("llm", {})
        if llm_raw:
            # Inline overrides (only non-None values)
            if llm_raw.get("api_url"):
                llm.api_url = llm_raw["api_url"]
            if llm_raw.get("api_key"):
                llm.api_key = llm_raw["api_key"]
            if llm_raw.get("model"):
                llm.model = llm_raw["model"]
            if "samples" in llm_raw:
                llm.samples = llm_raw["samples"]
            if llm_raw.get("backend"):
                llm.backend = llm_raw["backend"]

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
            llm=LLMConfig.from_file(),
        )
