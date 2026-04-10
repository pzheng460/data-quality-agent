"""Pipeline orchestrator that chains filters and tracks stats."""

from dataclasses import dataclass, field
from typing import Iterator

from dq.config import PipelineConfig
from dq.stages.curation.filters.base import BaseFilter


# Registry mapping filter names to classes
_FILTER_REGISTRY: dict[str, type[BaseFilter]] = {}


def register_filter(name: str):
    """Decorator to register a filter class under a given name."""
    def wrapper(cls: type[BaseFilter]):
        _FILTER_REGISTRY[name] = cls
        cls.name = name
        return cls
    return wrapper


def get_filter_class(name: str) -> type[BaseFilter]:
    """Look up a filter class by name."""
    if name not in _FILTER_REGISTRY:
        raise ValueError(f"Unknown filter: {name!r}. Available: {list(_FILTER_REGISTRY.keys())}")
    return _FILTER_REGISTRY[name]


@dataclass
class FilterStats:
    """Stats for a single filter."""

    name: str
    docs_in: int = 0
    docs_out: int = 0
    docs_dropped: int = 0
    sample_drops: list[dict] = field(default_factory=list)

    @property
    def drop_rate(self) -> float:
        return self.docs_dropped / self.docs_in if self.docs_in > 0 else 0.0


@dataclass
class PipelineStats:
    """Overall pipeline statistics."""

    total_in: int = 0
    total_out: int = 0
    filter_stats: list[FilterStats] = field(default_factory=list)

    @property
    def total_dropped(self) -> int:
        return self.total_in - self.total_out

    @property
    def overall_drop_rate(self) -> float:
        return self.total_dropped / self.total_in if self.total_in > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert stats to a serializable dict."""
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total_dropped": self.total_dropped,
            "overall_drop_rate": round(self.overall_drop_rate, 4),
            "filters": [
                {
                    "name": fs.name,
                    "docs_in": fs.docs_in,
                    "docs_out": fs.docs_out,
                    "docs_dropped": fs.docs_dropped,
                    "drop_rate": round(fs.drop_rate, 4),
                    "sample_drops": fs.sample_drops[:5],
                }
                for fs in self.filter_stats
            ],
        }


class Pipeline:
    """Pipeline orchestrator that chains filters and tracks statistics."""

    def __init__(self, config: PipelineConfig | None = None, max_drop_samples: int = 5) -> None:
        self.config = config or PipelineConfig.default()
        self.max_drop_samples = max_drop_samples
        self.filters: list[BaseFilter] = []
        self.stats = PipelineStats()

        self._build_filters()

    def _build_filters(self) -> None:
        """Instantiate filter objects from config."""
        for fc in self.config.filters:
            if not fc.enabled:
                continue
            cls = get_filter_class(fc.name)
            self.filters.append(cls(text_field=self.config.text_field, **fc.params))

    def process_doc(self, doc: dict) -> tuple[bool, dict | None]:
        """Run a single document through all filters.

        Returns:
            (keep, doc) - keep is True if doc passes all filters.
            doc may be modified (e.g. PII redaction).
        """
        for f in self.filters:
            keep, info = f.filter(doc)
            if not keep:
                return False, info
        return True, None

    def run(self, docs: Iterator[dict], progress: bool = False) -> Iterator[dict]:
        """Run pipeline on a stream of documents, yielding kept docs.

        Also populates self.stats.
        """
        from tqdm import tqdm

        # Initialize stats
        self.stats = PipelineStats()
        filter_stats = [FilterStats(name=f.name) for f in self.filters]
        self.stats.filter_stats = filter_stats

        doc_iter = tqdm(docs, desc="Processing", disable=not progress)

        for doc in doc_iter:
            self.stats.total_in += 1
            kept = True

            for i, f in enumerate(self.filters):
                filter_stats[i].docs_in += 1
                keep, info = f.filter(doc)

                if not keep:
                    filter_stats[i].docs_dropped += 1
                    if len(filter_stats[i].sample_drops) < self.max_drop_samples:
                        text = doc.get(self.config.text_field, "")
                        filter_stats[i].sample_drops.append({
                            "text_preview": text[:200],
                            "reason": info,
                        })
                    kept = False
                    # Still count as docs_in for remaining filters? No - stop here
                    break
                else:
                    filter_stats[i].docs_out += 1

            if kept:
                self.stats.total_out += 1
                yield doc

    def dry_run(self, docs: Iterator[dict], progress: bool = False) -> PipelineStats:
        """Run pipeline without yielding docs, just collect stats."""
        # Consume the generator to collect stats
        for _ in self.run(docs, progress=progress):
            pass
        return self.stats
