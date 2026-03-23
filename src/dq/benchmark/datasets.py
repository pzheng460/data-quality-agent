"""Dataset loading functions for benchmarking."""

import json
import logging
import random
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

FINEWEB_DATASET = "HuggingFaceFW/fineweb"
FINEWEB_CONFIG = "sample-10BT"
ALPACA_DATASET = "tatsu-lab/alpaca"
ALPACA_ORIGINAL = "tatsu-lab/alpaca"
ALPACA_CLEANED = "yahma/alpaca-cleaned"


def _ensure_datasets():
    """Import datasets library or raise helpful error."""
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` library is required for benchmarks. "
            "Install with: uv pip install 'dq[bench]'"
        )


def _merge_alpaca_fields(item: dict) -> str:
    """Concatenate instruction + input + output into a single text field."""
    parts = [
        item.get("instruction", "") or "",
        item.get("input", "") or "",
        item.get("output", "") or "",
    ]
    return "\n".join(p for p in parts if p.strip())


def load_hf_dataset(
    dataset_id: str,
    n: int = 1000,
    split: str = "train",
    text_field: str = "text",
    seed: int = 42,
    config: str | None = None,
) -> list[dict]:
    """Load n samples from any HuggingFace dataset via streaming.

    Args:
        dataset_id: HuggingFace dataset ID (e.g. 'allenai/dolma3_mix-6T').
        n: Number of samples to load.
        split: Dataset split to use.
        text_field: Field name containing text.
        seed: Random seed for reproducibility.
        config: Optional dataset config/subset name.

    Returns:
        List of dicts with 'text' field.
    """
    load_dataset = _ensure_datasets()

    logger.info("Loading %s (%d samples, streaming)...", dataset_id, n)
    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": True,
    }
    if config:
        kwargs["name"] = config

    try:
        ds = load_dataset(dataset_id, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_id}': {e}") from e

    samples: list[dict] = []
    for item in ds:
        text = item.get(text_field, "")
        if text:
            samples.append({"text": text})
        if len(samples) >= n:
            break

    logger.info("Loaded %d samples from %s.", len(samples), dataset_id)
    return samples


def load_fineweb_sample(n: int = 1000, seed: int = 42) -> list[dict]:
    """Load n random samples from FineWeb sample-10BT (high-quality pre-training data)."""
    load_dataset = _ensure_datasets()

    logger.info("Loading FineWeb sample (%d docs)...", n)
    try:
        ds = load_dataset(
            FINEWEB_DATASET,
            name=FINEWEB_CONFIG,
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    except Exception as e:
        raise RuntimeError(f"Failed to load FineWeb dataset: {e}") from e

    samples: list[dict] = []
    for item in ds:
        samples.append({"text": item["text"]})
        if len(samples) >= n:
            break

    logger.info("Loaded %d FineWeb samples.", len(samples))
    return samples


def load_alpaca_sample(n: int = 1000, seed: int = 42) -> list[dict]:
    """Load n random samples from Stanford Alpaca 52K (mediocre SFT data).

    Concatenates instruction + input + output into a single 'text' field.
    """
    load_dataset = _ensure_datasets()

    logger.info("Loading Alpaca sample (%d docs)...", n)
    try:
        ds = load_dataset(ALPACA_DATASET, split="train")
        ds = ds.shuffle(seed=seed)
    except Exception as e:
        raise RuntimeError(f"Failed to load Alpaca dataset: {e}") from e

    samples: list[dict] = []
    for item in ds:
        text = _merge_alpaca_fields(item)
        samples.append({"text": text})
        if len(samples) >= n:
            break

    logger.info("Loaded %d Alpaca samples.", len(samples))
    return samples


def load_alpaca_original(n: int | None = None, keep_fields: bool = False) -> list[dict]:
    """Load samples from tatsu-lab/stanford_alpaca (original, known quality issues).

    Downloads directly from GitHub since HuggingFace Hub removed the dataset.

    Args:
        n: Number of samples. None or 0 means all samples.
        keep_fields: If True, preserve instruction/input/output fields alongside text.
    """
    cache_path = Path.home() / ".cache" / "dq" / "alpaca_original.json"
    if not cache_path.exists():
        logger.info("Downloading Alpaca original from GitHub...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        try:
            urlretrieve(url, cache_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download Alpaca original: {e}") from e

    logger.info("Loading Alpaca original from cache...")
    with open(cache_path) as f:
        raw = json.load(f)

    # Shuffle with seed for reproducibility
    rng = random.Random(42)
    rng.shuffle(raw)

    limit = n if n and n > 0 else len(raw)
    samples = []
    for item in raw[:limit]:
        doc: dict[str, Any] = {"text": _merge_alpaca_fields(item)}
        if keep_fields:
            doc["instruction"] = item.get("instruction", "") or ""
            doc["input"] = item.get("input", "") or ""
            doc["output"] = item.get("output", "") or ""
        samples.append(doc)
    logger.info("Loaded %d Alpaca original samples.", len(samples))
    return samples


def load_alpaca_cleaned(n: int | None = None, keep_fields: bool = False) -> list[dict]:
    """Load samples from yahma/alpaca-cleaned (community-cleaned version).

    Args:
        n: Number of samples. None or 0 means all samples.
        keep_fields: If True, preserve instruction/input/output fields alongside text.
    """
    load_dataset = _ensure_datasets()
    logger.info("Loading Alpaca cleaned (yahma/alpaca-cleaned)...")
    try:
        ds = load_dataset(ALPACA_CLEANED, split="train")
    except Exception as e:
        raise RuntimeError(f"Failed to load Alpaca cleaned dataset: {e}") from e

    samples: list[dict] = []
    limit = n if n and n > 0 else len(ds)
    for i, item in enumerate(ds):
        if i >= limit:
            break
        doc: dict[str, Any] = {"text": _merge_alpaca_fields(item)}
        if keep_fields:
            doc["instruction"] = item.get("instruction", "") or ""
            doc["input"] = item.get("input", "") or ""
            doc["output"] = item.get("output", "") or ""
        samples.append(doc)
    logger.info("Loaded %d Alpaca cleaned samples.", len(samples))
    return samples