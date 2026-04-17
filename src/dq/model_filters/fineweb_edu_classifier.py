"""FineWeb-Edu quality classifier (batched, CPU/GPU).

Thin wrapper around HuggingFaceFW/fineweb-edu-classifier — a BERT-scale model
trained on LLM-annotated educational-quality labels (scores 0..5).

Use as a cheap alternative to the LLM judge at scale:
    ~10-50 ms/doc on CPU, <2 ms/doc on a single GPU,
    vs ~1-10 s/doc for an LLM API call.

The object lazily loads the model on first use. Safe to construct once and
call `score_batch(...)` many times.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "HuggingFaceFW/fineweb-edu-classifier"


class FineWebEduClassifier:
    """Batched scorer. Holds a HF tokenizer + model across calls."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as e:
            raise RuntimeError(
                "FineWebEduClassifier needs torch + transformers. "
                "Install with: pip install torch transformers"
            ) from e

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._device = self._pick_device(device)
        logger.info("Loading %s on %s", model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _pick_device(device: str) -> str:
        import torch
        if device in ("cuda", "cpu"):
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def score_batch(self, texts: list[str]) -> list[float]:
        """Return one float score per input text (roughly 0..5)."""
        import torch

        if not texts:
            return []
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t or "" for t in texts[i:i + self.batch_size]]
            enc = self._tokenizer(
                chunk,
                truncation=True,
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits.detach().float().cpu()
            if logits.shape[-1] == 1:
                # Regression head (fineweb-edu default)
                scores = logits.squeeze(-1).tolist()
            else:
                # Multi-class head: argmax
                scores = logits.argmax(dim=-1).float().tolist()
            out.extend(float(s) for s in scores)
        return out

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]

    # alias for symmetry
    score_batch = score_batch  # type: ignore


# Cache one classifier per (model_name, device) tuple to avoid reloading.
_cache: dict[tuple[str, str], FineWebEduClassifier] = {}


def get_classifier(model_name: str = _DEFAULT_MODEL, device: str = "auto",
                   max_length: int = 512, batch_size: int = 32) -> FineWebEduClassifier:
    key = (model_name, device)
    clf = _cache.get(key)
    if clf is None:
        clf = FineWebEduClassifier(model_name, device, max_length, batch_size)
        _cache[key] = clf
    return clf
