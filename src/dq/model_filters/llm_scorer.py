"""LLM Quality Scorer using FineWeb-Edu classifier (or compatible).

Scores documents on educational/quality content (0-5 scale).
Gracefully skips if transformers/torch not installed.
"""

from __future__ import annotations

import logging
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

logger = logging.getLogger(__name__)

_transformers_available = None

# Singleton cache for scorer pipelines
_scorer_cache: dict[str, Any] = {}


def _check_transformers() -> bool:
    global _transformers_available
    if _transformers_available is None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            _transformers_available = True
        except ImportError:
            _transformers_available = False
    return _transformers_available


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_scorer(model_name: str, device: str):
    """Load and cache the classification pipeline."""
    key = f"{model_name}@{device}"
    if key not in _scorer_cache:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        _scorer_cache[key] = (model, tokenizer)
        logger.info("Loaded LLM scorer model %s on %s", model_name, device)
    return _scorer_cache[key]


@register_filter("llm_quality")
class LLMQualityScorer(BaseFilter):
    """Score documents using a trained quality classifier (FineWeb-Edu style).

    Uses a sequence classification model to predict an educational/quality
    score (0-5). Documents scoring below the threshold are dropped.

    Args:
        model_name: HuggingFace model for quality scoring.
        threshold: Minimum score to keep a document.
        batch_size: Batch size for inference (unused in streaming mode).
        device: Device for inference ('auto', 'cuda', 'cpu').
        text_field: Document field containing text.
    """

    name = "llm_quality"

    def __init__(
        self,
        model_name: str = "HuggingFaceFW/fineweb-edu-classifier",
        threshold: float = 3.0,
        batch_size: int = 16,
        device: str = "auto",
        text_field: str = "text",
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device
        self._available = _check_transformers()
        self._loaded = False

        if not self._available:
            logger.warning("transformers/torch not installed — LLMQualityScorer will pass all docs. "
                           "Install with: pip install transformers torch")

    def _ensure_model(self):
        if self._loaded or not self._available:
            return
        self.device = _resolve_device(self.device)
        _get_scorer(self.model_name, self.device)
        self._loaded = True

    def _score_text(self, text: str) -> float:
        """Score a single text."""
        import torch

        model, tokenizer = _get_scorer(self.model_name, self.device)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            score = logits.item()
        return score

    def filter(self, doc: dict) -> tuple[bool, dict]:
        self._ensure_model()
        if not self._available:
            return True, {"edu_score": -1.0, "reason": "skipped"}

        text = self.get_text(doc)
        if not text.strip():
            return False, {"edu_score": 0.0, "reason": "empty text"}

        score = self._score_text(text)
        keep = score >= self.threshold
        return keep, {"edu_score": round(score, 4)}


def score_documents(docs: list[dict], model_name: str = "HuggingFaceFW/fineweb-edu-classifier",
                    text_field: str = "text", device: str = "auto") -> list[float]:
    """Score a list of documents without filtering.

    Args:
        docs: List of document dicts.
        model_name: HuggingFace model for scoring.
        text_field: Field containing text.
        device: Device for inference.

    Returns:
        List of scores (one per document).
    """
    scorer = LLMQualityScorer(model_name=model_name, threshold=0.0, device=device, text_field=text_field)
    scores = []
    for doc in docs:
        _, info = scorer.filter(doc)
        scores.append(info.get("edu_score", -1.0))
    return scores
