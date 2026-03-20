"""Perplexity filter using a small language model (CCNet / Llama 3 style).

Very high perplexity = gibberish/low quality.
Very low perplexity = repetitive/boilerplate.
Gracefully skips if transformers/torch not installed.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

logger = logging.getLogger(__name__)

_transformers_available = None

# Singleton cache for loaded models
_model_cache: dict[str, tuple[Any, Any]] = {}


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
    """Resolve 'auto' to cuda/cpu."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_model(model_name: str, device: str):
    """Load and cache a causal LM + tokenizer."""
    key = f"{model_name}@{device}"
    if key not in _model_cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        model.to(device)
        model.eval()
        _model_cache[key] = (model, tokenizer)
        logger.info("Loaded perplexity model %s on %s", model_name, device)
    return _model_cache[key]


@register_filter("perplexity")
class PerplexityFilter(BaseFilter):
    """Filter documents by perplexity from a small language model.

    Documents with very high perplexity (gibberish) or very low perplexity
    (boilerplate) are dropped.

    Args:
        model_name: HuggingFace model name for perplexity computation.
        max_perplexity: Upper bound — drop if perplexity exceeds this.
        min_perplexity: Lower bound — drop if perplexity is below this.
        batch_size: Batch size for inference (unused in streaming mode).
        device: Device for inference ('auto', 'cuda', 'cpu').
        text_field: Document field containing text.
    """

    name = "perplexity"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        max_perplexity: float = 1000.0,
        min_perplexity: float = 5.0,
        batch_size: int = 8,
        device: str = "auto",
        text_field: str = "text",
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.model_name = model_name
        self.max_perplexity = max_perplexity
        self.min_perplexity = min_perplexity
        self.batch_size = batch_size
        self.device = device
        self._available = _check_transformers()
        self._loaded = False

        if not self._available:
            logger.warning("transformers/torch not installed — PerplexityFilter will pass all docs. "
                           "Install with: pip install transformers torch")

    def _ensure_model(self):
        if self._loaded or not self._available:
            return
        self.device = _resolve_device(self.device)
        _get_model(self.model_name, self.device)
        self._loaded = True

    def _compute_perplexity(self, text: str) -> float:
        """Compute token-level perplexity with sliding window."""
        import torch

        model, tokenizer = _get_model(self.model_name, self.device)

        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        input_ids = encodings.input_ids.to(self.device)
        seq_len = input_ids.size(1)

        if seq_len <= 1:
            return 0.0

        max_length = getattr(model.config, "max_position_embeddings", 2048)
        stride = max(1, max_length // 2)

        nlls = []
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_begin = max(begin, stride) if begin > 0 else 1
            input_slice = input_ids[:, begin:end]

            with torch.no_grad():
                outputs = model(input_slice, labels=input_slice)

            # Manually compute NLL for the target portion
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_slice[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Only count tokens in the target range
            offset = target_begin - begin - 1  # -1 because of the shift
            if offset < 0:
                offset = 0
            nlls.extend(token_losses[offset:].tolist())

            if end == seq_len:
                break

        if not nlls:
            return 0.0

        avg_nll = sum(nlls) / len(nlls)
        return math.exp(avg_nll)

    def filter(self, doc: dict) -> tuple[bool, dict]:
        self._ensure_model()
        if not self._available:
            return True, {"perplexity": -1.0, "reason": "skipped"}

        text = self.get_text(doc)
        if not text.strip():
            return False, {"perplexity": float("inf"), "reason": "empty text"}

        ppl = self._compute_perplexity(text)

        keep = self.min_perplexity <= ppl <= self.max_perplexity
        info: dict[str, Any] = {"perplexity": round(ppl, 2)}
        if not keep:
            if ppl > self.max_perplexity:
                info["reason"] = "perplexity too high (gibberish)"
            else:
                info["reason"] = "perplexity too low (boilerplate)"
        return keep, info
