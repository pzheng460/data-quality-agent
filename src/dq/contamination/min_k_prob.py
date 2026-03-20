"""Min-K% Prob black-box contamination detection.

Based on: "Detecting Pretraining Data from Large Language Models" (Shi et al., 2023).
If a model assigns high probability to ALL tokens in a text, that text was likely
in training data. Min-K% looks at the K% lowest-probability tokens.

Requires: transformers, torch (optional dependencies).
"""

from __future__ import annotations

import logging
from typing import Any

from dq.contamination.report import ContaminationResult

logger = logging.getLogger(__name__)

_TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class MinKProbDetector:
    """Detect contamination using Min-K% Prob method.

    Black-box approach: uses a language model to compute token
    probabilities and flags documents where the minimum K%
    of token probs are suspiciously high (memorized).

    Args:
        model_name: HuggingFace model name for computing log probs.
        k_percent: Percentage of lowest-probability tokens to consider.
        threshold: Score above which a document is flagged as contaminated.
            Higher (less negative) = more likely contaminated.
        device: Device for inference ("auto", "cpu", "cuda").
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        k_percent: float = 20.0,
        threshold: float = -0.5,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.k_percent = k_percent
        self.threshold = threshold
        self._device_str = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

        if not _TRANSFORMERS_AVAILABLE:
            logger.warning(
                "transformers/torch not installed. "
                "MinKProbDetector will skip all documents. "
                "Install with: uv sync --extra model"
            )

    @property
    def available(self) -> bool:
        """Check if required dependencies are installed."""
        return _TRANSFORMERS_AVAILABLE

    def _load_model(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return
        if not _TRANSFORMERS_AVAILABLE:
            return

        if self._device_str == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self._device_str)

        logger.info("Loading model '%s' on %s...", self.model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self._device.type == "cuda" else torch.float32,
        ).to(self._device)
        self._model.eval()
        logger.info("Model loaded.")

    def compute_min_k_prob(self, text: str) -> float:
        """Compute Min-K% Prob score for a single text.

        Args:
            text: Document text.

        Returns:
            Score (average of bottom K% log probabilities).
            Higher (less negative) = more likely contaminated.
            Returns float('-inf') if unable to compute.
        """
        if not _TRANSFORMERS_AVAILABLE:
            return float("-inf")

        self._load_model()

        if not text.strip():
            return float("-inf")

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 2:
            return float("-inf")

        # Get log probabilities
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift: predict token[i+1] from position[i]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Log softmax to get log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1).squeeze(0)  # (seq_len - 1,)

        # Take bottom K% of log probs
        k = max(1, int(len(token_log_probs) * self.k_percent / 100.0))
        bottom_k = torch.topk(token_log_probs, k, largest=False).values

        # Average of bottom K% log probs
        score = bottom_k.mean().item()
        return score

    def check_contamination(self, doc: str) -> ContaminationResult:
        """Check a single document for contamination.

        Args:
            doc: Document text.

        Returns:
            ContaminationResult with Min-K% score.
        """
        score = self.compute_min_k_prob(doc)

        return ContaminationResult(
            is_contaminated=score > self.threshold if score != float("-inf") else False,
            overlap_ratio=0.0,
            matched_ngrams=0,
            total_ngrams=0,
            matched_benchmark="",
            score=score,
            method="min_k_prob",
        )

    def check_batch(self, texts: list[str]) -> list[ContaminationResult]:
        """Check a batch of documents.

        Args:
            texts: List of document texts.

        Returns:
            List of ContaminationResult objects.
        """
        return [self.check_contamination(text) for text in texts]
