"""FastText-based quality classifier filter (DCLM style).

Uses a binary fastText classifier to score documents as high-quality or not.
Gracefully skips if fasttext is not installed.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

logger = logging.getLogger(__name__)

_fasttext_available = None


def _check_fasttext() -> bool:
    global _fasttext_available
    if _fasttext_available is None:
        try:
            import fasttext  # noqa: F401
            _fasttext_available = True
        except ImportError:
            _fasttext_available = False
    return _fasttext_available


@register_filter("fasttext_quality")
class FastTextQualityFilter(BaseFilter):
    """Filter documents using a pre-trained fastText quality classifier.

    Args:
        model_path: Path to the fastText .bin model file.
        threshold: Minimum score to keep a document (0-1).
        label: The positive label in the model (e.g. "__label__hq").
        text_field: Document field containing text.
    """

    name = "fasttext_quality"

    def __init__(
        self,
        model_path: str = "",
        threshold: float = 0.5,
        label: str = "__label__hq",
        text_field: str = "text",
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.model_path = model_path
        self.threshold = threshold
        self.label = label
        self._model = None
        self._available = _check_fasttext()

        if not self._available:
            logger.warning("fasttext not installed — FastTextQualityFilter will pass all docs. "
                           "Install with: pip install fasttext-wheel")

    def _load_model(self):
        """Lazy-load the fastText model on first use."""
        if self._model is not None:
            return
        if not self._available:
            return
        if not self.model_path:
            logger.warning("No fastText model_path configured — passing all docs")
            return
        import fasttext
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = fasttext.load_model(self.model_path)
        logger.info("Loaded fastText model from %s", self.model_path)

    def filter(self, doc: dict) -> tuple[bool, dict]:
        self._load_model()
        if self._model is None:
            return True, {"score": -1.0, "label": "skipped", "reason": "no model"}

        text = self.get_text(doc).replace("\n", " ")[:5000]
        labels, probs = self._model.predict(text)
        label = labels[0]
        score = float(probs[0])

        # If the predicted label matches the positive label, use score directly;
        # otherwise the quality score is 1 - score.
        if label == self.label:
            quality_score = score
        else:
            quality_score = 1.0 - score

        keep = quality_score >= self.threshold
        return keep, {"score": round(quality_score, 4), "label": label}


def train_fasttext_classifier(
    positive_file: str,
    negative_file: str,
    output_path: str,
    epoch: int = 25,
    lr: float = 0.1,
    dim: int = 100,
    wordNgrams: int = 2,
) -> str:
    """Train a binary fastText classifier for quality filtering.

    Args:
        positive_file: Path to file with high-quality texts (one per line).
        negative_file: Path to file with low-quality texts (one per line).
        output_path: Path for the output .bin model file.
        epoch: Number of training epochs.
        lr: Learning rate.
        dim: Embedding dimension.
        wordNgrams: Word n-gram size.

    Returns:
        Path to the trained model.
    """
    if not _check_fasttext():
        raise ImportError("fasttext-wheel is required to train. Install with: pip install fasttext-wheel")

    import tempfile

    import fasttext

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        with open(positive_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tmp.write(f"__label__hq {line}\n")
        with open(negative_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tmp.write(f"__label__lq {line}\n")
        train_path = tmp.name

    model = fasttext.train_supervised(
        input=train_path,
        epoch=epoch,
        lr=lr,
        dim=dim,
        wordNgrams=wordNgrams,
    )
    model.save_model(output_path)
    logger.info("Trained fastText model saved to %s", output_path)
    return output_path
