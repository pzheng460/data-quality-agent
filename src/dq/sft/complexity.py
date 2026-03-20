"""DEITA Complexity Scorer for SFT data.

Scores instruction complexity using an LLM (OpenAI-compatible API).
More complex instructions produce more valuable training data.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_openai_available = None

COMPLEXITY_PROMPT = """Score the complexity of the following instruction on a scale of 1 to 6, where:
1 = Very simple, trivial task (e.g., "Say hello")
2 = Simple task with clear answer (e.g., "What is 2+2?")
3 = Moderate task requiring some thought (e.g., "Explain photosynthesis")
4 = Complex task requiring analysis (e.g., "Compare and contrast two economic theories")
5 = Very complex task requiring deep expertise (e.g., "Design a distributed system for...")
6 = Extremely complex, multi-step reasoning task

Respond with ONLY a single integer (1-6).

Instruction:
{instruction}

Complexity score:"""


def _check_openai() -> bool:
    global _openai_available
    if _openai_available is None:
        try:
            import openai  # noqa: F401
            _openai_available = True
        except ImportError:
            _openai_available = False
    return _openai_available


class ComplexityScorer:
    """Score instruction complexity using the DEITA approach.

    This is a scorer, not a filter — it adds a `complexity_score` field
    to each document without dropping any.

    Args:
        api_url: OpenAI-compatible API base URL (None for default OpenAI).
        api_key: API key (None to use OPENAI_API_KEY env var).
        model: Model name for scoring.
        batch_size: Number of documents to score per API batch.
        max_retries: Max retries on API failure.
        retry_delay: Seconds between retries.
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        batch_size: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None
        self._available = _check_openai()

        if not self._available:
            logger.warning("openai not installed — ComplexityScorer will skip scoring. "
                           "Install with: pip install openai")

    def _get_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is not None:
            return self._client
        if not self._available:
            return None

        import openai

        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_url:
            kwargs["base_url"] = self.api_url

        self._client = openai.OpenAI(**kwargs)
        return self._client

    def _score_one(self, instruction: str) -> float:
        """Score a single instruction via API with retry logic."""
        client = self._get_client()
        if client is None:
            return -1.0

        prompt = COMPLEXITY_PROMPT.format(instruction=instruction[:4000])

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                text = response.choices[0].message.content.strip()
                score = float(text)
                return max(1.0, min(6.0, score))
            except (ValueError, TypeError):
                logger.warning("Failed to parse complexity score: %r", text)
                return 3.0  # default middle score
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning("API call failed (attempt %d): %s", attempt + 1, e)
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("API call failed after %d retries: %s", self.max_retries, e)
                    return -1.0
        return -1.0

    def score(self, doc: dict, instruction_field: str = "instruction") -> dict:
        """Score a single document's instruction complexity.

        Args:
            doc: Document dict with an instruction field.
            instruction_field: Name of the instruction field.

        Returns:
            The document dict with `complexity_score` added.
        """
        instruction = doc.get(instruction_field, "")
        if not instruction:
            doc["complexity_score"] = -1.0
            return doc

        doc["complexity_score"] = self._score_one(instruction)
        return doc

    def score_batch(self, docs: list[dict], instruction_field: str = "instruction") -> list[dict]:
        """Score a batch of documents.

        Args:
            docs: List of document dicts.
            instruction_field: Name of the instruction field.

        Returns:
            Documents with `complexity_score` added.
        """
        for doc in docs:
            self.score(doc, instruction_field)
        return docs
