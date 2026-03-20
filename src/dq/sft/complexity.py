# DEPRECATED: Use SFTQualityJudge (llm_judge.py) for binary quality classification.
# This absolute scoring approach (1-6) is unreliable. Kept for backward compatibility.

"""DEITA Complexity Scorer for SFT data.

Scores instruction complexity using an LLM (OpenAI-compatible API).
More complex instructions produce more valuable training data.
"""

from __future__ import annotations

import logging
import time

from dq.llm_client import get_client, get_default_model

logger = logging.getLogger(__name__)

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


class ComplexityScorer:
    """Score instruction complexity using the DEITA approach.

    Uses shared LLM client from dq.llm_client. Configure via:
    - Explicit params (api_url, api_key, model)
    - Env vars: DQ_API_BASE_URL, DQ_API_KEY, DQ_MODEL
    - Fallback: OPENAI_BASE_URL, OPENAI_API_KEY

    Args:
        api_url: OpenAI-compatible API base URL.
        api_key: API key.
        model: Model name for scoring.
        max_retries: Max retries on API failure.
        retry_delay: Seconds between retries.
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model or get_default_model()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _get_client(self):
        return get_client(api_url=self.api_url, api_key=self.api_key)

    def _score_one(self, instruction: str) -> float:
        """Score a single instruction via API with retry logic."""
        client = self._get_client()
        if client is None:
            return -1.0

        prompt = COMPLEXITY_PROMPT.format(instruction=instruction[:4000])
        text = ""

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                text = response.choices[0].message.content.strip()
                # Extract first digit from response
                for ch in text:
                    if ch.isdigit():
                        score = float(ch)
                        return max(1.0, min(6.0, score))
                logger.warning("No digit in complexity score: %r", text)
                return 3.0
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

        Returns the document dict with `complexity_score` added.
        """
        instruction = doc.get(instruction_field, "")
        if not instruction:
            doc["complexity_score"] = -1.0
            return doc
        doc["complexity_score"] = self._score_one(instruction)
        return doc

    def score_batch(self, docs: list[dict], instruction_field: str = "instruction") -> list[dict]:
        """Score a batch of documents."""
        for doc in docs:
            self.score(doc, instruction_field)
        return docs
