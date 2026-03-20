"""DEITA Quality Scorer for SFT data.

Scores response quality given an instruction using an LLM (OpenAI-compatible API).
Higher quality responses provide better training signal.
"""

from __future__ import annotations

import logging
import time

from dq.llm_client import get_client, get_default_model

logger = logging.getLogger(__name__)

QUALITY_PROMPT = """Score the quality of the following response given the instruction on a scale of 1 to 6, where:
1 = Very poor: Incorrect, irrelevant, or incoherent response
2 = Poor: Partially relevant but with significant errors or missing key information
3 = Acceptable: Relevant but lacking depth or contains minor errors
4 = Good: Correct, relevant, and reasonably detailed
5 = Very good: Accurate, comprehensive, and well-structured
6 = Excellent: Outstanding quality, thorough, insightful, and perfectly addresses the instruction

Respond with ONLY a single integer (1-6).

Instruction:
{instruction}

Response:
{output}

Quality score:"""


class QualityScorer:
    """Score response quality using the DEITA approach.

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

    def _score_one(self, instruction: str, output: str) -> float:
        """Score a single instruction-response pair via API with retry logic."""
        client = self._get_client()
        if client is None:
            return -1.0

        prompt = QUALITY_PROMPT.format(
            instruction=instruction[:4000],
            output=output[:4000],
        )
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
                for ch in text:
                    if ch.isdigit():
                        score = float(ch)
                        return max(1.0, min(6.0, score))
                logger.warning("No digit in quality score: %r", text)
                return 3.0
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning("API call failed (attempt %d): %s", attempt + 1, e)
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("API call failed after %d retries: %s", self.max_retries, e)
                    return -1.0
        return -1.0

    def score(self, doc: dict, instruction_field: str = "instruction",
              output_field: str = "output") -> dict:
        """Score a single document's response quality.

        Returns the document dict with `quality_score` added.
        """
        instruction = doc.get(instruction_field, "")
        output = doc.get(output_field, "")
        if not instruction or not output:
            doc["quality_score"] = -1.0
            return doc
        doc["quality_score"] = self._score_one(instruction, output)
        return doc

    def score_batch(self, docs: list[dict], instruction_field: str = "instruction",
                    output_field: str = "output") -> list[dict]:
        """Score a batch of documents."""
        for doc in docs:
            self.score(doc, instruction_field, output_field)
        return docs
