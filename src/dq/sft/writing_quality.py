"""Writing Quality Scorer for pre-training data.

Scores text on writing quality (1-6) using an LLM (OpenAI-compatible API).
Higher quality indicates better-written, more polished text.
"""

from __future__ import annotations

import logging
import time

from dq.llm_client import get_client, get_default_model

logger = logging.getLogger(__name__)

WRITING_QUALITY_PROMPT = """Score the writing quality of the following text on a scale of 1 to 6, where:
1 = Very poor (incoherent, broken text, machine-generated garbage)
2 = Poor (many errors, unclear structure, hard to follow)
3 = Acceptable (readable but unremarkable, some issues)
4 = Good (clear, well-structured, few errors)
5 = Very good (engaging, polished, professional)
6 = Excellent (publication-quality writing, masterful prose)

Respond with ONLY a single integer (1-6).

Text:
{text}

Writing quality score:"""


class WritingQualityScorer:
    """Score text writing quality using an LLM.

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

    def _score_one(self, text: str) -> float:
        """Score a single text via API with retry logic."""
        client = self._get_client()
        if client is None:
            return -1.0

        prompt = WRITING_QUALITY_PROMPT.format(text=text[:4000])

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0.0,
                )
                resp_text = response.choices[0].message.content.strip()
                for ch in resp_text:
                    if ch.isdigit():
                        score = float(ch)
                        return max(1.0, min(6.0, score))
                logger.warning("No digit in writing quality score: %r", resp_text)
                return 3.0
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning("API call failed (attempt %d): %s", attempt + 1, e)
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("API call failed after %d retries: %s", self.max_retries, e)
                    return -1.0
        return -1.0

    def score(self, doc: dict, text_field: str = "text") -> dict:
        """Score a single document's writing quality.

        Returns the document dict with `writing_quality_score` added.
        """
        text = doc.get(text_field, "")
        if not text:
            doc["writing_quality_score"] = -1.0
            return doc
        doc["writing_quality_score"] = self._score_one(text)
        return doc

    def score_batch(self, docs: list[dict], text_field: str = "text") -> list[dict]:
        """Score a batch of documents."""
        for doc in docs:
            self.score(doc, text_field)
        return docs
