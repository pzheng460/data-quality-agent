# DEPRECATED: Use PretrainingQualityJudge (model_filters/llm_quality_judge.py) for binary quality classification.
# This absolute scoring approach (1-6) is unreliable. Kept for backward compatibility.

"""Educational Value Scorer for pre-training data.

Scores text on educational value (1-6) using an LLM (OpenAI-compatible API).
Higher educational value indicates better learning potential in the text.
"""

from __future__ import annotations

import logging
import time

from dq.llm_client import get_client, get_default_model

logger = logging.getLogger(__name__)

EDUCATIONAL_VALUE_PROMPT = """Score the educational value of the following text on a scale of 1 to 6, where:
1 = No educational value (ads, spam, boilerplate, navigation menus)
2 = Minimal value (social media chatter, trivial content, listicles)
3 = Some value (news articles, basic informational content)
4 = Good educational content (tutorials, explanations, how-to guides)
5 = High value (textbook-quality, in-depth analysis, detailed technical content)
6 = Exceptional (expert-level, comprehensive treatment, original research)

Respond with ONLY a single integer (1-6).

Text:
{text}

Educational value score:"""


class EducationalValueScorer:
    """Score text educational value using an LLM.

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

        prompt = EDUCATIONAL_VALUE_PROMPT.format(text=text[:4000])

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
                logger.warning("No digit in educational value score: %r", resp_text)
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
        """Score a single document's educational value.

        Returns the document dict with `educational_value_score` added.
        """
        text = doc.get(text_field, "")
        if not text:
            doc["educational_value_score"] = -1.0
            return doc
        doc["educational_value_score"] = self._score_one(text)
        return doc

    def score_batch(self, docs: list[dict], text_field: str = "text") -> list[dict]:
        """Score a batch of documents."""
        for doc in docs:
            self.score(doc, text_field)
        return docs
