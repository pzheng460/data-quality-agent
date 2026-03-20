"""Pre-training Quality Judge - Rule-based binary classification for plain text data.

Replaces absolute 1-6 scoring with binary HIGH/LOW classification based on 3 specific rules:
- information_density: Contains substantive, informative content
- coherence: Well-structured, complete paragraphs
- originality: Appears to be original content

ALL rules pass → HIGH quality. ANY rule fails → LOW quality.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from dq.llm_client import get_client, get_default_model

logger = logging.getLogger(__name__)

# Prompt for binary rule-based classification
PRETRAIN_JUDGE_PROMPT = """You are a quality judge for pre-training text data. Evaluate the text against these 3 rules and return structured JSON.

Rules:
1. information_density: Does the text contain substantive, informative content? PASS if yes, FAIL if ads, navigation menus, cookie notices, or boilerplate.
2. coherence: Is the text well-structured with complete paragraphs? PASS if coherent, FAIL if truncated, fragmented, or garbled text.
3. originality: Does the text appear to be original content? PASS if original, FAIL if SEO spam, template-generated, or machine-generated filler.

Text: {text}

Respond with ONLY valid JSON in this exact format:
{{
  "information_density": {{"pass": true/false, "reason": "brief explanation if failed"}},
  "coherence": {{"pass": true/false, "reason": "brief explanation if failed"}},
  "originality": {{"pass": true/false, "reason": "brief explanation if failed"}}
}}"""


class PretrainingQualityJudge:
    """Binary quality classifier for pre-training (plain text) data.

    Uses rule-based binary classification instead of absolute scoring.
    Each rule returns pass/fail + reason. Final verdict: ALL pass → HIGH, ANY fail → LOW.

    Args:
        api_url: OpenAI-compatible API base URL.
        api_key: API key.
        model: Model name for judging.
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
        """Get LLM client using shared infrastructure."""
        return get_client(api_url=self.api_url, api_key=self.api_key)

    def _parse_rules_json(self, response_text: str) -> dict[str, dict[str, Any]]:
        """Parse rules from LLM response, with fallback if malformed."""
        # Try direct JSON parsing first
        try:
            data = json.loads(response_text.strip())
            if isinstance(data, dict) and all(
                key in data for key in [
                    "information_density", "coherence", "originality"
                ]
            ):
                return data
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: Extract pass/fail via regex
        logger.warning("Failed to parse pretrain judge response as JSON, using regex fallback: %r",
                      response_text[:200])

        rules = {}
        rule_names = ["information_density", "coherence", "originality"]

        for rule in rule_names:
            # Look for patterns like "information_density": {"pass": true/false
            pattern = rf'"{rule}":\s*{{\s*"pass":\s*(true|false)'
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                rules[rule] = {
                    "pass": match.group(1).lower() == "true",
                    "reason": ""
                }
            else:
                # Default to fail if can't parse
                rules[rule] = {"pass": False, "reason": "Parse error - defaulted to fail"}

        return rules

    def judge_one(self, text: str) -> dict[str, Any]:
        """Judge a single text document.

        Returns:
            dict with keys:
            - quality: "high" | "low"
            - rules: dict of rule results
            - failed_rules: list of rule names that failed
        """
        client = self._get_client()
        if client is None:
            logger.error("No LLM client available for pretrain quality judgment")
            return {
                "quality": "low",
                "rules": {},
                "failed_rules": ["api_unavailable"],
                "error": "No API client available"
            }

        # Truncate text to avoid token limits
        text = text[:4000]

        prompt = PRETRAIN_JUDGE_PROMPT.format(text=text)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.0,
                )
                response_text = response.choices[0].message.content.strip()
                rules = self._parse_rules_json(response_text)

                # Determine overall quality
                failed_rules = [
                    rule_name for rule_name, rule_result in rules.items()
                    if not rule_result.get("pass", False)
                ]

                quality = "high" if len(failed_rules) == 0 else "low"

                return {
                    "quality": quality,
                    "rules": rules,
                    "failed_rules": failed_rules
                }

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning("Pretrain judge API call failed (attempt %d): %s", attempt + 1, e)
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("Pretrain judge API call failed after %d retries: %s", self.max_retries, e)
                    return {
                        "quality": "low",
                        "rules": {},
                        "failed_rules": ["api_error"],
                        "error": str(e)
                    }

        return {
            "quality": "low",
            "rules": {},
            "failed_rules": ["unknown_error"]
        }

    def judge_batch(self, docs: list[dict], text_field: str = "text") -> list[dict]:
        """Judge multiple documents.

        Adds judgment results to each document in place.
        """
        for doc in docs:
            text = doc.get(text_field, "")

            if not text.strip():
                doc.update({
                    "pretrain_quality": "low",
                    "pretrain_rules": {},
                    "pretrain_failed_rules": ["empty_text"]
                })
                continue

            result = self.judge_one(text)
            doc.update({
                "pretrain_quality": result["quality"],
                "pretrain_rules": result["rules"],
                "pretrain_failed_rules": result["failed_rules"]
            })

            if "error" in result:
                doc["pretrain_error"] = result["error"]

        return docs