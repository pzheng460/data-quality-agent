"""SFT Quality Judge - Rule-based binary classification for instruction-response data.

Replaces absolute 1-6 scoring with binary HIGH/LOW classification based on 5 specific rules:
- instruction_following: Response addresses the instruction
- factuality: No obvious factual errors
- completeness: Covers what the instruction asks for
- format_compliance: Matches requested format
- harmlessness: No harmful/dangerous content

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
SFT_JUDGE_PROMPT = """You are a quality judge for instruction-response pairs. Evaluate the response against these 5 rules and return structured JSON.

Rules:
1. instruction_following: Does the response directly address what the instruction asks for? PASS if yes, FAIL if off-topic or ignores key requirements.
2. factuality: Are there any obvious factual errors? PASS if no errors, FAIL if contains fabricated facts or wrong information.
3. completeness: Does the response cover what the instruction asks for? PASS if complete, FAIL if missing critical steps or incomplete answer.
4. format_compliance: If the instruction specifies a format, does the response follow it? PASS if format is correct or no format specified, FAIL if wrong format (e.g., asked for JSON but gave prose).
5. harmlessness: Is the content safe and appropriate? PASS if safe, FAIL if contains dangerous instructions, discrimination, or harmful content.

Instruction: {instruction}

Response: {output}

Respond with ONLY valid JSON in this exact format:
{{
  "instruction_following": {{"pass": true/false, "reason": "brief explanation if failed"}},
  "factuality": {{"pass": true/false, "reason": "brief explanation if failed"}},
  "completeness": {{"pass": true/false, "reason": "brief explanation if failed"}},
  "format_compliance": {{"pass": true/false, "reason": "brief explanation if failed"}},
  "harmlessness": {{"pass": true/false, "reason": "brief explanation if failed"}}
}}"""


class SFTQualityJudge:
    """Binary quality classifier for SFT (instruction-response) data.

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
                    "instruction_following", "factuality", "completeness",
                    "format_compliance", "harmlessness"
                ]
            ):
                return data
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: Extract pass/fail via regex
        logger.warning("Failed to parse judge response as JSON, using regex fallback: %r",
                      response_text[:200])

        rules = {}
        rule_names = ["instruction_following", "factuality", "completeness",
                     "format_compliance", "harmlessness"]

        for rule in rule_names:
            # Look for patterns like "instruction_following": {"pass": true/false
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

    def judge_one(self, instruction: str, output: str) -> dict[str, Any]:
        """Judge a single instruction-response pair.

        Returns:
            dict with keys:
            - quality: "high" | "low"
            - rules: dict of rule results
            - failed_rules: list of rule names that failed
        """
        client = self._get_client()
        if client is None:
            logger.error("No LLM client available for SFT quality judgment")
            return {
                "quality": "low",
                "rules": {},
                "failed_rules": ["api_unavailable"],
                "error": "No API client available"
            }

        # Truncate inputs to avoid token limits
        instruction = instruction[:4000]
        output = output[:4000]

        prompt = SFT_JUDGE_PROMPT.format(instruction=instruction, output=output)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
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
                    logger.warning("SFT judge API call failed (attempt %d): %s", attempt + 1, e)
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("SFT judge API call failed after %d retries: %s", self.max_retries, e)
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

    def judge_batch(self, docs: list[dict], instruction_field: str = "instruction",
                   output_field: str = "output") -> list[dict]:
        """Judge multiple documents.

        Adds judgment results to each document in place.
        """
        for doc in docs:
            instruction = doc.get(instruction_field, "")
            output = doc.get(output_field, "")

            if not instruction or not output:
                doc.update({
                    "sft_quality": "low",
                    "sft_rules": {},
                    "sft_failed_rules": ["missing_fields"]
                })
                continue

            result = self.judge_one(instruction, output)
            doc.update({
                "sft_quality": result["quality"],
                "sft_rules": result["rules"],
                "sft_failed_rules": result["failed_rules"]
            })

            if "error" in result:
                doc["sft_error"] = result["error"]

        return docs