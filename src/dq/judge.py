"""Unified LLM Binary Judge — data-driven rule evaluation for any text data.

Rules are declared as data (name, description, scope). The judge auto-selects
applicable rules based on input type (SFT pair vs plain text), builds a prompt,
and parses structured JSON responses.

Usage:
    judge = LLMJudge()
    result = judge.judge_sft("Explain X", "X is...")     # universal + sft rules
    result = judge.judge_text("Article about physics...")  # universal + pretrain rules
    result = judge.judge("custom text", rules=["factuality", "coherence"])  # pick rules
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from dq.llm_client import get_client, get_default_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule definitions — add new rules here, everything else auto-adapts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeRule:
    """A single binary evaluation rule."""
    name: str
    description: str
    scope: str  # "universal", "sft", "pretrain"


# All rules in one place. To add a rule: append here. Done.
RULES: list[JudgeRule] = [
    # Universal — apply to both SFT and pretrain
    JudgeRule(
        name="factuality",
        description="Are there any obvious factual errors? PASS if no errors, FAIL if contains fabricated facts or wrong information.",
        scope="universal",
    ),
    JudgeRule(
        name="coherence",
        description="Is the text well-structured and logically organized? PASS if coherent, FAIL if fragmented, contradictory, or garbled.",
        scope="universal",
    ),
    JudgeRule(
        name="harmlessness",
        description="Is the content safe and appropriate? PASS if safe, FAIL if contains dangerous instructions, discrimination, or harmful content.",
        scope="universal",
    ),
    # SFT-only — require instruction + output
    JudgeRule(
        name="instruction_following",
        description="Does the response directly address what the instruction asks for? PASS if yes, FAIL if off-topic or ignores key requirements.",
        scope="sft",
    ),
    JudgeRule(
        name="completeness",
        description="Does the response cover what the instruction asks for? PASS if complete, FAIL if missing critical steps or incomplete answer.",
        scope="sft",
    ),
    JudgeRule(
        name="format_compliance",
        description="If the instruction specifies a format, does the response follow it? PASS if correct or no format specified, FAIL if wrong format.",
        scope="sft",
    ),
    # Pretrain-only — single text evaluation
    JudgeRule(
        name="information_density",
        description="Does the text contain substantive, informative content? PASS if yes, FAIL if ads, navigation menus, cookie notices, or boilerplate.",
        scope="pretrain",
    ),
    JudgeRule(
        name="originality",
        description="Does the text appear to be original content? PASS if original, FAIL if SEO spam, template-generated, or machine-generated filler.",
        scope="pretrain",
    ),
]

# Index for fast lookup
RULES_BY_NAME: dict[str, JudgeRule] = {r.name: r for r in RULES}
RULE_NAMES_UNIVERSAL = [r.name for r in RULES if r.scope == "universal"]
RULE_NAMES_SFT = [r.name for r in RULES if r.scope in ("universal", "sft")]
RULE_NAMES_PRETRAIN = [r.name for r in RULES if r.scope in ("universal", "pretrain")]


# ---------------------------------------------------------------------------
# Prompt builder — auto-generates from rule definitions
# ---------------------------------------------------------------------------

def _build_prompt(rules: list[JudgeRule], text: str,
                  instruction: str | None = None, output: str | None = None) -> str:
    """Build judge prompt from rule definitions and input."""
    # Rules section
    rules_text = "\n".join(
        f"{i+1}. {r.name}: {r.description}"
        for i, r in enumerate(rules)
    )

    # Input section
    if instruction is not None and output is not None:
        input_text = f"Instruction: {instruction}\n\nResponse: {output}"
    else:
        input_text = f"Text: {text}"

    # Expected JSON format
    json_example = "{\n" + ",\n".join(
        f'  "{r.name}": {{"pass": true/false, "reason": "brief explanation if failed"}}'
        for r in rules
    ) + "\n}"

    return f"""You are a quality judge. Evaluate the content against these rules and return structured JSON.

Rules:
{rules_text}

{input_text}

Respond with ONLY valid JSON in this exact format:
{json_example}"""


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------

class LLMJudge:
    """Unified binary quality judge for any text data.

    Auto-selects rules based on input type. Rules are data-driven —
    add new rules to RULES list above, no code changes needed.

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

    def _parse_response(self, response_text: str, expected_rules: list[str]) -> dict[str, dict[str, Any]]:
        """Parse rule results from LLM response."""
        # Try direct JSON
        try:
            data = json.loads(response_text.strip())
            if isinstance(data, dict) and all(r in data for r in expected_rules):
                return data
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: regex extraction
        logger.warning("JSON parse failed, using regex fallback: %r", response_text[:200])
        rules = {}
        for rule in expected_rules:
            pattern = rf'"{rule}":\s*{{\s*"pass":\s*(true|false)'
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                rules[rule] = {"pass": match.group(1).lower() == "true", "reason": ""}
            else:
                rules[rule] = {"pass": False, "reason": "Parse error — defaulted to fail"}
        return rules

    def _call_llm(self, prompt: str, expected_rules: list[str]) -> dict[str, Any]:
        """Call LLM and parse response. Handles retries."""
        client = self._get_client()
        if client is None:
            return {
                "quality": "low", "rules": {},
                "failed_rules": ["api_unavailable"],
                "error": "No API client available",
            }

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.0,
                )
                text = response.choices[0].message.content.strip()
                rules = self._parse_response(text, expected_rules)

                failed = [r for r, v in rules.items() if not v.get("pass", False)]
                return {
                    "quality": "high" if not failed else "low",
                    "rules": rules,
                    "failed_rules": failed,
                }

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning("Judge API call failed (attempt %d): %s", attempt + 1, e)
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("Judge API call failed after %d retries: %s", self.max_retries, e)
                    return {
                        "quality": "low", "rules": {},
                        "failed_rules": ["api_error"],
                        "error": str(e),
                    }

        return {"quality": "low", "rules": {}, "failed_rules": ["unknown_error"]}

    # --- Public API ---

    def judge(self, text: str, rules: list[str] | None = None,
              instruction: str | None = None, output: str | None = None) -> dict[str, Any]:
        """Judge with explicit rule selection.

        Args:
            text: Plain text (for pretrain) or fallback.
            rules: List of rule names to evaluate. If None, auto-selects.
            instruction: SFT instruction (optional).
            output: SFT response (optional).

        Returns:
            {"quality": "high"|"low", "rules": {...}, "failed_rules": [...]}
        """
        if rules is None:
            rule_names = RULE_NAMES_SFT if instruction is not None else RULE_NAMES_PRETRAIN
        else:
            rule_names = rules

        rule_objs = [RULES_BY_NAME[r] for r in rule_names if r in RULES_BY_NAME]
        if not rule_objs:
            return {"quality": "high", "rules": {}, "failed_rules": []}

        # Truncate to avoid token limits
        text = (text or "")[:4000]
        if instruction is not None:
            instruction = instruction[:4000]
        if output is not None:
            output = output[:4000]

        prompt = _build_prompt(rule_objs, text, instruction, output)
        return self._call_llm(prompt, [r.name for r in rule_objs])

    def judge_sft(self, instruction: str, output: str) -> dict[str, Any]:
        """Judge an SFT instruction-response pair (universal + sft rules)."""
        return self.judge(
            text=f"{instruction}\n{output}",
            rules=RULE_NAMES_SFT,
            instruction=instruction,
            output=output,
        )

    def judge_text(self, text: str) -> dict[str, Any]:
        """Judge a plain text document (universal + pretrain rules)."""
        return self.judge(text=text, rules=RULE_NAMES_PRETRAIN)


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------

class SFTQualityJudge(LLMJudge):
    """Backward-compatible alias. Use LLMJudge directly for new code."""

    def judge_one(self, instruction: str, output: str) -> dict[str, Any]:
        return self.judge_sft(instruction, output)

    def judge_batch(self, docs: list[dict], instruction_field: str = "instruction",
                    output_field: str = "output") -> list[dict]:
        for doc in docs:
            instr = doc.get(instruction_field, "")
            out = doc.get(output_field, "")
            if not instr or not out:
                doc.update({"sft_quality": "low", "sft_rules": {},
                            "sft_failed_rules": ["missing_fields"]})
                continue
            result = self.judge_one(instr, out)
            doc.update({"sft_quality": result["quality"], "sft_rules": result["rules"],
                        "sft_failed_rules": result["failed_rules"]})
            if "error" in result:
                doc["sft_error"] = result["error"]
        return docs


class PretrainingQualityJudge(LLMJudge):
    """Backward-compatible alias. Use LLMJudge directly for new code."""

    def judge_one(self, text: str) -> dict[str, Any]:
        return self.judge_text(text)

    def judge_batch(self, docs: list[dict], text_field: str = "text") -> list[dict]:
        for doc in docs:
            text = doc.get(text_field, "")
            if not text.strip():
                doc.update({"pretrain_quality": "low", "pretrain_rules": {},
                            "pretrain_failed_rules": ["empty_text"]})
                continue
            result = self.judge_one(text)
            doc.update({"pretrain_quality": result["quality"], "pretrain_rules": result["rules"],
                        "pretrain_failed_rules": result["failed_rules"]})
            if "error" in result:
                doc["pretrain_error"] = result["error"]
        return docs
