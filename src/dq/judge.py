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

from dq.llm_client import get_client, get_default_model, get_backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule definitions — add new rules here, everything else auto-adapts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeRule:
    """A single evaluation rule.

    mode:
      - "binary"    : LLM answers pass/fail. Internally score ∈ {0, 1}; passes when score == 1.
      - "score"     : LLM answers on a 1..max_score scale. Passes when score >= threshold.
    threshold:
      - binary mode: ignored (always 1).
      - score mode:  minimum score to count as pass (default 3.0 on a 1..5 scale).
    max_score:
      - binary: 1
      - score:  5 (default). Change to 10 or other if you want a wider scale.
    """
    name: str
    description: str
    scope: str  # "universal", "sft", "pretrain"
    mode: str = "binary"      # "binary" | "score"
    threshold: float = 1.0    # pass if score >= threshold
    max_score: int = 1        # binary=1; score mode typically 5 or 10


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

# Preserve built-in defaults so UI can show them and callers can reset.
_DEFAULT_RULES = list(RULES)


def _rebuild_indices() -> None:
    """Recompute RULES_BY_NAME and scope-filtered lists from RULES."""
    global RULES_BY_NAME, RULE_NAMES_UNIVERSAL, RULE_NAMES_SFT, RULE_NAMES_PRETRAIN
    RULES_BY_NAME = {r.name: r for r in RULES}
    RULE_NAMES_UNIVERSAL = [r.name for r in RULES if r.scope == "universal"]
    RULE_NAMES_SFT = [r.name for r in RULES if r.scope in ("universal", "sft")]
    RULE_NAMES_PRETRAIN = [r.name for r in RULES if r.scope in ("universal", "pretrain")]


RULES_BY_NAME: dict[str, JudgeRule] = {}
RULE_NAMES_UNIVERSAL: list[str] = []
RULE_NAMES_SFT: list[str] = []
RULE_NAMES_PRETRAIN: list[str] = []
_rebuild_indices()


def _rule_to_dict(r: JudgeRule) -> dict:
    return {
        "name": r.name,
        "description": r.description,
        "scope": r.scope,
        "mode": r.mode,
        "threshold": r.threshold,
        "max_score": r.max_score,
    }


def get_effective_rules() -> list[dict]:
    """Return the current rule set as dicts (for serialization / UI)."""
    return [_rule_to_dict(r) for r in RULES]


def get_default_rules() -> list[dict]:
    """Return the code-defined defaults (frozen). Used to reset the rule set."""
    return [_rule_to_dict(r) for r in _DEFAULT_RULES]


def apply_rule_overrides(rules: list[dict] | None) -> None:
    """Replace the active rule list. Pass None or [] to reset to defaults.

    Each rule dict must have: name, description, scope ("universal"/"sft"/"pretrain").
    """
    global RULES, RULE_NAMES_SFT, RULE_NAMES_PRETRAIN
    if not rules:
        RULES[:] = list(_DEFAULT_RULES)
    else:
        validated = []
        for r in rules:
            if not r.get("name") or not r.get("description"):
                continue
            scope = r.get("scope", "universal")
            if scope not in ("universal", "sft", "pretrain"):
                scope = "universal"
            mode = r.get("mode", "binary")
            if mode not in ("binary", "score"):
                mode = "binary"
            max_score = int(r.get("max_score", 5 if mode == "score" else 1))
            default_threshold = (max_score + 1) / 2.0 if mode == "score" else 1.0
            threshold = float(r.get("threshold", default_threshold))
            validated.append(JudgeRule(
                name=r["name"], description=r["description"], scope=scope,
                mode=mode, max_score=max_score, threshold=threshold,
            ))
        RULES[:] = validated
    _rebuild_indices()


# ---------------------------------------------------------------------------
# Prompt template — user-editable wrapping around the (locked) JSON schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptTemplate:
    """Editable prompt wrapping. JSON-output schema is not here — it is
    auto-generated from rules to prevent the user from breaking parsing.
    """
    system: str = (
        "You are a quality judge. Evaluate the content against these rules "
        "and return structured JSON."
    )
    rules_header: str = "Rules:"
    input_header_text: str = "Text:"
    input_header_sft_instruction: str = "Instruction:"
    input_header_sft_response: str = "Response:"
    trailer: str = "Respond with ONLY valid JSON in this exact format:"


_DEFAULT_TEMPLATE = PromptTemplate()
_ACTIVE_TEMPLATE: PromptTemplate = _DEFAULT_TEMPLATE


def get_effective_template() -> dict:
    return {k: getattr(_ACTIVE_TEMPLATE, k) for k in _ACTIVE_TEMPLATE.__dataclass_fields__}


def get_default_template() -> dict:
    return {k: getattr(_DEFAULT_TEMPLATE, k) for k in _DEFAULT_TEMPLATE.__dataclass_fields__}


def apply_template_override(tpl: dict | None) -> None:
    """Replace the in-process prompt template. None / empty dict resets to defaults.
    Missing fields keep their default value; extra fields are ignored.
    """
    global _ACTIVE_TEMPLATE
    if not tpl:
        _ACTIVE_TEMPLATE = _DEFAULT_TEMPLATE
        return
    defaults = get_default_template()
    merged = {k: (tpl.get(k) or defaults[k]) for k in defaults}
    _ACTIVE_TEMPLATE = PromptTemplate(**merged)


def _build_prompt(rules: list[JudgeRule], text: str,
                  instruction: str | None = None, output: str | None = None) -> str:
    """Build judge prompt from the active template + auto-generated JSON schema.

    The user can edit the template (system message, headers, etc.) but the
    JSON output contract is locked — that's what keeps parsing reliable.
    """
    tpl = _ACTIVE_TEMPLATE

    rule_lines: list[str] = []
    shape_lines: list[str] = []
    for i, r in enumerate(rules):
        if r.mode == "score":
            rule_lines.append(
                f"{i+1}. {r.name} [score 1..{int(r.max_score)}, pass ≥ {r.threshold}]: {r.description}"
            )
            shape_lines.append(
                f'  "{r.name}": {{"score": <integer 1..{int(r.max_score)}>, "reason": "brief justification"}}'
            )
        else:
            rule_lines.append(f"{i+1}. {r.name} [PASS/FAIL]: {r.description}")
            shape_lines.append(
                f'  "{r.name}": {{"pass": true/false, "reason": "brief explanation if failed"}}'
            )

    rules_text = "\n".join(rule_lines)
    json_example = "{\n" + ",\n".join(shape_lines) + "\n}"

    if instruction is not None and output is not None:
        input_text = f"{tpl.input_header_sft_instruction} {instruction}\n\n{tpl.input_header_sft_response} {output}"
    else:
        input_text = f"{tpl.input_header_text} {text}"

    return (
        f"{tpl.system}\n\n"
        f"{tpl.rules_header}\n{rules_text}\n\n"
        f"{input_text}\n\n"
        f"{tpl.trailer}\n{json_example}"
    )


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

    def _call_api(self, client, prompt: str, backend: str) -> str:
        """Call LLM API and return response text. Supports anthropic and openai backends."""
        # Detect backend from client class name.
        # Only use Anthropic API if client is actually an Anthropic instance.
        # MagicMock and OpenAI clients use the OpenAI chat.completions path.
        client_type = type(client).__name__
        if client_type == "Anthropic":
            response = client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0,
            )
            return response.content[0].text.strip()
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

    def _parse_response(self, response_text: str, expected_rules: list[str]) -> dict[str, dict[str, Any]]:
        """Parse raw LLM output into {rule_name: {pass?, score?, reason}}.

        Applies threshold/mode logic later in _call_llm.
        """
        try:
            data = json.loads(response_text.strip())
            if isinstance(data, dict) and all(r in data for r in expected_rules):
                return data
        except (json.JSONDecodeError, KeyError):
            pass

        logger.warning("JSON parse failed, using regex fallback: %r", response_text[:200])
        rules: dict[str, dict[str, Any]] = {}
        for rule in expected_rules:
            pass_m = re.search(rf'"{rule}":\s*{{[^}}]*"pass":\s*(true|false)', response_text, re.IGNORECASE | re.DOTALL)
            score_m = re.search(rf'"{rule}":\s*{{[^}}]*"score":\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE | re.DOTALL)
            entry: dict[str, Any] = {"reason": ""}
            if score_m:
                entry["score"] = float(score_m.group(1))
            if pass_m:
                entry["pass"] = pass_m.group(1).lower() == "true"
            if not score_m and not pass_m:
                entry = {"pass": False, "reason": "Parse error — defaulted to fail"}
            rules[rule] = entry
        return rules

    def _call_llm(self, prompt: str, rule_objs: list[JudgeRule]) -> dict[str, Any]:
        """Call LLM, parse response, and apply per-rule pass/fail based on rule mode+threshold."""
        expected = [r.name for r in rule_objs]
        client = self._get_client()
        if client is None:
            return {
                "quality": "low", "rules": {},
                "failed_rules": ["api_unavailable"],
                "error": "No API client available",
            }

        backend = get_backend()
        rule_by_name = {r.name: r for r in rule_objs}

        for attempt in range(self.max_retries):
            try:
                text = self._call_api(client, prompt, backend)
                raw = self._parse_response(text, expected)

                rules_out: dict[str, dict[str, Any]] = {}
                for name in expected:
                    r = rule_by_name[name]
                    entry = dict(raw.get(name, {}))
                    if r.mode == "score":
                        try:
                            score = float(entry.get("score", 0))
                        except (TypeError, ValueError):
                            score = 0.0
                        entry["score"] = score
                        entry["mode"] = "score"
                        entry["pass"] = score >= r.threshold
                    else:
                        entry["mode"] = "binary"
                        entry.setdefault("pass", False)
                        entry["score"] = 1.0 if entry["pass"] else 0.0
                    rules_out[name] = entry

                failed = [n for n, v in rules_out.items() if not v.get("pass", False)]
                return {
                    "quality": "high" if not failed else "low",
                    "rules": rules_out,
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
        return self._call_llm(prompt, rule_objs)

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
