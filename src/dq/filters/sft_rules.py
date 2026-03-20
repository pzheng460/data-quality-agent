"""SFT-specific rule-based filter (Layer 1).

Catches common SFT data quality issues that don't require an LLM:
- Empty outputs
- Outputs too short relative to instruction
- Instruction echo/copy
- AI refusal patterns
- Language mismatch between instruction and output
"""

from __future__ import annotations

import re
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter
from dq.utils.stats import word_count

# Default AI refusal patterns (case-insensitive prefix match)
DEFAULT_REFUSAL_PATTERNS = [
    "as an ai language model",
    "as an ai assistant",
    "i cannot",
    "i'm sorry, but i",
    "i apologize",
    "i'm not able to",
    "i am not able to",
    "i'm unable to",
    "i am unable to",
]

# Regex for CJK characters
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff"
    r"\uf900-\ufaff\U00020000-\U0002a6df]"
)


def _cjk_ratio(text: str) -> float:
    """Fraction of alphabetic chars that are CJK."""
    if not text:
        return 0.0
    cjk = len(_CJK_RE.findall(text))
    # Count all alphabetic chars (includes CJK since str.isalpha() is True for CJK)
    alpha = sum(1 for c in text if c.isalpha())
    if alpha == 0:
        return 0.0
    return cjk / alpha


def _simple_similarity(a: str, b: str) -> float:
    """Character-level Jaccard-ish overlap between two strings."""
    if not a or not b:
        return 0.0
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    if not a_lower or not b_lower:
        return 0.0
    # Use character n-gram (3-gram) overlap for robustness
    def char_ngrams(s: str, n: int = 3) -> set[str]:
        return {s[i:i+n] for i in range(max(0, len(s) - n + 1))}

    a_grams = char_ngrams(a_lower)
    b_grams = char_ngrams(b_lower)
    if not a_grams or not b_grams:
        return 1.0 if a_lower == b_lower else 0.0
    intersection = len(a_grams & b_grams)
    union = len(a_grams | b_grams)
    return intersection / union if union > 0 else 0.0


@register_filter("sft_rules")
class SFTRulesFilter(BaseFilter):
    """SFT-specific rule-based filter (Layer 1).

    Checks for common SFT data quality issues without requiring an LLM.
    Works on docs with instruction/output fields OR merged text field.

    Args:
        text_field: Field containing merged text (fallback).
        instruction_field: Field containing the instruction.
        output_field: Field containing the output/response.
        min_output_words: Minimum output words when instruction is long.
        min_instruction_words_for_short_check: Instruction must be this long
            to trigger the output_too_short rule.
        max_copy_similarity: Max similarity between instruction and output.
        refusal_patterns: List of AI refusal prefixes (case-insensitive).
        lang_mismatch_threshold: CJK ratio difference to flag mismatch.
    """

    def __init__(
        self,
        text_field: str = "text",
        instruction_field: str = "instruction",
        output_field: str = "output",
        min_output_words: int = 5,
        min_instruction_words_for_short_check: int = 20,
        max_copy_similarity: float = 0.80,
        refusal_patterns: list[str] | None = None,
        lang_mismatch_threshold: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.instruction_field = instruction_field
        self.output_field = output_field
        self.min_output_words = min_output_words
        self.min_instruction_words_for_short_check = min_instruction_words_for_short_check
        self.max_copy_similarity = max_copy_similarity
        self.refusal_patterns = [
            p.lower() for p in (refusal_patterns or DEFAULT_REFUSAL_PATTERNS)
        ]
        self.lang_mismatch_threshold = lang_mismatch_threshold

    def _extract_fields(self, doc: dict) -> tuple[str, str]:
        """Extract instruction and output from doc."""
        instruction = doc.get(self.instruction_field, "") or ""
        output = doc.get(self.output_field, "") or ""

        if instruction and output:
            return instruction, output

        # Fallback: split merged text on first newline
        text = self.get_text(doc)
        if text:
            parts = text.split("\n", 1)
            instruction = parts[0].strip()
            output = parts[1].strip() if len(parts) > 1 else ""

        return instruction, output

    def filter(self, doc: dict) -> tuple[bool, dict]:
        """Return (keep, info) — stops at first failure."""
        instruction, output = self._extract_fields(doc)

        # Rule 1: empty_output
        if not output or not output.strip():
            return False, {"filter": self.name, "reason": "empty_output"}

        # Rule 2: output_too_short
        instr_wc = word_count(instruction)
        out_wc = word_count(output)
        if (instr_wc >= self.min_instruction_words_for_short_check
                and out_wc < self.min_output_words):
            return False, {"filter": self.name, "reason": "output_too_short",
                           "instruction_words": instr_wc, "output_words": out_wc}

        # Rule 3: instruction_copy
        sim = _simple_similarity(instruction, output)
        if sim > self.max_copy_similarity:
            return False, {"filter": self.name, "reason": "instruction_copy",
                           "similarity": round(sim, 3)}

        # Rule 4: ai_refusal
        out_lower = output.strip().lower()
        for pattern in self.refusal_patterns:
            if out_lower.startswith(pattern):
                return False, {"filter": self.name, "reason": "ai_refusal",
                               "pattern": pattern}

        # Rule 5: language_mismatch
        instr_cjk = _cjk_ratio(instruction)
        out_cjk = _cjk_ratio(output)
        if (abs(instr_cjk - out_cjk) > self.lang_mismatch_threshold
                and instruction and output):
            return False, {"filter": self.name, "reason": "language_mismatch",
                           "instruction_cjk": round(instr_cjk, 2),
                           "output_cjk": round(out_cjk, 2)}

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        """Check all rules and return all failures."""
        instruction, output = self._extract_fields(doc)
        failures: list[dict] = []

        # Rule 1: empty_output
        is_empty = not output or not output.strip()
        if is_empty:
            failures.append({
                "filter": self.name, "rule": "empty_output",
                "value": 0, "threshold": 1,
            })

        # Rule 2: output_too_short
        instr_wc = word_count(instruction)
        out_wc = word_count(output)
        if (instr_wc >= self.min_instruction_words_for_short_check
                and out_wc < self.min_output_words):
            failures.append({
                "filter": self.name, "rule": "output_too_short",
                "value": out_wc, "threshold": self.min_output_words,
            })

        # Rule 3: instruction_copy
        if not is_empty:
            sim = _simple_similarity(instruction, output)
            if sim > self.max_copy_similarity:
                failures.append({
                    "filter": self.name, "rule": "instruction_copy",
                    "value": round(sim, 3), "threshold": self.max_copy_similarity,
                })

        # Rule 4: ai_refusal
        if not is_empty:
            out_lower = output.strip().lower()
            for pattern in self.refusal_patterns:
                if out_lower.startswith(pattern):
                    failures.append({
                        "filter": self.name, "rule": "ai_refusal",
                        "value": pattern, "threshold": "prefix_match",
                    })
                    break  # One refusal match is enough

        # Rule 5: language_mismatch
        if instruction and output and not is_empty:
            instr_cjk = _cjk_ratio(instruction)
            out_cjk = _cjk_ratio(output)
            if abs(instr_cjk - out_cjk) > self.lang_mismatch_threshold:
                failures.append({
                    "filter": self.name, "rule": "language_mismatch",
                    "value": round(abs(instr_cjk - out_cjk), 2),
                    "threshold": self.lang_mismatch_threshold,
                })

        return len(failures) == 0, failures
