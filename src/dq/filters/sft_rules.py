"""SFT-specific rule-based filter (Layer 1).

Catches common SFT data quality issues that don't require an LLM:
- Empty outputs
- Outputs too short relative to instruction (with closed-form task awareness)
- Instruction echo/copy
- AI refusal patterns (with content-after-refusal check)
- Language mismatch between instruction and output

References:
- AlpaGasus (Chen et al., 2023) — LLM-based filtering for SFT quality
- DEITA (Liu et al., 2023) — Automatic data selection for instruction tuning
- InsTag (Lu et al., 2023) — Instruction tagging (open-ended vs closed-form)
"""

from __future__ import annotations

import re
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter
from dq.utils.stats import word_count

# Default AI refusal patterns (case-insensitive prefix match)
# These indicate the model refused to answer rather than providing content.
DEFAULT_REFUSAL_PATTERNS = [
    "i cannot",
    "i'm sorry, but i cannot",
    "i'm sorry, but i can't",
    "i apologize, but i cannot",
    "i apologize, but i can't",
    "i'm not able to",
    "i am not able to",
    "i'm unable to",
    "i am unable to",
]

# Patterns that look like refusal but often precede real content.
# These are only flagged if the output is too short to contain real content.
SOFT_REFUSAL_PATTERNS = [
    "as an ai language model",
    "as an ai assistant",
    "as an ai,",
    "i'm sorry, but i",  # Generic sorry — may have content after
    "i apologize",        # Generic apology — may have content after
]

# Instruction patterns indicating closed-form tasks where short output is expected.
# Based on InsTag (Lu et al., 2023) taxonomy of instruction types.
_CLOSED_FORM_PATTERNS = re.compile(
    r"(?i)\b("
    r"classif[y|ication]|categoriz|label|"  # classification
    r"output\s+[01]|output\s+(true|false|yes|no)|"  # binary output
    r"answer\s+(yes|no|true|false)|"
    r"(is|are|was|were)\s+.{1,40}\?$|"  # yes/no questions
    r"name\s+the|list\s+the|identify\s+the|extract\s+the|"  # extraction
    r"what\s+is\s+the\s+(name|number|date|year|country|city|capital)|"  # factoid QA
    r"who\s+(is|was|are|were)\b|"  # person extraction
    r"how\s+many\b|"  # counting
    r"choose\s+(one|the|from|between)|select\s+(one|the|from)|"  # selection
    r"pick\s+(one|the)|which\s+(one|of)\b|"  # selection
    r"translate\s+.{1,20}\s+to\b|"  # translation (can be short)
    r"convert\s+.{1,30}\s+to\b|"  # conversion
    r"spell|abbreviat|acronym|"  # short answers
    r"true\s+or\s+false|yes\s+or\s+no|"  # explicit binary
    r"in\s+one\s+word|in\s+a\s+word|one-word\s+answer"  # explicit short
    r")"
)

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
        min_output_words_closed_form: int = 1,
        min_instruction_words_for_short_check: int = 20,
        max_copy_similarity: float = 0.80,
        refusal_patterns: list[str] | None = None,
        soft_refusal_patterns: list[str] | None = None,
        min_words_after_refusal: int = 50,
        lang_mismatch_threshold: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.instruction_field = instruction_field
        self.output_field = output_field
        self.min_output_words = min_output_words
        self.min_output_words_closed_form = min_output_words_closed_form
        self.min_instruction_words_for_short_check = min_instruction_words_for_short_check
        self.max_copy_similarity = max_copy_similarity
        self.refusal_patterns = [
            p.lower() for p in (refusal_patterns or DEFAULT_REFUSAL_PATTERNS)
        ]
        self.soft_refusal_patterns = [
            p.lower() for p in (soft_refusal_patterns or SOFT_REFUSAL_PATTERNS)
        ]
        self.min_words_after_refusal = min_words_after_refusal
        self.lang_mismatch_threshold = lang_mismatch_threshold

    def _extract_fields(self, doc: dict) -> tuple[str, str, bool]:
        """Extract instruction and output from doc.

        Returns:
            (instruction, output, has_sft_fields) — has_sft_fields is False
            when the doc has no recognizable instruction/output structure.
        """
        instruction = doc.get(self.instruction_field, "") or ""
        output = doc.get(self.output_field, "") or ""

        if instruction or output:
            return instruction, output, True

        # Check common alternative field names before giving up
        for ifield in ("prompt", "input", "question", "query", "human"):
            val = doc.get(ifield, "")
            if val:
                instruction = val
                break
        for ofield in ("response", "answer", "reply", "assistant", "completion"):
            val = doc.get(ofield, "")
            if val:
                output = val
                break

        if instruction or output:
            return instruction, output, True

        # No SFT fields found — this is not SFT data
        return "", "", False

    def _is_closed_form(self, instruction: str) -> bool:
        """Check if instruction expects a short/closed-form answer.

        Based on InsTag (Lu et al., 2023) instruction taxonomy.
        """
        return bool(_CLOSED_FORM_PATTERNS.search(instruction))

    def _is_refusal(self, output: str) -> tuple[bool, str]:
        """Check if output is an AI refusal.

        Hard refusals (e.g., "I cannot do X") always flag.
        Soft refusals (e.g., "As an AI, ...") only flag if output lacks
        substantive content after the refusal prefix.
        """
        out_lower = output.strip().lower()

        # Hard refusal — always reject
        for pattern in self.refusal_patterns:
            if out_lower.startswith(pattern):
                return True, pattern

        # Soft refusal — only reject if no substantial content follows
        for pattern in self.soft_refusal_patterns:
            if out_lower.startswith(pattern):
                # Check if there's real content after the refusal-like opening
                if word_count(output) < self.min_words_after_refusal:
                    return True, pattern
                # Has enough content → not a real refusal, just a preamble
                return False, ""

        return False, ""

    def filter(self, doc: dict) -> tuple[bool, dict]:
        """Return (keep, info) — stops at first failure."""
        instruction, output, has_sft = self._extract_fields(doc)

        # Skip non-SFT data entirely — rules are meaningless without
        # instruction/output structure
        if not has_sft:
            return True, {"filter": self.name, "reason": "skipped_not_sft"}

        # Rule 1: empty_output
        if not output or not output.strip():
            return False, {"filter": self.name, "reason": "empty_output"}

        # Rule 2: output_too_short (with closed-form awareness)
        instr_wc = word_count(instruction)
        out_wc = word_count(output)
        is_closed = self._is_closed_form(instruction)
        min_words = self.min_output_words_closed_form if is_closed else self.min_output_words
        if (instr_wc >= self.min_instruction_words_for_short_check
                and out_wc < min_words):
            return False, {"filter": self.name, "reason": "output_too_short",
                           "instruction_words": instr_wc, "output_words": out_wc,
                           "closed_form": is_closed}

        # Rule 3: instruction_copy
        sim = _simple_similarity(instruction, output)
        if sim > self.max_copy_similarity:
            return False, {"filter": self.name, "reason": "instruction_copy",
                           "similarity": round(sim, 3)}

        # Rule 4: ai_refusal (with content-after-refusal check)
        is_refusal, pattern = self._is_refusal(output)
        if is_refusal:
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
        instruction, output, has_sft = self._extract_fields(doc)

        # Skip non-SFT data entirely
        if not has_sft:
            return True, []

        failures: list[dict] = []

        # Rule 1: empty_output
        is_empty = not output or not output.strip()
        if is_empty:
            failures.append({
                "filter": self.name, "rule": "empty_output",
                "value": 0, "threshold": 1,
            })

        # Rule 2: output_too_short (with closed-form awareness)
        instr_wc = word_count(instruction)
        out_wc = word_count(output)
        is_closed = self._is_closed_form(instruction)
        min_words = self.min_output_words_closed_form if is_closed else self.min_output_words
        if (instr_wc >= self.min_instruction_words_for_short_check
                and out_wc < min_words):
            failures.append({
                "filter": self.name, "rule": "output_too_short",
                "value": out_wc, "threshold": min_words,
            })

        # Rule 3: instruction_copy
        if not is_empty:
            sim = _simple_similarity(instruction, output)
            if sim > self.max_copy_similarity:
                failures.append({
                    "filter": self.name, "rule": "instruction_copy",
                    "value": round(sim, 3), "threshold": self.max_copy_similarity,
                })

        # Rule 4: ai_refusal (with content-after-refusal check)
        if not is_empty:
            is_refusal, pattern = self._is_refusal(output)
            if is_refusal:
                failures.append({
                    "filter": self.name, "rule": "ai_refusal",
                    "value": pattern, "threshold": "prefix_match",
                })

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
