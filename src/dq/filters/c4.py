"""C4 dataset quality filters.

Implements filters from the C4 (Colossal Clean Crawled Corpus) paper.
Aligned with datatrove's C4QualityFilter reference implementation.
"""

import re
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter
from dq.utils.stats import split_sentences

# Aligned with datatrove's constants
_END_PUNCTUATION = (".", "?", "!", '"', "'")
_ELLIPSIS = "..."
_JAVASCRIPT_RE = re.compile(r"\bjavascript\b", re.IGNORECASE)
_POLICY_SUBSTRINGS = [
    "terms of use", "privacy policy", "cookie policy",
    "uses cookies", "use of cookies", "use cookies",
]
_LOREM_RE = re.compile(r"lorem ipsum", re.IGNORECASE)


@register_filter("c4")
class C4Filter(BaseFilter):
    """C4 quality filters (aligned with datatrove).

    Applies line-level and document-level cleaning:
    - Remove lines without terminal punctuation (excluding ellipsis)
    - Remove lines mentioning javascript
    - Remove lines with policy/cookie language
    - Drop docs with lorem ipsum
    - Drop docs with curly braces (optional)
    - Drop docs with fewer than min_sentences
    """

    def __init__(
        self,
        text_field: str = "text",
        remove_no_terminal_punct: bool = True,
        remove_javascript_lines: bool = True,
        remove_policy_lines: bool = True,
        remove_lorem_ipsum: bool = True,
        remove_curly_brace: bool = False,
        min_sentences: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.remove_no_terminal_punct = remove_no_terminal_punct
        self.remove_javascript_lines = remove_javascript_lines
        self.remove_policy_lines = remove_policy_lines
        self.remove_lorem_ipsum = remove_lorem_ipsum
        self.remove_curly_brace = remove_curly_brace
        self.min_sentences = min_sentences

    def _filter_lines(self, text: str) -> tuple[list[str], dict]:
        """Apply line-level filtering, return kept lines and removal stats."""
        lines = text.splitlines()
        kept_lines: list[str] = []
        js_removed = 0
        policy_removed = 0
        no_punct_removed = 0

        for line in lines:
            stripped = line.strip()

            # Terminal punctuation check (matching datatrove order):
            # endswith END_PUNCTUATION but NOT endswith ellipsis
            # MUST come before lorem/curly/javascript/policy so that
            # lines without terminal punct are skipped first.
            if self.remove_no_terminal_punct and stripped:
                if not stripped.endswith(_END_PUNCTUATION) or stripped.endswith(_ELLIPSIS):
                    no_punct_removed += 1
                    continue

            # lorem ipsum: reject entire doc
            if self.remove_lorem_ipsum and "lorem ipsum" in stripped.lower():
                return [], {"reject": "lorem_ipsum"}

            # javascript
            if self.remove_javascript_lines and "javascript" in stripped.lower():
                js_removed += 1
                continue

            # curly brace: reject entire doc
            if self.remove_curly_brace and "{" in stripped:
                return [], {"reject": "curly_brace"}

            # policy substrings (matching datatrove exactly)
            if self.remove_policy_lines:
                lower = stripped.lower()
                if any(p in lower for p in _POLICY_SUBSTRINGS):
                    policy_removed += 1
                    continue

            kept_lines.append(stripped)

        stats = {
            "js_removed": js_removed,
            "policy_removed": policy_removed,
            "no_punct_removed": no_punct_removed,
        }
        return kept_lines, stats

    def _count_sentences(self, kept_lines: list[str]) -> int:
        """Count sentences using spacy (matching datatrove's C4 with split_paragraph=True)."""
        total = 0
        for line in kept_lines:
            if line.strip():
                sents = split_sentences(line)
                total += len(sents)
        return total

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)

        kept_lines, stats = self._filter_lines(text)

        # Document-level rejection
        if "reject" in stats:
            return False, {"filter": self.name, "reason": stats["reject"]}

        if not kept_lines or not "\n".join(kept_lines).strip():
            return False, {"filter": self.name, "reason": "empty_after_line_filter"}

        # Sentence count check (using spacy, matching datatrove)
        num_sentences = self._count_sentences(kept_lines)
        if num_sentences < self.min_sentences:
            return False, {"filter": self.name, "reason": "too_few_sentences", "value": num_sentences}

        # Update the doc text with cleaned version
        doc[self.text_field] = "\n".join(kept_lines).strip()
        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        text = self.get_text(doc)
        failures: list[dict] = []

        kept_lines, stats = self._filter_lines(text)

        if "reject" in stats:
            reason = stats["reject"]
            failures.append({"filter": self.name, "rule": reason, "value": True, "threshold": False})

        if not kept_lines or not "\n".join(kept_lines).strip():
            contributors = []
            if stats.get("js_removed", 0) > 0:
                contributors.append(f"javascript({stats['js_removed']})")
            if stats.get("policy_removed", 0) > 0:
                contributors.append(f"policy({stats['policy_removed']})")
            if stats.get("no_punct_removed", 0) > 0:
                contributors.append(f"no_terminal_punct({stats['no_punct_removed']})")
            detail = ", ".join(contributors) if contributors else "all lines empty"
            failures.append({"filter": self.name, "rule": "empty_after_line_filter", "value": detail, "threshold": "non-empty"})
        else:
            num_sentences = self._count_sentences(kept_lines)
            if num_sentences < self.min_sentences:
                contributors = []
                if stats.get("no_punct_removed", 0) > 0:
                    contributors.append(f"no_terminal_punct_removed({stats['no_punct_removed']})")
                if stats.get("js_removed", 0) > 0:
                    contributors.append(f"javascript_removed({stats['js_removed']})")
                if stats.get("policy_removed", 0) > 0:
                    contributors.append(f"policy_removed({stats['policy_removed']})")
                detail = f"sentences={num_sentences}" + (f" after removing {', '.join(contributors)}" if contributors else "")
                failures.append({"filter": self.name, "rule": "min_sentences", "value": detail, "threshold": self.min_sentences})

        return len(failures) == 0, failures
