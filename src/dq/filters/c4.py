"""C4 dataset quality filters.

Implements filters from the C4 (Colossal Clean Crawled Corpus) paper.
"""

import re
from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter

_TERMINAL_PUNCT = re.compile(r"[.!?。！？;；\"'\"\']$")
_JAVASCRIPT_RE = re.compile(r"\bjavascript\b", re.IGNORECASE)
_POLICY_RE = re.compile(
    r"terms of use|privacy policy|cookie policy|use of cookies|"
    r"uses cookies|terms of service|terms and conditions",
    re.IGNORECASE,
)
_LOREM_RE = re.compile(r"lorem ipsum", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?。！？]*[.!?。！？]")


@register_filter("c4")
class C4Filter(BaseFilter):
    """C4 quality filters.

    Applies line-level and document-level cleaning:
    - Remove lines without terminal punctuation
    - Remove lines mentioning javascript
    - Remove lines with policy/cookie language
    - Drop docs with lorem ipsum
    - Drop docs with curly braces (optional, aggressive)
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
        min_sentences: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.remove_no_terminal_punct = remove_no_terminal_punct
        self.remove_javascript_lines = remove_javascript_lines
        self.remove_policy_lines = remove_policy_lines
        self.remove_lorem_ipsum = remove_lorem_ipsum
        self.remove_curly_brace = remove_curly_brace
        self.min_sentences = min_sentences

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)

        # Document-level checks first
        if self.remove_lorem_ipsum and _LOREM_RE.search(text):
            return False, {"filter": self.name, "reason": "lorem_ipsum"}

        if self.remove_curly_brace and "{" in text:
            return False, {"filter": self.name, "reason": "curly_brace"}

        # Line-level filtering
        lines = text.split("\n")
        kept_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                kept_lines.append(line)
                continue
            if self.remove_javascript_lines and _JAVASCRIPT_RE.search(stripped):
                continue
            if self.remove_policy_lines and _POLICY_RE.search(stripped):
                continue
            if self.remove_no_terminal_punct and not _TERMINAL_PUNCT.search(stripped):
                continue
            kept_lines.append(line)

        cleaned = "\n".join(kept_lines).strip()

        if not cleaned:
            return False, {"filter": self.name, "reason": "empty_after_line_filter"}

        # Sentence count check
        sentences = _SENTENCE_RE.findall(cleaned)
        if len(sentences) < self.min_sentences:
            return False, {"filter": self.name, "reason": "too_few_sentences", "value": len(sentences)}

        # Update the doc text with cleaned version
        doc[self.text_field] = cleaned
        return True, {}
