"""Gopher quality and repetition filters.

Implements all rules from the Gopher paper (Rae et al., 2021).
Aligned with datatrove's reference implementation.
"""

from typing import Any

from dq.stages.curation.filters.base import BaseFilter
from dq.pipeline import register_filter
from dq.utils.stats import (
    alpha_ratio,
    avg_word_length,
    bullet_lines_ratio,
    count_stopwords,
    dup_ngram_char_frac,
    duplicate_line_char_frac,
    duplicate_line_ratio,
    duplicate_paragraph_char_frac,
    duplicate_paragraph_ratio,
    ellipsis_lines_ratio,
    get_words,
    symbol_word_ratio,
    top_ngram_ratio,
    word_count,
    GOPHER_STOP_WORDS,
)


@register_filter("gopher_quality")
class GopherQualityFilter(BaseFilter):
    """Gopher paper quality heuristics (aligned with datatrove).

    Tokenizes text ONCE and reuses the word list for all checks.
    """

    def __init__(
        self,
        text_field: str = "text",
        min_words: int = 50,
        max_words: int = 100_000,
        min_avg_word_len: float = 3.0,
        max_avg_word_len: float = 10.0,
        max_symbol_ratio: float = 0.1,
        max_bullet_lines_ratio: float = 0.9,
        max_ellipsis_lines_ratio: float = 0.3,
        min_stopwords: int = 2,
        min_alpha_ratio: float = 0.8,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.min_words = min_words
        self.max_words = max_words
        self.min_avg_word_len = min_avg_word_len
        self.max_avg_word_len = max_avg_word_len
        self.max_symbol_ratio = max_symbol_ratio
        self.max_bullet_lines_ratio = max_bullet_lines_ratio
        self.max_ellipsis_lines_ratio = max_ellipsis_lines_ratio
        self.min_stopwords = min_stopwords
        self.min_alpha_ratio = min_alpha_ratio

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        words = get_words(text)  # tokenize ONCE

        wc = word_count(words=words)
        if wc < self.min_words:
            return False, {"filter": self.name, "reason": "too_few_words", "value": wc}
        if wc > self.max_words:
            return False, {"filter": self.name, "reason": "too_many_words", "value": wc}

        awl = avg_word_length(words=words)
        if awl < self.min_avg_word_len:
            return False, {"filter": self.name, "reason": "avg_word_len_too_short", "value": awl}
        if awl > self.max_avg_word_len:
            return False, {"filter": self.name, "reason": "avg_word_len_too_long", "value": awl}

        hash_ratio, ellipsis_ratio_val = symbol_word_ratio(text, words=words)
        if hash_ratio > self.max_symbol_ratio:
            return False, {"filter": self.name, "reason": "high_hash_ratio", "value": hash_ratio}
        if ellipsis_ratio_val > self.max_symbol_ratio:
            return False, {"filter": self.name, "reason": "high_ellipsis_ratio", "value": ellipsis_ratio_val}

        blr = bullet_lines_ratio(text)
        if blr > self.max_bullet_lines_ratio:
            return False, {"filter": self.name, "reason": "too_many_bullets", "value": blr}

        elr = ellipsis_lines_ratio(text)
        if elr > self.max_ellipsis_lines_ratio:
            return False, {"filter": self.name, "reason": "too_many_end_ellipsis", "value": elr}

        ar = alpha_ratio(words=words)
        if ar < self.min_alpha_ratio:
            return False, {"filter": self.name, "reason": "low_alpha_ratio", "value": ar}

        sw = count_stopwords(words=words)
        if sw < self.min_stopwords:
            return False, {"filter": self.name, "reason": "too_few_stopwords", "value": sw}

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        text = self.get_text(doc)
        words = get_words(text)  # tokenize ONCE
        failures: list[dict] = []

        wc = word_count(words=words)
        if wc < self.min_words:
            failures.append({"filter": self.name, "rule": "min_words", "value": wc, "threshold": self.min_words})
        if wc > self.max_words:
            failures.append({"filter": self.name, "rule": "max_words", "value": wc, "threshold": self.max_words})

        awl = avg_word_length(words=words)
        if awl < self.min_avg_word_len:
            failures.append({"filter": self.name, "rule": "min_avg_word_len", "value": awl, "threshold": self.min_avg_word_len})
        if awl > self.max_avg_word_len:
            failures.append({"filter": self.name, "rule": "max_avg_word_len", "value": awl, "threshold": self.max_avg_word_len})

        hash_ratio, ellipsis_ratio_val = symbol_word_ratio(text, words=words)
        if hash_ratio > self.max_symbol_ratio:
            failures.append({"filter": self.name, "rule": "hash_ratio", "value": hash_ratio, "threshold": self.max_symbol_ratio})
        if ellipsis_ratio_val > self.max_symbol_ratio:
            failures.append({"filter": self.name, "rule": "ellipsis_ratio", "value": ellipsis_ratio_val, "threshold": self.max_symbol_ratio})

        blr = bullet_lines_ratio(text)
        if blr > self.max_bullet_lines_ratio:
            failures.append({"filter": self.name, "rule": "bullet_lines_ratio", "value": blr, "threshold": self.max_bullet_lines_ratio})

        elr = ellipsis_lines_ratio(text)
        if elr > self.max_ellipsis_lines_ratio:
            failures.append({"filter": self.name, "rule": "ellipsis_lines_ratio", "value": elr, "threshold": self.max_ellipsis_lines_ratio})

        ar = alpha_ratio(words=words)
        if ar < self.min_alpha_ratio:
            failures.append({"filter": self.name, "rule": "alpha_ratio", "value": ar, "threshold": self.min_alpha_ratio})

        sw = count_stopwords(words=words)
        if sw < self.min_stopwords:
            failures.append({"filter": self.name, "rule": "stopwords", "value": sw, "threshold": self.min_stopwords})

        return len(failures) == 0, failures


@register_filter("gopher_repetition")
class GopherRepetitionFilter(BaseFilter):
    """Gopher paper repetition-based filters (aligned with datatrove).

    Tokenizes text ONCE and reuses the word list for all checks.
    """

    def __init__(
        self,
        text_field: str = "text",
        max_top_2gram: float = 0.20,
        max_top_3gram: float = 0.18,
        max_top_4gram: float = 0.16,
        max_dup_line_ratio: float = 0.30,
        max_dup_para_ratio: float = 0.30,
        max_dup_line_char_frac: float = 0.20,
        max_dup_para_char_frac: float = 0.20,
        max_dup_5gram_frac: float = 0.15,
        max_dup_6gram_frac: float = 0.14,
        max_dup_7gram_frac: float = 0.13,
        max_dup_8gram_frac: float = 0.12,
        max_dup_9gram_frac: float = 0.11,
        max_dup_10gram_frac: float = 0.10,
        max_char_repetition: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.max_top_2gram = max_top_2gram
        self.max_top_3gram = max_top_3gram
        self.max_top_4gram = max_top_4gram
        self.max_dup_line_ratio = max_dup_line_ratio
        self.max_dup_para_ratio = max_dup_para_ratio
        self.max_dup_line_char_frac = max_dup_line_char_frac
        self.max_dup_para_char_frac = max_dup_para_char_frac
        self.dup_ngram_thresholds = [
            (5, max_dup_5gram_frac),
            (6, max_dup_6gram_frac),
            (7, max_dup_7gram_frac),
            (8, max_dup_8gram_frac),
            (9, max_dup_9gram_frac),
            (10, max_dup_10gram_frac),
        ]
        self.max_char_repetition = max_char_repetition

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        words = get_words(text)  # tokenize ONCE

        dpr = duplicate_paragraph_ratio(text)
        if dpr > self.max_dup_para_ratio:
            return False, {"filter": self.name, "reason": "high_dup_para_ratio", "value": dpr}

        dpcf = duplicate_paragraph_char_frac(text)
        if dpcf > self.max_dup_para_char_frac:
            return False, {"filter": self.name, "reason": "high_dup_para_char_frac", "value": dpcf}

        dlr = duplicate_line_ratio(text)
        if dlr > self.max_dup_line_ratio:
            return False, {"filter": self.name, "reason": "high_dup_line_ratio", "value": dlr}

        dlcf = duplicate_line_char_frac(text)
        if dlcf > self.max_dup_line_char_frac:
            return False, {"filter": self.name, "reason": "high_dup_line_char_frac", "value": dlcf}

        for n, threshold, label in [
            (2, self.max_top_2gram, "top_2gram"),
            (3, self.max_top_3gram, "top_3gram"),
            (4, self.max_top_4gram, "top_4gram"),
        ]:
            ratio = top_ngram_ratio(words, n, text)
            if ratio > threshold:
                return False, {"filter": self.name, "reason": f"high_{label}", "value": ratio}

        for n, threshold in self.dup_ngram_thresholds:
            frac = dup_ngram_char_frac(words, n, text)
            if frac > threshold:
                return False, {"filter": self.name, "reason": f"dup_{n}gram_frac", "value": frac}

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        text = self.get_text(doc)
        words = get_words(text)  # tokenize ONCE
        failures: list[dict] = []

        dpr = duplicate_paragraph_ratio(text)
        if dpr > self.max_dup_para_ratio:
            failures.append({"filter": self.name, "rule": "dup_para_ratio", "value": dpr, "threshold": self.max_dup_para_ratio})

        dpcf = duplicate_paragraph_char_frac(text)
        if dpcf > self.max_dup_para_char_frac:
            failures.append({"filter": self.name, "rule": "dup_para_char_frac", "value": dpcf, "threshold": self.max_dup_para_char_frac})

        dlr = duplicate_line_ratio(text)
        if dlr > self.max_dup_line_ratio:
            failures.append({"filter": self.name, "rule": "dup_line_ratio", "value": dlr, "threshold": self.max_dup_line_ratio})

        dlcf = duplicate_line_char_frac(text)
        if dlcf > self.max_dup_line_char_frac:
            failures.append({"filter": self.name, "rule": "dup_line_char_frac", "value": dlcf, "threshold": self.max_dup_line_char_frac})

        for n, threshold, label in [
            (2, self.max_top_2gram, "top_2gram"),
            (3, self.max_top_3gram, "top_3gram"),
            (4, self.max_top_4gram, "top_4gram"),
        ]:
            ratio = top_ngram_ratio(words, n, text)
            if ratio > threshold:
                failures.append({"filter": self.name, "rule": label, "value": ratio, "threshold": threshold})

        for n, threshold in self.dup_ngram_thresholds:
            frac = dup_ngram_char_frac(words, n, text)
            if frac > threshold:
                failures.append({"filter": self.name, "rule": f"dup_{n}gram_frac", "value": frac, "threshold": threshold})

        return len(failures) == 0, failures
