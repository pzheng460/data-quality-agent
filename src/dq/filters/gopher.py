"""Gopher quality and repetition filters.

Implements all rules from the Gopher paper (Rae et al., 2021).
"""

from typing import Any

from dq.filters.base import BaseFilter
from dq.pipeline import register_filter
from dq.utils.stats import (
    alpha_ratio,
    avg_word_length,
    count_stopwords,
    dup_ngram_char_frac,
    duplicate_line_ratio,
    duplicate_paragraph_ratio,
    get_words,
    lines_ending_with_punct,
    symbol_word_ratio,
    top_ngram_ratio,
    word_count,
)


@register_filter("gopher_quality")
class GopherQualityFilter(BaseFilter):
    """Gopher paper quality heuristics.

    Checks word count, average word length, symbol ratio,
    terminal punctuation ratio, stopword count, and alpha ratio.
    """

    def __init__(
        self,
        text_field: str = "text",
        min_words: int = 50,
        max_words: int = 100_000,
        min_avg_word_len: float = 3.0,
        max_avg_word_len: float = 10.0,
        max_symbol_ratio: float = 0.1,
        min_lines_end_punct: float = 0.1,
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
        self.min_lines_end_punct = min_lines_end_punct
        self.min_stopwords = min_stopwords
        self.min_alpha_ratio = min_alpha_ratio

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)

        wc = word_count(text)
        if wc < self.min_words:
            return False, {"filter": self.name, "reason": "too_few_words", "value": wc}
        if wc > self.max_words:
            return False, {"filter": self.name, "reason": "too_many_words", "value": wc}

        awl = avg_word_length(text)
        if awl < self.min_avg_word_len:
            return False, {"filter": self.name, "reason": "avg_word_len_too_short", "value": awl}
        if awl > self.max_avg_word_len:
            return False, {"filter": self.name, "reason": "avg_word_len_too_long", "value": awl}

        sr = symbol_word_ratio(text)
        if sr > self.max_symbol_ratio:
            return False, {"filter": self.name, "reason": "high_symbol_ratio", "value": sr}

        lep = lines_ending_with_punct(text)
        if lep < self.min_lines_end_punct:
            return False, {"filter": self.name, "reason": "low_terminal_punct", "value": lep}

        sw = count_stopwords(text)
        if sw < self.min_stopwords:
            return False, {"filter": self.name, "reason": "too_few_stopwords", "value": sw}

        ar = alpha_ratio(text)
        if ar < self.min_alpha_ratio:
            return False, {"filter": self.name, "reason": "low_alpha_ratio", "value": ar}

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        text = self.get_text(doc)
        failures: list[dict] = []

        wc = word_count(text)
        if wc < self.min_words:
            failures.append({"filter": self.name, "rule": "min_words", "value": wc, "threshold": self.min_words})
        if wc > self.max_words:
            failures.append({"filter": self.name, "rule": "max_words", "value": wc, "threshold": self.max_words})

        awl = avg_word_length(text)
        if awl < self.min_avg_word_len:
            failures.append({"filter": self.name, "rule": "min_avg_word_len", "value": awl, "threshold": self.min_avg_word_len})
        if awl > self.max_avg_word_len:
            failures.append({"filter": self.name, "rule": "max_avg_word_len", "value": awl, "threshold": self.max_avg_word_len})

        sr = symbol_word_ratio(text)
        if sr > self.max_symbol_ratio:
            failures.append({"filter": self.name, "rule": "symbol_ratio", "value": sr, "threshold": self.max_symbol_ratio})

        lep = lines_ending_with_punct(text)
        if lep < self.min_lines_end_punct:
            failures.append({"filter": self.name, "rule": "lines_end_punct", "value": lep, "threshold": self.min_lines_end_punct})

        sw = count_stopwords(text)
        if sw < self.min_stopwords:
            failures.append({"filter": self.name, "rule": "stopwords", "value": sw, "threshold": self.min_stopwords})

        ar = alpha_ratio(text)
        if ar < self.min_alpha_ratio:
            failures.append({"filter": self.name, "rule": "alpha_ratio", "value": ar, "threshold": self.min_alpha_ratio})

        return len(failures) == 0, failures


@register_filter("gopher_repetition")
class GopherRepetitionFilter(BaseFilter):
    """Gopher paper repetition-based filters.

    Checks top n-gram ratios, duplicate line/paragraph ratios,
    and character-level repetition.
    """

    def __init__(
        self,
        text_field: str = "text",
        max_top_2gram: float = 0.20,
        max_top_3gram: float = 0.18,
        max_top_4gram: float = 0.16,
        max_dup_line_ratio: float = 0.30,
        max_dup_para_ratio: float = 0.30,
        # Gopher Table A1: duplicate {n}-gram character fraction thresholds
        max_dup_5gram_frac: float = 0.15,
        max_dup_6gram_frac: float = 0.14,
        max_dup_7gram_frac: float = 0.13,
        max_dup_8gram_frac: float = 0.12,
        max_dup_9gram_frac: float = 0.11,
        max_dup_10gram_frac: float = 0.10,
        # Deprecated: kept for config backward compat, no longer used in filtering
        max_char_repetition: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(text_field=text_field, **kwargs)
        self.max_top_2gram = max_top_2gram
        self.max_top_3gram = max_top_3gram
        self.max_top_4gram = max_top_4gram
        self.max_dup_line_ratio = max_dup_line_ratio
        self.max_dup_para_ratio = max_dup_para_ratio
        self.dup_ngram_thresholds = [
            (5, max_dup_5gram_frac),
            (6, max_dup_6gram_frac),
            (7, max_dup_7gram_frac),
            (8, max_dup_8gram_frac),
            (9, max_dup_9gram_frac),
            (10, max_dup_10gram_frac),
        ]
        self.max_char_repetition = max_char_repetition  # deprecated, kept for compat

    def filter(self, doc: dict) -> tuple[bool, dict]:
        text = self.get_text(doc)
        words = get_words(text)

        # Top n-gram checks (word-level: most frequent n-gram's char coverage)
        for n, threshold, label in [
            (2, self.max_top_2gram, "top_2gram"),
            (3, self.max_top_3gram, "top_3gram"),
            (4, self.max_top_4gram, "top_4gram"),
        ]:
            ratio = top_ngram_ratio(words, n)
            if ratio > threshold:
                return False, {"filter": self.name, "reason": f"high_{label}", "value": ratio}

        # Duplicate line ratio
        dlr = duplicate_line_ratio(text)
        if dlr > self.max_dup_line_ratio:
            return False, {"filter": self.name, "reason": "high_dup_line_ratio", "value": dlr}

        # Duplicate paragraph ratio
        dpr = duplicate_paragraph_ratio(text)
        if dpr > self.max_dup_para_ratio:
            return False, {"filter": self.name, "reason": "high_dup_para_ratio", "value": dpr}

        # Duplicate word n-gram character fraction (Gopher Table A1)
        for n, threshold in self.dup_ngram_thresholds:
            frac = dup_ngram_char_frac(words, n)
            if frac > threshold:
                return False, {"filter": self.name, "reason": f"dup_{n}gram_frac", "value": frac}

        return True, {}

    def filter_detailed(self, doc: dict) -> tuple[bool, list[dict]]:
        text = self.get_text(doc)
        words = get_words(text)
        failures: list[dict] = []

        for n, threshold, label in [
            (2, self.max_top_2gram, "top_2gram"),
            (3, self.max_top_3gram, "top_3gram"),
            (4, self.max_top_4gram, "top_4gram"),
        ]:
            ratio = top_ngram_ratio(words, n)
            if ratio > threshold:
                failures.append({"filter": self.name, "rule": label, "value": ratio, "threshold": threshold})

        dlr = duplicate_line_ratio(text)
        if dlr > self.max_dup_line_ratio:
            failures.append({"filter": self.name, "rule": "dup_line_ratio", "value": dlr, "threshold": self.max_dup_line_ratio})

        dpr = duplicate_paragraph_ratio(text)
        if dpr > self.max_dup_para_ratio:
            failures.append({"filter": self.name, "rule": "dup_para_ratio", "value": dpr, "threshold": self.max_dup_para_ratio})

        # Duplicate word n-gram character fraction (Gopher Table A1)
        for n, threshold in self.dup_ngram_thresholds:
            frac = dup_ngram_char_frac(words, n)
            if frac > threshold:
                failures.append({"filter": self.name, "rule": f"dup_{n}gram_frac", "value": frac, "threshold": threshold})

        return len(failures) == 0, failures
