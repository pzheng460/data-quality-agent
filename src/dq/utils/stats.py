"""Document statistics utilities.

All word-level computations are aligned with datatrove's reference implementation
(HuggingFace) to ensure consistent filter results.

Performance note: filters should call get_words() ONCE and pass the result
to all stat functions that need words, to avoid redundant spacy tokenization.
"""

import re
from collections import Counter
from functools import lru_cache

# ── spacy tokenizer (matching datatrove's SpaCyTokenizer) ──────────

_PUNCTUATION_SET: set[str] | None = None


def _get_punctuation_set() -> set[str]:
    """Lazy-load datatrove's PUNCTUATION_SET for non-symbol word filtering."""
    global _PUNCTUATION_SET
    if _PUNCTUATION_SET is None:
        from datatrove.pipeline.filters.gopher_quality_filter import PUNCTUATION_SET
        _PUNCTUATION_SET = PUNCTUATION_SET
    return _PUNCTUATION_SET


@lru_cache(maxsize=1)
def _get_spacy_tokenizer():
    """Load spacy tokenizer (same as datatrove uses for English)."""
    import importlib.metadata  # noqa: F401 — needed by datatrove internals
    from datatrove.utils.word_tokenizers import load_word_tokenizer
    from datatrove.utils.typeshelper import Languages
    return load_word_tokenizer(Languages.english)


@lru_cache(maxsize=8)
def tokenize_words(text: str) -> list[str]:
    """Tokenize text into words using spacy (matching datatrove).

    LRU cache avoids redundant tokenization when multiple filters
    process the same document (e.g. gopher_quality → gopher_repetition).
    Falls back to str.split() if spacy is unavailable.
    """
    try:
        tok = _get_spacy_tokenizer()
        return tok.word_tokenize(text)
    except Exception:
        return text.split()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using spacy (matching datatrove's C4 filter).

    When C4's split_paragraph=True, it uses span_tokenize on each line.
    """
    try:
        tok = _get_spacy_tokenizer()
        spans = tok.span_tokenize(text)
        if not spans:
            return [text] if text.strip() else []
        parts = []
        start = 0
        for _, end in spans:
            parts.append(text[start:end])
            start = end
        if start < len(text):
            parts.append(text[start:])
        return [p for p in parts if p.strip()]
    except Exception:
        # Fallback to regex
        import re as _re
        return _re.findall(r"[^.!?。！？]*[.!?。！？]", text) or ([text] if text.strip() else [])


# ── CJK utilities ──────────────────────────────────────────────────

_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff\u3000-\u303f"
    r"\uf900-\ufaff\U00020000-\U0002a6df\U0002a700-\U0002b73f]"
)


def is_cjk_heavy(text: str, threshold: float = 0.3) -> bool:
    """Check if text is CJK-heavy (more than threshold fraction of chars are CJK)."""
    if not text:
        return False
    cjk_count = len(_CJK_RE.findall(text))
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count == 0:
        return False
    return cjk_count / alpha_count > threshold


def get_words(text: str) -> list[str]:
    """Split text into words using spacy tokenizer (matching datatrove).

    For CJK-heavy text, each CJK char is a word (unchanged).
    """
    if is_cjk_heavy(text):
        words = []
        for segment in re.split(r"(\s+)", text):
            segment = segment.strip()
            if not segment:
                continue
            if _CJK_RE.search(segment):
                tokens = re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]|[a-zA-Z]+", segment)
                words.extend(tokens)
            else:
                words.append(segment)
        return words
    return tokenize_words(text)


def get_non_symbol_words(words: list[str]) -> list[str]:
    """Filter out pure-punctuation tokens (matching datatrove's non_symbol_words).

    A word is a symbol if ALL its characters are in PUNCTUATION_SET.
    """
    punct = _get_punctuation_set()
    return [w for w in words if any(ch not in punct for ch in w)]


# ── Word-level statistics (accept pre-tokenized words) ─────────────

def word_count(text: str | None = None, *, words: list[str] | None = None) -> int:
    """Count non-symbol words (matching datatrove)."""
    if words is None:
        words = get_words(text)
    return len(get_non_symbol_words(words))


def avg_word_length(text: str | None = None, *, words: list[str] | None = None) -> float:
    """Average word length of non-symbol words (matching datatrove)."""
    if words is None:
        words = get_words(text)
    non_symbol = get_non_symbol_words(words)
    if not non_symbol:
        return 0.0
    return sum(len(w) for w in non_symbol) / len(non_symbol)


def alpha_ratio(text: str | None = None, *, words: list[str] | None = None) -> float:
    """Fraction of words containing at least one alphabetic character (matching datatrove)."""
    if words is None:
        words = get_words(text)
    if not words:
        return 0.0
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    return alpha_words / len(words)


def symbol_word_ratio(text: str, *, words: list[str] | None = None) -> tuple[float, float]:
    """Ratio of # symbols and ellipsis to total words (matching datatrove).

    Returns (hash_ratio, ellipsis_ratio).
    """
    if words is None:
        words = get_words(text)
    n_words = len(words)
    if n_words == 0:
        return 0.0, 0.0
    hash_ratio = text.count("#") / n_words
    ellipsis_ratio = (text.count("...") + text.count("\u2026")) / n_words
    return hash_ratio, ellipsis_ratio


def lines_ending_with_punct(text: str) -> float:
    """Fraction of lines ending with terminal punctuation."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    terminal = set(".!?\u3002\uff01\uff1f;\uff1b")
    count = sum(1 for l in lines if l and l[-1] in terminal)
    return count / len(lines)


# Gopher default stop words (matching datatrove exactly)
GOPHER_STOP_WORDS = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})


def count_stopwords(text: str | None = None, stop_words: frozenset[str] | None = None,
                    *, words: list[str] | None = None) -> int:
    """Count unique stop words present in text (matching datatrove).

    Uses set intersection — counts unique stop word types, not occurrences.
    """
    if stop_words is None:
        stop_words = GOPHER_STOP_WORDS
    if words is None:
        words = get_words(text)
    return len(stop_words.intersection(set(words)))


# ── N-gram and repetition statistics ───────────────────────────────

def ngram_counts(words: list[str], n: int) -> Counter:
    """Count n-grams in a list of words."""
    if len(words) < n:
        return Counter()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return Counter(grams)


def top_ngram_ratio(words: list[str], n: int, text: str) -> float:
    """Character coverage of the most frequent n-gram / len(text) (matching datatrove)."""
    if len(words) < n or not text:
        return 0.0
    ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counter = Counter(ngrams)
    top_ngram, top_count = counter.most_common(1)[0]
    top_char_length = len(top_ngram) * top_count
    return top_char_length / len(text)


def duplicate_line_ratio(text: str) -> float:
    """Fraction of duplicate lines (matching datatrove's find_duplicates).

    datatrove's _line_splitter uses regex r"\\n+" which merges consecutive
    newlines (empty lines are excluded). Counts only subsequent occurrences
    as duplicates (first seen = not duplicate).
    """
    lines = re.split(r"\n+", text)
    lines = [l for l in lines if l]
    if not lines:
        return 0.0
    seen: set[str] = set()
    dup_count = 0
    for line in lines:
        if line in seen:
            dup_count += 1
        else:
            seen.add(line)
    return dup_count / len(lines)


def duplicate_paragraph_ratio(text: str) -> float:
    """Fraction of duplicate paragraphs (matching datatrove's find_duplicates).

    datatrove splits on paragraph_exp (r"\\n{2,}") then uses find_duplicates.
    """
    paragraphs = re.split(r"\n{2,}", text.strip())
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return 0.0
    seen: set[str] = set()
    dup_count = 0
    for p in paragraphs:
        if p in seen:
            dup_count += 1
        else:
            seen.add(p)
    return dup_count / len(paragraphs)


def char_repetition_ratio(text: str, n: int = 10) -> float:
    """DEPRECATED: old character-level n-gram implementation. Use dup_ngram_char_frac instead.

    Kept for backward compat in tests. Do not use in filters.
    """
    if len(text) < n:
        return 0.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    counts = Counter(grams)
    repeated_chars = sum((c - 1) * n for c in counts.values() if c > 1)
    return min(repeated_chars / len(text), 1.0)


def dup_ngram_char_frac(words: list[str], n: int, text: str) -> float:
    """Gopher's duplicate word n-gram character fraction (matching datatrove).

    datatrove's find_all_duplicate: walks word n-grams, counts duplicate char
    coverage, divides by len(text) (full text including spaces).
    """
    n_words = len(words)
    if n_words < n or not text:
        return 0.0

    unique: set[str] = set()
    repeated_chars = 0
    idx = 0
    while idx <= n_words - n:
        ngram = "".join(words[idx:idx + n])
        if ngram in unique:
            repeated_chars += len(ngram)
            idx += n
        else:
            unique.add(ngram)
            idx += 1

    return repeated_chars / len(text)
