"""Document statistics utilities."""

import re
from collections import Counter


# Regex for CJK characters (Chinese, Japanese, Korean)
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
    """Split text into words. For CJK-heavy text, each CJK char is a word."""
    if is_cjk_heavy(text):
        # For CJK text: split CJK chars individually, keep non-CJK words
        words = []
        for segment in re.split(r"(\s+)", text):
            segment = segment.strip()
            if not segment:
                continue
            if _CJK_RE.search(segment):
                # Extract individual CJK chars and non-CJK tokens
                tokens = re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]|[a-zA-Z]+", segment)
                words.extend(tokens)
            else:
                words.append(segment)
        return words
    return text.split()


def word_count(text: str) -> int:
    """Count words in text."""
    return len(get_words(text))


def avg_word_length(text: str) -> float:
    """Average word length."""
    words = get_words(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def alpha_ratio(text: str) -> float:
    """Ratio of alphabetic (or CJK) characters to total characters."""
    if not text:
        return 0.0
    alpha_count = sum(1 for c in text if c.isalpha())
    total = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    if total == 0:
        return 0.0
    return alpha_count / total


def symbol_word_ratio(text: str) -> float:
    """Ratio of symbol-heavy tokens to total words."""
    words = get_words(text)
    if not words:
        return 0.0
    symbols = {"#", "...", "…"}
    count = sum(1 for w in words if w in symbols)
    return count / len(words)


def lines_ending_with_punct(text: str) -> float:
    """Fraction of lines ending with terminal punctuation."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    terminal = set(".!?。！？;；")
    count = sum(1 for l in lines if l and l[-1] in terminal)
    return count / len(lines)


def count_stopwords(text: str) -> int:
    """Count English stopwords in text."""
    _stopwords = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
    }
    words = text.lower().split()
    return sum(1 for w in words if w in _stopwords)


def ngram_counts(words: list[str], n: int) -> Counter:
    """Count n-grams in a list of words."""
    if len(words) < n:
        return Counter()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return Counter(grams)


def top_ngram_ratio(words: list[str], n: int) -> float:
    """Ratio of the most frequent n-gram count to total n-grams."""
    counts = ngram_counts(words, n)
    if not counts:
        return 0.0
    total = sum(counts.values())
    top = counts.most_common(1)[0][1]
    return top / total


def duplicate_line_ratio(text: str) -> float:
    """Fraction of duplicate lines (by count, not unique)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    dup_count = sum(c for c in counts.values() if c > 1)
    return dup_count / len(lines)


def duplicate_paragraph_ratio(text: str) -> float:
    """Fraction of duplicate paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return 0.0
    counts = Counter(paragraphs)
    dup_count = sum(c for c in counts.values() if c > 1)
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


def dup_ngram_char_frac(words: list[str], n: int) -> float:
    """Gopher's duplicate word n-gram character fraction.

    Walks through word n-grams sequentially. When a duplicate is found,
    its character length is added to the count and the window skips past it.
    Returns the fraction of total joined-text characters covered by duplicates.

    This is the correct implementation per Gopher Table A1 and datatrove's
    GopherRepetitionFilter.find_all_duplicate().
    """
    n_words = len(words)
    if n_words < n:
        return 0.0
    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return 0.0

    unique: set[str] = set()
    repeated_chars = 0
    idx = 0
    while idx <= n_words - n:
        ngram = "".join(words[idx:idx + n])
        if ngram in unique:
            repeated_chars += len(ngram)
            idx += n  # skip past duplicate
        else:
            unique.add(ngram)
            idx += 1

    return repeated_chars / total_chars
