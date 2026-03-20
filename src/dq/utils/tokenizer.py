"""Simple tokenizer utilities."""

import re


def simple_tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def char_ngrams(text: str, n: int = 5) -> list[str]:
    """Generate character-level n-grams from text."""
    text = text.lower().strip()
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def word_ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate word-level n-grams."""
    if len(words) < n:
        return []
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
