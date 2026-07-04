"""Content n-gram extraction for the L1 "running jokes" channel.

Pure functions only (no I/O): the handler feeds learned message tokens in,
gets back the bigrams/trigrams worth counting in ``chat_hot_ngrams``.
A "content" n-gram contains at least one token that is not a stopword and is
long enough to carry meaning; punctuation-only tokens disqualify an n-gram.
"""
from __future__ import annotations

from app.core.lexicon import STOPWORDS

# A token shorter than this cannot be the content anchor of an n-gram
# (filters "да", "ну", "он" — high-frequency filler the stopword list misses).
MIN_CONTENT_TOKEN_LEN = 3
# Hard cap of n-grams recorded per message: bounds the per-message write batch
# (executemany) regardless of message length.
MAX_NGRAMS_PER_MESSAGE = 24


def _is_wordlike(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def _is_content_token(token: str) -> bool:
    return (
        len(token) >= MIN_CONTENT_TOKEN_LEN
        and token not in STOPWORDS
        and _is_wordlike(token)
    )


def extract_content_ngrams(
    tokens: list[str],
    *,
    max_per_message: int = MAX_NGRAMS_PER_MESSAGE,
) -> list[tuple[str, ...]]:
    """Adjacent bigrams and trigrams that contain at least one content token.

    Preserves first-seen order, dedupes within the message, and caps the total
    so a long message cannot flood the hot-ngram table.
    """
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []
    for size in (2, 3):
        for i in range(len(tokens) - size + 1):
            ngram = tuple(tokens[i : i + size])
            if ngram in seen:
                continue
            if not all(_is_wordlike(token) for token in ngram):
                continue
            if not any(_is_content_token(token) for token in ngram):
                continue
            seen.add(ngram)
            result.append(ngram)
            if len(result) >= max_per_message:
                return result
    return result
