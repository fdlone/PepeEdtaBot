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
    # casefold: STOPWORDS are lowercase but tokens keep their case in the
    # case-preserved profile (normalize_lower=false).
    return (
        len(token) >= MIN_CONTENT_TOKEN_LEN
        and token.casefold() not in STOPWORDS
        and _is_wordlike(token)
    )


def is_content_ngram(ngram: tuple[str, ...]) -> bool:
    """Whether an adjacent n-gram counts as a content phrase.

    Named and public because the phrase index derives the same phrases from the
    stored chain rather than from the message stream, and the two must not drift
    apart on what a phrase is (generation-phrase-index spec). Previously this
    lived inline in the loop below, where nothing else could reach it.

    Punctuation disqualifies the whole n-gram: a phrase must be a traversable
    adjacency, otherwise inserting it as a unit produces text the chat never
    said.
    """
    return all(_is_wordlike(token) for token in ngram) and any(
        _is_content_token(token) for token in ngram
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
    # Position-major so a long message hitting the cap still yields a mix of
    # sizes (size-major would exhaust the cap on bigrams alone).
    for i in range(len(tokens) - 1):
        for size in (2, 3):
            if i + size > len(tokens):
                continue
            ngram = tuple(tokens[i : i + size])
            if ngram in seen:
                continue
            if not is_content_ngram(ngram):
                continue
            seen.add(ngram)
            result.append(ngram)
            if len(result) >= max_per_message:
                return result
    return result
