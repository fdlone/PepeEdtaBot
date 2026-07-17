"""Slot mutations: one content word of a candidate swapped for a chat word.

Pure functions only (no I/O). A mutated copy of a generated candidate gets
exactly one content word replaced with a frequent word of the same chat, and
then competes in candidate scoring against its original — bad mutations lose
to the existing quality signals (verbatim penalty conversely rewards the
introduced novelty), so no separate acceptance gate is needed here.

Morphology guard is heuristic and dependency-free: the replacement must share
the wordform ending with the original (for Russian a matching ending is a
cheap proxy for matching case/number/gender) and be of comparable length.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

from app.core.lexicon import STOPWORDS

# A word shorter than this is never mutated nor used as a replacement: short
# Russian words are mostly function words and pronouns the stopword list
# cannot enumerate, and their endings carry no agreement signal.
MIN_MUTABLE_WORD_LEN = 4
# Wordform ending length that must match between original and replacement.
ENDING_MATCH_LEN = 2
# Replacement length may deviate from the original by at most this share.
MAX_LENGTH_DELTA_SHARE = 0.4
# Words seen fewer times than this in the chat model are not offered as
# replacements: one-off typos and foreign fragments are not "chat style".
MIN_REPLACEMENT_FREQUENCY = 3


def is_mutable_word(token: str) -> bool:
    """True for a content word long enough to carry agreement in its ending."""
    return (
        len(token) >= MIN_MUTABLE_WORD_LEN
        and token.isalpha()
        and token.casefold() not in STOPWORDS
    )


def eligible_slot_indexes(
    tokens: list[str],
    *,
    protected_tokens: frozenset[str],
    context_tokens: frozenset[str],
) -> list[int]:
    """Positions in ``tokens`` where a mutation may happen.

    The first and last tokens are excluded (the opening token carries the
    context anchor, the closing one the ending quality), as are words that
    overlap the context (echoing the context is what anchoring scores reward —
    mutating it away would fight the scorer) and words of hot n-grams (local
    memes must stay verbatim to be recognized).
    """
    eligible: list[int] = []
    for index in range(1, len(tokens) - 1):
        token = tokens[index]
        if not is_mutable_word(token):
            continue
        folded = token.casefold()
        if folded in context_tokens or folded in protected_tokens:
            continue
        eligible.append(index)
    return eligible


def pick_replacement(
    original: str,
    frequencies: Mapping[str, int],
    *,
    excluded_tokens: frozenset[str],
    rng: random.Random,
) -> str | None:
    """A chat word agreeing with ``original`` by ending, or None.

    Candidates are drawn from the chat word-frequency dictionary, filtered by
    the morphology guard, and sampled proportionally to frequency so the
    replacement sounds like the chat rather than like its long tail.
    """
    original_folded = original.casefold()
    ending = original_folded[-ENDING_MATCH_LEN:]
    max_delta = int(len(original) * MAX_LENGTH_DELTA_SHARE)
    pool: list[str] = []
    weights: list[int] = []
    for word, count in frequencies.items():
        if count < MIN_REPLACEMENT_FREQUENCY:
            continue
        folded = word.casefold()
        if folded == original_folded or folded in excluded_tokens:
            continue
        if not is_mutable_word(word):
            continue
        if folded[-ENDING_MATCH_LEN:] != ending:
            continue
        if abs(len(word) - len(original)) > max_delta:
            continue
        pool.append(word)
        weights.append(count)
    if not pool:
        return None
    return rng.choices(pool, weights=weights, k=1)[0]


def mutate_candidate_tokens(
    tokens: list[str],
    *,
    frequencies: Mapping[str, int],
    protected_tokens: frozenset[str],
    context_tokens: Iterable[str],
    rng: random.Random,
) -> list[str] | None:
    """A copy of ``tokens`` with one slot mutated, or None when impossible.

    Eligible slots are tried in random order until one yields a replacement;
    the replacement must not already occur in the candidate or the context
    (that would be an echo, not novelty).
    """
    context_folded = frozenset(token.casefold() for token in context_tokens)
    slots = eligible_slot_indexes(
        tokens,
        protected_tokens=protected_tokens,
        context_tokens=context_folded,
    )
    if not slots:
        return None
    rng.shuffle(slots)
    own_tokens = frozenset(token.casefold() for token in tokens)
    for slot in slots:
        replacement = pick_replacement(
            tokens[slot],
            frequencies,
            excluded_tokens=own_tokens | context_folded,
            rng=rng,
        )
        if replacement is None:
            continue
        mutated = list(tokens)
        mutated[slot] = replacement
        return mutated
    return None
