"""Slot mutations: one content word of a candidate swapped for a chat word.

Pure functions only (no I/O). A mutated copy of a generated candidate gets
exactly one content word replaced with a frequent word of the same chat, and
then competes in candidate scoring against its original — bad mutations lose
to the existing quality signals (verbatim penalty conversely rewards the
introduced novelty), so no separate acceptance gate is needed here.

The morphology guard is two-layered. The cheap ending/length heuristic
prefilters candidates; ``pymorphy3`` then requires real agreement — same part
of speech and matching case/number/gender/tense/person where the original has
them. The heuristic alone was measured (2026-07-17, prod copy) to let ~40% of
winning mutations be grammatically broken: matching endings routinely cross
part-of-speech lines in Russian («тянут»/«минут», «ладно»/«окно»).
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

import pymorphy3

from app.core.lexicon import STOPWORDS
from app.core.markov import JUMP_CONNECTIVE_TOKENS

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

# Discourse words of the splice scaffolding (M4 jumps, verbatim extensions)
# are never mutated: swapping the connective breaks the seam the reply was
# built around («, турбаче тогда где»). The short conjunctions of those
# phrases are already excluded by the stopword/length guards.
PROTECTED_DISCOURSE_WORDS: frozenset[str] = frozenset(
    token
    for phrase in JUMP_CONNECTIVE_TOKENS
    for token in phrase
    if token.isalpha()
)

# Grammemes that must carry over from the original to the replacement when
# the original's parse has them.
_AGREEMENT_GRAMMEMES = ("case", "number", "gender", "tense", "person", "mood")

_MORPH: pymorphy3.MorphAnalyzer | None = None


def _morph() -> pymorphy3.MorphAnalyzer:
    # Lazy singleton: the analyzer loads its dictionaries on first use (~1s),
    # so processes that never mutate (knob off, tests) pay nothing.
    global _MORPH
    if _MORPH is None:
        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH


@lru_cache(maxsize=16384)
def _tags(word: str) -> tuple[Any, ...]:
    return tuple(parse.tag for parse in _morph().parse(word))


def agrees_morphologically(original: str, replacement: str) -> bool:
    """True when the replacement can stand in the original's grammatical slot.

    The original's most probable parse fixes the slot: part of speech plus
    every agreement grammeme it carries. The replacement passes if any of its
    parses matches that profile exactly.
    """
    original_tags = _tags(original)
    if not original_tags:
        return False
    original_tag = original_tags[0]
    if original_tag.POS is None:
        return False
    for tag in _tags(replacement):
        if tag.POS != original_tag.POS:
            continue
        if all(
            getattr(tag, grammeme) == getattr(original_tag, grammeme)
            for grammeme in _AGREEMENT_GRAMMEMES
            if getattr(original_tag, grammeme) is not None
        ):
            return True
    return False


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
        if folded in PROTECTED_DISCOURSE_WORDS:
            continue
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
        if folded in PROTECTED_DISCOURSE_WORDS:
            continue
        if not agrees_morphologically(original, word):
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
