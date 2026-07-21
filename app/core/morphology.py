"""Approximate Russian stemming: a fold key for token matching.

Lives in its own dependency-free module because both ``candidate_scorer``
(relevance overlap, PR #66) and ``markov`` (context-affine start selection)
need it, and ``candidate_scorer`` already imports ``markov`` — importing the
stemmer from there would be a cycle.
"""
from __future__ import annotations

from functools import lru_cache

# Russian inflectional endings, longest match wins. Deliberately NOT a full
# Snowball: adjective/noun/verb endings only, one strip per token, and a
# minimum stem length so short words pass through untouched. Occasional
# over/under-stripping is fine — the stem is used purely as a fold key for
# matching, never shown to anyone.
_RU_INFLECTION_ENDINGS: tuple[str, ...] = tuple(
    sorted(
        {
            # adjectives / participles
            "ыми", "ими", "ого", "его", "ому", "ему",
            "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой",
            "ым", "им", "ых", "их", "ую", "юю",
            # nouns
            "иями", "ями", "ами", "иях", "ях", "ах",
            "ием", "ем", "ом", "ев", "ов", "ией", "ей",
            "ам", "ям", "ию", "ью", "ия", "ья",
            "а", "я", "ы", "и", "е", "у", "ю", "о",
            # verbs (light)
            "ться", "ешь", "ете", "ет", "ут", "ют", "ит", "ат", "ят",
            "ила", "ил", "ло", "ли", "л", "ть",
        },
        key=len,
        reverse=True,
    )
)

_MIN_STEM_LEN = 3


# Memoized: the context-affine start selection (markov.weighted_start3_choice)
# re-stems the SAME static learned starts on every generation — profiling
# showed ~11.6M calls collapse to ~6.5k unique tokens (99.9% hit rate), and
# caching cut generation latency 203ms -> 79ms with byte-identical output.
# Pure function of the string, so the cache is instance-independent (no bearing
# on the single-instance assumption). maxsize bounds it to the corpus vocab.
@lru_cache(maxsize=100_000)
def stem_token(token: str) -> str:
    """Fold a Russian inflected form to an approximate stem.

    "гнойному"/"гнойный" -> "гнойн", "пидору"/"пидора" -> "пидор",
    "славу" -> "слав". Tokens shorter than the minimum stem plus an ending
    (and non-Cyrillic tokens) come back unchanged.
    """
    for ending in _RU_INFLECTION_ENDINGS:
        if token.endswith(ending) and len(token) - len(ending) >= _MIN_STEM_LEN:
            return token[: -len(ending)]
    return token
