"""Approximate Russian stemming: a fold key for token matching.

Lives in its own dependency-free module because both ``candidate_scorer``
(relevance overlap, PR #66) and ``markov`` (context-affine start selection)
need it, and ``candidate_scorer`` already imports ``markov`` — importing the
stemmer from there would be a cycle.
"""
from __future__ import annotations

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
