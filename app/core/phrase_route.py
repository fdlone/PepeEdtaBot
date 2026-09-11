"""Phrase selection for the phrase route (M3R-210, change phrase-route).

Pure: takes the index rows already read for the message's anchors and picks
which phrases the route assembles around. No RNG, no I/O — the ranking is
deterministic by construction, so the route's only draws are the assembler's.
"""

from __future__ import annotations

from collections.abc import Iterable

# How many anchors of the message are looked at — the same cap the
# associative route uses (ASSOC_ANCHORS_MAX): a message is about a few things.
PHRASE_ANCHORS_MAX = 3


def rank_phrases(
    rows: Iterable[tuple[tuple[str, ...], int]],
    *,
    anchors: list[str],
    message_tokens: Iterable[str],
    slots: int,
) -> list[tuple[str, ...]]:
    """Phrases to assemble around, best-supported first, round-robin by anchor.

    ``rows`` come ordered by support descending then lexically (the repository
    contract), so per anchor the first row is the best and ties are stable.
    Rules (design D3): a phrase made only of the message's own tokens is a
    copy of the input, not a route; one phrase per anchor per round so two
    slots do not spend themselves on one anchor; a phrase whose tokens all sit
    inside an already chosen one is the same trajectory again (the
    self-standing rule of ``phrase_census``), not a second candidate.
    """
    if slots <= 0 or not anchors:
        return []
    message = {token.casefold() for token in message_tokens}
    per_anchor: dict[str, list[tuple[str, ...]]] = {anchor: [] for anchor in anchors}
    for phrase, _count in rows:
        if all(token.casefold() in message for token in phrase):
            continue
        for anchor in anchors:
            if anchor in phrase:
                per_anchor[anchor].append(phrase)
                break
    picked: list[tuple[str, ...]] = []
    queues = [list(phrases) for phrases in per_anchor.values()]
    while len(picked) < slots and any(queues):
        for queue in queues:
            while queue and len(picked) < slots:
                phrase = queue.pop(0)
                if any(set(phrase) <= set(chosen) for chosen in picked):
                    continue
                picked.append(phrase)
                break
    return picked
