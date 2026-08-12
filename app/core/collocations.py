"""Association measures for chat memes (M2R-300, TZ §10.1).

Frequency cannot tell a meme from a filler. Measured on the prod copy, ranking
bigrams by raw association strength puts one-character tokens at the top, while
the pairs a human would call memes sit at normalized PMI 0.91–0.94 with lift in
the hundreds. This module is the arithmetic that separates the two, and nothing
else: no SQL, no clock, no RNG.

Three measures, each answering a different question:

- **normalized PMI** — how much more often do these two occur together than
  chance predicts, on a scale that does not reward rarity without bound;
- **lift** — the same ratio unnormalized, kept because it is the number a human
  reading the report recognizes;
- **LLR** (Dunning) — how much evidence there is for the association at all.
  Deliberately NOT the ranking key: LLR grows with frequency, so it re-creates
  the frequency ranking this phase exists to replace. It stays as a support
  measure and a tie-breaker.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

# A pair below this many joint occurrences carries no evidence of anything. The
# registry's own knob may raise it; it may not go below 2, because a pair seen
# once is not a pair that recurs. Measured on the prod copy: 88% of bigrams
# occur exactly once, so this bound is also what keeps the daily pass cheap.
MIN_SUPPORTABLE_JOINT_COUNT = 2


class Association(NamedTuple):
    """One pair's association measures plus the composite ranking score."""

    npmi: float
    lift: float
    llr: float
    meme_score: float


def normalized_pmi(joint: float, left: float, right: float, total: float) -> float:
    """PMI scaled to [-1, 1] (Bouma). 1 = the two never occur apart.

    Normalizing matters here: raw PMI is unbounded and maximal for pairs seen
    once, which is precisely the noise the support threshold exists to exclude.
    """
    if joint <= 0 or left <= 0 or right <= 0 or total <= 0:
        return 0.0
    p_joint = joint / total
    p_left = left / total
    p_right = right / total
    if p_joint >= 1.0:
        return 1.0
    return math.log2(p_joint / (p_left * p_right)) / -math.log2(p_joint)


def lift(joint: float, left: float, right: float, total: float) -> float:
    """How many times more likely than chance. 1.0 means independent."""
    if joint <= 0 or left <= 0 or right <= 0 or total <= 0:
        return 0.0
    return (joint * total) / (left * right)


def log_likelihood_ratio(joint: float, left: float, right: float, total: float) -> float:
    """Dunning's LLR over the pair's 2x2 contingency table."""
    k11 = joint
    k12 = max(left - joint, 0.0)
    k21 = max(right - joint, 0.0)
    k22 = max(total - k11 - k12 - k21, 0.0)

    def entropy(*cells: float) -> float:
        cell_total = sum(cells)
        if cell_total <= 0:
            return 0.0
        return sum(c * math.log(c / cell_total) for c in cells if c > 0)

    return 2.0 * (
        entropy(k11, k12, k21, k22)
        - entropy(k11 + k12, k21 + k22)
        - entropy(k11 + k21, k12 + k22)
    )


def support_factor(joint: float, min_support: float) -> float:
    """Ramp from 0 to 1 as joint count reaches the support target.

    A pair just over the hard threshold should not rank beside one seen fifty
    times purely because both are "above threshold". Saturating rather than
    growing without bound keeps this a qualifier, not a second frequency term —
    which would undo the point of ranking by association.
    """
    if min_support <= 0:
        return 1.0
    return min(1.0, max(0.0, joint / min_support))


def recency_factor(recent_share: float) -> float:
    """Weight by how much of the pair's life is inside the recent window.

    ``recent_share`` in [0, 1]: 1.0 means every occurrence is recent. A meme the
    chat has stopped saying should fall out of the list on its own rather than
    waiting to be retired by hand.
    """
    return min(1.0, max(0.0, recent_share))


class CollocationEffect(NamedTuple):
    """What the active collocations did to one candidate (M2R-320).

    ``withheld`` counts breaks that were NOT penalized because the chain never
    offered the collocation's right token there. It is reported separately on
    purpose: it is the evidence that the guard earns its place, and if it turns
    out to be always zero the guard is dead code rather than a safeguard.
    """

    bonus_hits: int
    penalty_hits: int
    withheld: int

    def delta(self, bonus: float, penalty: float) -> float:
        """Score adjustment for the configured weights.

        Both weights default to 0, so this is 0 until the gate says otherwise.
        """
        return self.bonus_hits * bonus - self.penalty_hits * penalty


def collocation_effect(
    tokens: list[str],
    active: frozenset[tuple[str, str]],
    was_available: Callable[[tuple[str, ...], str], bool],
) -> CollocationEffect:
    """Count intact reproductions and genuine breaks in one candidate.

    ``was_available(state, token)`` answers "did the chain hold a transition
    from this state to this token". A break is only a break if the answer is
    yes: otherwise the candidate is being punished for a continuation the corpus
    never contained, which measures the corpus rather than the candidate. That
    is the same mistake Phase 2's M2R-110 made in a different costume.

    The callable is expected to answer from state the caller already has —
    scoring runs per candidate and must not issue a query per pair.
    """
    if not active or len(tokens) < 2:
        return CollocationEffect(0, 0, 0)

    lefts = {left for left, _ in active}
    bonus = penalty = withheld = 0
    for index in range(len(tokens) - 1):
        left, following = tokens[index], tokens[index + 1]
        if left not in lefts:
            continue
        if (left, following) in active:
            bonus += 1
            continue
        state = (tokens[index - 1], left) if index else (left,)
        for pair_left, pair_right in active:
            if pair_left != left or pair_right == following:
                continue
            if was_available(state, pair_right):
                penalty += 1
            else:
                withheld += 1
    return CollocationEffect(bonus, penalty, withheld)


def score_pair(
    *,
    joint: float,
    left: float,
    right: float,
    total: float,
    min_support: float,
    recent_share: float,
) -> Association:
    """All three measures plus ``meme_score`` (TZ §10.1).

    ``meme_score = normalized_pmi × support_factor × recency_factor``. A
    negative association scores zero rather than negative: the registry ranks
    memes, and "these two avoid each other" is a different finding with no place
    in this list.
    """
    npmi = normalized_pmi(joint, left, right, total)
    composite = (
        max(0.0, npmi)
        * support_factor(joint, min_support)
        * recency_factor(recent_share)
    )
    return Association(
        npmi=npmi,
        lift=lift(joint, left, right, total),
        llr=log_likelihood_ratio(joint, left, right, total),
        meme_score=composite,
    )
