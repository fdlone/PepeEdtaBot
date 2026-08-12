"""Seed-token scoring for statistical lexical anchoring (M2R-410, TZ §9.4).

A seed is the anchor a reply is grown around. Picking it by inverse document
frequency alone is a trap: a junk unique token (``foobar123``) has maximal IDF
and no support, and it would win every time. This module is the arithmetic that
avoids that — and nothing else: no SQL, no clock, no RNG.

    seed_score = normalized_idf × support_factor × branching_quality

- **normalized_idf** — how distinctive the token is, on the message's own
  scale, so a rare word in a rare message does not automatically outrank a
  distinctive word in an ordinary one;
- **support_factor** — how well the token is actually used in the model
  (reused from ``collocations``): rare-but-real is good, almost-never-seen is
  bad;
- **branching_quality** — a *band*, not a threshold: a token that barely
  branches would stall generation, one that branches enormously is an anchor
  about nothing, and the middle is what we want.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import NamedTuple

from app.core.collocations import support_factor
from app.core.lexicon import STOPWORDS


def idf(df: int, n_docs: int) -> float:
    """Inverse document frequency, ``log(n_docs / (1 + df))``.

    Returns 0 when there is no corpus yet (``n_docs <= 0``) — a fresh chat has
    no informative-ness to measure, and the caller treats a zero-scored token
    as no anchor (the transparent fallback of §9.4), never a crash. df is the
    incremental aggregate of M2R-400; it accumulates on live prod and is never
    recomputed.
    """
    if n_docs <= 0:
        return 0.0
    return math.log(n_docs / (1 + max(df, 0)))


def branching_quality(
    branch: float,
    *,
    minimum: float,
    ideal: float,
    maximum: float,
) -> float:
    """Trapezoidal band over a token's branching (TZ §9.4).

    ``0`` below ``minimum`` (unusable — generation would immediately stall),
    ``1`` across ``[minimum, ideal]``, then a linear decay to ``0`` by
    ``maximum`` (an anchor that continues into anything says nothing). The
    bounds are knobs, calibrated by eval. Degenerate/misordered bounds collapse
    to the safe answer rather than raising: ``minimum >= maximum`` → 0 outside
    the point, and ``ideal`` is clamped into ``[minimum, maximum]``.
    """
    if branch < minimum or branch <= 0:
        return 0.0
    ideal = min(max(ideal, minimum), maximum)
    if branch <= ideal:
        return 1.0
    if branch >= maximum:
        return 0.0
    # Linear decay from 1 at ``ideal`` to 0 at ``maximum``.
    return (maximum - branch) / (maximum - ideal)


class SeedScore(NamedTuple):
    """One token's seed score plus the factors that produced it."""

    token: str
    normalized_idf: float
    support: float
    branching: float
    score: float


def is_scorable(token: str, *, min_token_len: int) -> bool:
    """A token eligible to be scored as a seed at all (TZ §9.4).

    Stopwords and short tokens are excluded before scoring, not penalized
    after: they are never anchors, and letting them into the ranking only
    dilutes it. Casefolded stopword check — the model may be case-preserving.
    """
    return len(token) >= min_token_len and token.casefold() not in STOPWORDS


def score_seeds(
    tokens: Iterable[str],
    *,
    df_of: dict[str, int],
    support_of: dict[str, int],
    forward_branch_of: dict[str, int],
    reverse_branch_of: dict[str, int],
    n_docs: int,
    min_support: float,
    branch_min: float,
    branch_ideal: float,
    branch_max: float,
    min_token_len: int,
) -> list[SeedScore]:
    """Score a message's tokens as seed candidates, best first (TZ §9.4).

    ``*_of`` are lookups the caller has already gathered (df, total model
    count, forward and reverse branching per token) — this module does no I/O.
    Normalization is by the message's own maximum IDF, so the score is
    comparable across messages; a message whose tokens are all unscorable, or a
    corpus with ``n_docs == 0``, yields an empty list (the transparent
    fallback, not an error). Branching quality is applied to the *smaller* of
    the two directions: a seed unusable in either direction is unusable.
    """
    eligible = [
        token
        for token in dict.fromkeys(tokens)  # de-dupe, keep first-seen order
        if is_scorable(token, min_token_len=min_token_len)
    ]
    if not eligible or n_docs <= 0:
        return []

    idf_of = {token: idf(df_of.get(token, 0), n_docs) for token in eligible}
    max_idf = max(idf_of.values())
    if max_idf <= 0.0:
        return []

    scored: list[SeedScore] = []
    for token in eligible:
        normalized_idf = idf_of[token] / max_idf
        support = support_factor(support_of.get(token, 0), min_support)
        branch = min(
            forward_branch_of.get(token, 0),
            reverse_branch_of.get(token, 0),
        )
        quality = branching_quality(
            branch, minimum=branch_min, ideal=branch_ideal, maximum=branch_max
        )
        score = normalized_idf * support * quality
        scored.append(
            SeedScore(token, normalized_idf, support, quality, score)
        )
    scored.sort(key=lambda item: -item.score)
    return scored
