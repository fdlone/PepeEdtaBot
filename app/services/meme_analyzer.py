"""The daily meme analysis pass (M2R-300, TZ §10.1).

Reads the chat's bigram aggregates, scores them by association rather than
frequency, and rewrites the collocation registry. Runs from the existing daily
maintenance rather than from a script somebody has to remember — a job nobody
schedules is a feature that quietly does nothing.

Cost is a stated budget, not an afterthought: measured on the prod copy the pass
takes ~41 ms with the support threshold applied in SQL (88 ms without it),
because 88% of the chat's bigrams occur exactly once and can never be memes.
The pass runs under the process-wide DB lock, so that number is what it blocks
message handling for, and it is recorded per pass so growth shows up as a
number before it shows up as a stall.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.core.collocations import score_pair
from app.core.temporal import decay_multiplier
from app.repositories.collocations_repo import CollocationsRepo


@dataclass(frozen=True, slots=True)
class MemeSettings:
    """Process-level settings for the daily pass.

    Separate from the per-chat runtime knobs on purpose: this is a background
    job, not a per-reply decision, and giving it its own small object keeps the
    learn path from having to carry a RuntimeState it does not otherwise need.
    Defaults mirror the registry.
    """

    min_joint_count: int = 3
    min_support: float = 10.0
    recency_days: float = 30.0
    max_entries: int = 100


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """What one pass did — the telemetry the spec asks to be observable."""

    scored_pairs: int
    stored_pairs: int
    duration_ms: float


async def analyze_chat_memes(
    repo: CollocationsRepo,
    chat_id: int,
    *,
    now: int,
    min_joint_count: int,
    min_support: float,
    recency_days: float,
    max_entries: int,
) -> AnalysisResult:
    """Score the chat's pairs and rewrite its registry.

    ``now`` is passed in rather than read here for the same reason the temporal
    layer takes it as an argument: a pass whose result depends on the wall clock
    cannot be tested or replayed.
    """
    started = time.perf_counter()
    pairs = await repo.read_pair_counts(chat_id, min_joint_count=min_joint_count)
    if not pairs:
        return AnalysisResult(0, 0, (time.perf_counter() - started) * 1000)

    left_totals, right_totals, total = await repo.read_token_marginals(chat_id)
    scored: list[tuple[str, str, int, float, float]] = []
    for left, right, joint, last_seen in pairs:
        # Recency from the pair's own last_seen — the temporal record Phase 3
        # added. A pair the chat has stopped saying decays out of the ranking on
        # its own instead of waiting to be retired by hand. Rows that predate
        # the temporal migration carry NULL and are treated as fully recent:
        # inventing an age for them would be inventing history.
        recent_share = (
            1.0
            if last_seen is None
            else decay_multiplier(now - last_seen, recency_days)
        )
        association = score_pair(
            joint=joint,
            left=left_totals.get(left, 0),
            right=right_totals.get(right, 0),
            total=total,
            min_support=min_support,
            recent_share=recent_share,
        )
        if association.meme_score <= 0.0:
            continue
        scored.append(
            (left, right, joint, association.npmi, association.meme_score)
        )

    scored.sort(key=lambda item: -item[4])
    capped = scored[:max_entries]
    await repo.replace_scored(chat_id, capped)
    return AnalysisResult(
        scored_pairs=len(pairs),
        stored_pairs=len(capped),
        duration_ms=(time.perf_counter() - started) * 1000,
    )
