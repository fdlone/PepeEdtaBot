"""Shadow order-4 selector (Markov 2.0R Phase 1, M2R-020).

Measures — without an order-4 index and without touching generation — how
often a variable-order 4→3→2 selector (TZ §5) *would have* chosen order 4.
The support estimate comes from the retained-message window
(``LearningService.get_order4_shadow_index``): a conservative lower bound of
full-history support, since the window is a subset of everything the chain
ever learned. Telemetry labels the estimator accordingly; the numbers feed the
pre-registered Phase 7 gate (``eval_thresholds.yaml``).

The analysis is post-hoc over an emitted reply's token sequence: consecutive
walk states are exactly the reply's consecutive tokens, so no walk-loop
instrumentation (and no RNG interference) is needed. Replies containing jumps
or splices are skipped by the caller — their token adjacency crosses splice
boundaries and would produce junk 4-grams.

The selector constants mirror TZ §5 defaults; they become knobs only in
Phase 7, if the gate opens.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.core.markov import pool_diagnostics

SHADOW_ORDER4_MIN_COUNT = 3  # TZ §5 MARKOV_ORDER4_MIN_COUNT default
SHADOW_CONFIDENCE_THRESHOLD = 0.35  # TZ §5 MARKOV_CONFIDENCE_THRESHOLD default
ESTIMATOR_LABEL = "window"

ShadowIndex = Mapping[tuple[str, str, str, str], Mapping[str, int]]


def shadow_order4_stats(
    tokens: list[str], index: ShadowIndex
) -> tuple[int, int]:
    """``(eligible_steps, would_select_order4)`` over one reply's tokens.

    A step is eligible when 4 tokens of history precede it. Order 4 would be
    selected when the window-estimated continuation pool of the 4-token state
    has support >= ``SHADOW_ORDER4_MIN_COUNT`` and confidence >=
    ``SHADOW_CONFIDENCE_THRESHOLD``. Casefolded on both sides — robust to the
    ``normalize_lower`` profile.
    """
    folded = [token.casefold() for token in tokens]
    eligible = 0
    selected = 0
    for position in range(4, len(folded)):
        state = (
            folded[position - 4],
            folded[position - 3],
            folded[position - 2],
            folded[position - 1],
        )
        eligible += 1
        continuations = index.get(state)
        if not continuations:
            continue
        counts = list(continuations.values())
        if sum(counts) < SHADOW_ORDER4_MIN_COUNT:
            continue
        _entropy, _normalized, _branching, confidence = pool_diagnostics(counts)
        if confidence >= SHADOW_CONFIDENCE_THRESHOLD:
            selected += 1
    return eligible, selected
