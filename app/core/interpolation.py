"""Soft interpolation of the step distribution with the order-2 projection.

Markov 2.0R Phase 9 (M2R-900). Replaces the hard backoff "order-3 found, do not
look at order-2" with

    P(w | w1,w2,w3) = (1-beta) * P3(w | w1,w2,w3) + beta * P2(w | w2,w3)

on every step where the order-3 state exists.

Why this is a different kind of knob from the temporal blend it is modelled on:
the blend REWEIGHTS the candidates a state already offers, and on this corpus
97.9% of order-3 states offer exactly one (94.6% of visits) — there is nothing
to reweight, which is precisely why Phase 2 and Phase 3 measured inert. This
merge ADDS candidates the state never had, so it is the first mechanism in the
project whose target metric has a way to move.

Two structural rules carried from the spec:

* the merge runs on NORMALIZED distributions, never on raw counts — layer
  volumes are incomparable, and summing counts would give the bigger layer the
  louder voice regardless of what it thinks;
* ``beta = 0`` returns None before reading or computing anything, which is what
  makes the disabled path byte-identical rather than merely close.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.temporal import TransitionRow


@dataclass(frozen=True, slots=True)
class InterpolatedPool:
    """Merged pool of one step plus its normalized distribution.

    ``rows`` and ``weights`` are aligned and ordered by token — the same order
    the sampler already requires of transition rows, so the merge does not
    introduce a second, conflicting order on the hot path.

    ``added`` and ``displacement`` are the intent-versus-effect pair: a
    configured beta proves intent, these two prove the merge did something.
    """

    rows: list[TransitionRow]
    weights: list[float]
    added: int
    displacement: float


def _normalized(rows: Sequence[TransitionRow], base: Sequence[float] | None) -> list[float]:
    """Distribution over ``rows``: the blend's weights when it ran, else counts.

    Composing this way keeps the two mechanisms from fighting over one point:
    the temporal blend answers "how much does each candidate of a layer weigh",
    this module answers "how much does each layer weigh". ``base`` is already a
    distribution when the blend produced it, so it is passed through untouched.
    """
    if base is not None:
        return list(base)
    counts = [float(max(row[1], 1)) for row in rows]
    total = math.fsum(counts)
    if total <= 0.0:
        uniform = 1.0 / len(rows)
        return [uniform] * len(rows)
    return [count / total for count in counts]


@dataclass(frozen=True, slots=True)
class OrderInterpolation:
    """Interpolation configuration for one generation.

    The default instance is the neutral one: ``beta = 0`` takes the early return
    in :meth:`merge`, so a generation that never builds a tuned instance samples
    exactly as it did before this module existed.
    """

    beta: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.beta > 0.0

    def merge(
        self,
        pool3: Sequence[TransitionRow],
        pool2: Sequence[TransitionRow],
        *,
        base3: Sequence[float] | None = None,
        base2: Sequence[float] | None = None,
    ) -> InterpolatedPool | None:
        """Merged distribution for one step, or None when nothing to merge.

        None is the contract that keeps the caller on its existing path with raw
        counts — returned when the feature is off, when either side is empty, or
        when the projection adds nothing and would only re-express the same
        distribution in different units. In all three cases the step samples
        exactly as it would without interpolation, which is what "degenerates to
        pure P3 rather than failing" means in practice.
        """
        if not self.enabled or not pool3 or not pool2:
            return None

        p3 = _normalized(pool3, base3)
        p2 = _normalized(pool2, base2)
        by_token3 = {row[0]: index for index, row in enumerate(pool3)}
        if all(row[0] in by_token3 for row in pool2) and len(pool2) == len(pool3):
            # The projection offers the same tokens as the state: merging would
            # shift mass between known candidates, not add any. That is the
            # reweighting Phase 2 and 3 already proved inert, and doing it here
            # would also swap raw counts for probabilities under the sampler for
            # no gain. Left to the existing path.
            return None

        beta = min(1.0, max(0.0, self.beta))
        merged: dict[str, float] = {}
        rows_by_token: dict[str, TransitionRow] = {}
        for row, share in zip(pool3, p3, strict=True):
            merged[row[0]] = (1.0 - beta) * share
            rows_by_token[row[0]] = row
        added = 0
        for row, share in zip(pool2, p2, strict=True):
            token = row[0]
            if token not in rows_by_token:
                # Keep the order-2 row only for tokens the state never had: for
                # shared tokens the order-3 row stays, so the merged pool carries
                # the state's own rows wherever it has an opinion.
                rows_by_token[token] = row
                added += 1
            merged[token] = merged.get(token, 0.0) + beta * share

        tokens = sorted(merged)
        rows = [rows_by_token[token] for token in tokens]
        weights = [merged[token] for token in tokens]

        # Total variation against pure P3 over the union: the merge can move mass
        # without moving the winner, and that move has to be visible as a number.
        pure3 = {row[0]: share for row, share in zip(pool3, p3, strict=True)}
        displacement = 0.5 * math.fsum(
            abs(merged[token] - pure3.get(token, 0.0)) for token in tokens
        )
        return InterpolatedPool(
            rows=rows, weights=weights, added=added, displacement=displacement
        )
