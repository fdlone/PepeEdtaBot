"""Two-layer transition weights: decaying short-term counter + compressed long
count (Markov 2.0R, M2R-210 — TZ §7-8).

The chain stores one integer per transition, so a phrase said a thousand times
two years ago outweighs what the chat says today. This module is the arithmetic
that fixes that, and nothing else: no SQL, no clock, no RNG. Both the writer
(learning) and the reader (sampling) call the same functions, which is the point
— a second implementation in SQL would be a second thing to keep correct, and
SQLite's math functions are a compile-time option we do not control in every
deployment.

Every function that involves decay takes the moment it is evaluated at as an
argument. Reading the clock in here would make generation non-reproducible and
quietly break the eval protocol's bit-for-bit requirement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

SECONDS_PER_DAY = 86_400.0

# Compression shapes for the long layer (TZ §8.1, MARKOV_LONG_COMPRESSION).
COMPRESSION_LOG = "log"
COMPRESSION_POW = "pow"
COMPRESSION_SHAPES = (COMPRESSION_LOG, COMPRESSION_POW)

# A transition pool row as it travels from SQL through the cache to the sampler:
# ``(token, count, s_value, s_updated_at)``. The last two are the short layer's
# stored pair; ``s_updated_at`` is None for rows that predate the temporal
# record or have never been observed since it began.
TransitionRow = tuple[str, int, float, int | None]


def decay_multiplier(elapsed_seconds: float, half_life_days: float) -> float:
    """``2^(-elapsed/half_life)`` — the factor an existing weight decays by.

    Guards the two degenerate inputs deliberately rather than letting them
    produce a plausible number:

    - a non-positive half-life has no meaning as a decay constant (the knob's
      bounds forbid it; this is the belt to that suspenders) — nothing is
      remembered;
    - a negative elapsed time means the clock went backwards between two
      observations. Extrapolating *growth* from clock skew would let a
      misconfigured host inflate the short layer, so time never runs backwards
      here: the weight is simply not decayed.
    """
    if half_life_days <= 0.0:
        return 0.0
    if elapsed_seconds <= 0.0:
        return 1.0
    return float(2.0 ** (-elapsed_seconds / (half_life_days * SECONDS_PER_DAY)))


def short_effective(
    s_value: float,
    s_updated_at: int | None,
    now: int,
    half_life_days: float,
) -> float:
    """The short layer's weight at ``now`` (TZ §7.1 read form).

    Equals the sum of every past observation decayed by its own age — exactly,
    at O(1) storage. An empty layer (never observed, or reset) reads as 0.
    """
    if s_value <= 0.0 or s_updated_at is None:
        return 0.0
    return s_value * decay_multiplier(now - s_updated_at, half_life_days)


def short_observed(
    s_value: float,
    s_updated_at: int | None,
    now: int,
    half_life_days: float,
) -> float:
    """The stored value after observing the transition at ``now`` (TZ §7.1).

    Decay what was there, then add this observation's full weight of 1. Storing
    the result together with ``now`` as the new ``s_updated_at`` is what keeps
    the identity above true without keeping any per-event rows.
    """
    return short_effective(s_value, s_updated_at, now, half_life_days) + 1.0


def compress_long(count: int, shape: str, beta: float) -> float:
    """Sublinear compression of a historical count (TZ §8.1).

    Without it, a count of 10000 against 20 gives 0.998/0.002 and no sane blend
    weight lets fresh language matter at all — the short layer degenerates into
    a cosmetic correction. Compression keeps the order of preference intact and
    narrows the dynamic range to a working one.
    """
    value = max(count, 0)
    if shape == COMPRESSION_POW:
        return float(float(value) ** beta)
    return float(math.log1p(value))


class BlendedPool(NamedTuple):
    """Blended sampling weights plus how far they moved the distribution.

    ``displacement`` is the total-variation distance between the blended
    distribution and the long layer alone: 0 means the blend changed nothing at
    this step, 1 means it replaced the distribution outright. It exists because
    "the knob is set" and "the knob is doing something" are different claims,
    and Phase 2 was the lesson in not confusing them.
    """

    weights: list[float]
    displacement: float


@dataclass(frozen=True, slots=True)
class TemporalBlend:
    """Blend configuration for one generation (TZ §8.3).

    The default instance is the neutral one: ``alpha = 0`` takes an early return
    before any arithmetic, so a generation that never builds a tuned instance
    samples exactly as Markov 1.x did.
    """

    alpha: float = 0.0
    half_life_days: float = 3.0
    compression: str = COMPRESSION_LOG
    beta: float = 0.6

    @property
    def enabled(self) -> bool:
        return self.alpha > 0.0

    def blend(self, rows: Sequence[TransitionRow], now: int) -> BlendedPool | None:
        """Blended weights for one pool, or ``None`` when the blend is off.

        ``None`` is the contract that makes neutrality structural: the caller
        keeps using the raw counts, so no float arithmetic happens on the
        disabled path and the sampled result is byte-identical to 1.x.

        Weights are normalized over the union of tokens either layer offers, so
        a token only the fresh layer knows stays reachable. If one layer has no
        mass for this state, the blend degenerates to the other rather than
        producing an invalid distribution (TZ §8.3).
        """
        if not self.enabled or not rows:
            return None

        long_weights = [
            compress_long(row[1], self.compression, self.beta) for row in rows
        ]
        short_weights = [
            short_effective(row[2], row[3], now, self.half_life_days) for row in rows
        ]
        long_total = math.fsum(long_weights)
        short_total = math.fsum(short_weights)

        if long_total <= 0.0 and short_total <= 0.0:
            # Nothing to weight by — every candidate is equally (un)supported.
            uniform = 1.0 / len(rows)
            return BlendedPool([uniform] * len(rows), 0.0)

        long_p = (
            [w / long_total for w in long_weights]
            if long_total > 0.0
            else [0.0] * len(rows)
        )
        if short_total <= 0.0:
            # Configured but inert at this step: no fresh observations here.
            return BlendedPool(long_p, 0.0)
        short_p = [w / short_total for w in short_weights]
        if long_total <= 0.0:
            return BlendedPool(short_p, 1.0)

        alpha = min(1.0, self.alpha)
        blended = [
            alpha * short + (1.0 - alpha) * long_
            for short, long_ in zip(short_p, long_p)
        ]
        displacement = 0.5 * math.fsum(
            abs(mixed - long_) for mixed, long_ in zip(blended, long_p)
        )
        return BlendedPool(blended, displacement)
