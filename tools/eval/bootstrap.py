"""Bootstrap confidence intervals (doc 05 §4), pure stdlib.

Percentile bootstrap over per-generation sample arrays, >=1000 resamples,
pooled over the protocol seeds. The resampling RNG is itself seeded so CIs are
reproducible run-to-run (spec: bit-for-bit reproducibility covers the report).
"""

from __future__ import annotations

import random
from statistics import mean

from .metrics import distinct_n

RESAMPLES = 1000
_CI_SEED = 5_0913  # fixed: CIs must not change between identical runs


def bootstrap_ci(
    samples: list[float], *, resamples: int = RESAMPLES
) -> tuple[float, float, float]:
    """Point estimate (mean) and 95% percentile-bootstrap CI for one metric."""
    if not samples:
        raise ValueError("bootstrap_ci needs at least one sample")
    point = mean(samples)
    if len(samples) == 1:
        return point, point, point
    rng = random.Random(_CI_SEED)
    n = len(samples)
    stats = sorted(
        mean(rng.choices(samples, k=n)) for _ in range(resamples)
    )
    lo = stats[max(0, round(0.025 * (resamples - 1)))]
    hi = stats[min(resamples - 1, round(0.975 * (resamples - 1)))]
    return point, lo, hi


def delta_ci(
    samples_a: list[float], samples_b: list[float], *, resamples: int = RESAMPLES
) -> tuple[float, float, float, bool]:
    """Delta (b - a) with a 95% bootstrap CI and its significance verdict.

    Doc 05 §4: a delta is significant only when its interval excludes zero;
    the report must always print the interval, not only the point.
    """
    if not samples_a or not samples_b:
        raise ValueError("delta_ci needs samples on both sides")
    rng = random.Random(_CI_SEED + 1)
    deltas = sorted(
        mean(rng.choices(samples_b, k=len(samples_b)))
        - mean(rng.choices(samples_a, k=len(samples_a)))
        for _ in range(resamples)
    )
    point = mean(samples_b) - mean(samples_a)
    lo = deltas[max(0, round(0.025 * (resamples - 1)))]
    hi = deltas[min(resamples - 1, round(0.975 * (resamples - 1)))]
    significant = lo > 0 or hi < 0
    return point, lo, hi, significant


def distinct_delta_ci(
    replies_a: list[tuple[str, ...]],
    replies_b: list[tuple[str, ...]],
    n: int,
    *,
    resamples: int = RESAMPLES,
) -> tuple[float, float, float, bool] | None:
    """Delta (b - a) of distinct-N with a 95% bootstrap CI and significance.

    distinct-N is a configuration-level ratio (unique n-grams / all n-grams),
    not a per-generation sample, so it is resampled over whole replies: each
    resample draws ``len(replies)`` replies with replacement and recomputes the
    ratio. This is what makes "distinct-2/3 rose" a checkable claim rather than
    a comparison of two point estimates whose noise is unknown.

    Returns ``None`` when either side has no n-grams to count.
    """
    point_a, _ = distinct_n(replies_a, n)
    point_b, _ = distinct_n(replies_b, n)
    if point_a is None or point_b is None:
        return None
    rng = random.Random(_CI_SEED + 2 + n)
    deltas: list[float] = []
    for _ in range(resamples):
        sample_a, _ = distinct_n(rng.choices(replies_a, k=len(replies_a)), n)
        sample_b, _ = distinct_n(rng.choices(replies_b, k=len(replies_b)), n)
        if sample_a is None or sample_b is None:
            continue
        deltas.append(sample_b - sample_a)
    if not deltas:
        return None
    deltas.sort()
    lo = deltas[max(0, round(0.025 * (len(deltas) - 1)))]
    hi = deltas[min(len(deltas) - 1, round(0.975 * (len(deltas) - 1)))]
    return point_b - point_a, lo, hi, (lo > 0 or hi < 0)
