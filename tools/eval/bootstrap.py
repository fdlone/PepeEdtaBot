"""Bootstrap confidence intervals (doc 05 §4), pure stdlib.

Percentile bootstrap over per-generation sample arrays, >=1000 resamples,
pooled over the protocol seeds. The resampling RNG is itself seeded so CIs are
reproducible run-to-run (spec: bit-for-bit reproducibility covers the report).
"""

from __future__ import annotations

import math
import random
from statistics import mean

from .metrics import distinct_n

RESAMPLES = 1000
_CI_SEED = 5_0913  # fixed: CIs must not change between identical runs


def bootstrap_ci(
    samples: list[float], *, resamples: int = RESAMPLES
) -> tuple[float, float, float]:
    """Point estimate (mean) and 95% percentile-bootstrap CI for one metric.

    ``nan`` — метка «этой записи в метрике нет»: метрики с фильтром выровнены
    по списку записей, чтобы дельта могла строить пары (см. ``complete_pairs``).
    Одиночному интервалу выравнивание не нужно, поэтому метки отбрасываются.
    """
    samples = [value for value in samples if not math.isnan(value)]
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


def complete_pairs(
    samples_a: list[float], samples_b: list[float]
) -> tuple[list[float], list[float]]:
    """Пары наблюдений: полупара — не наблюдение разности.

    Армы идут по одному и тому же набору промптов в одном и том же порядке
    (``run_matrix``), поэтому позиция и есть идентификатор наблюдения. Пара, в
    которой хотя бы одна половина помечена ``nan`` («записи в метрике нет»),
    выбывает целиком: у разности нет половины.

    Метки нужны именно ради этого. Первая редакция правки просто обрезала
    списки по общей длине — и это было бы верно, только если бы армы теряли
    одни и те же записи. Они теряют **разные**: метрики с фильтром по
    ``record.success`` отбрасывают те промпты, на которых не ответил именно
    этот арм, и после обрезки позиция переставала указывать на тот же промпт в
    обоих списках. Поэтому такие метрики выровнены по исходному списку записей
    (``metrics._aligned``), а не отфильтрованы.
    """
    size = min(len(samples_a), len(samples_b))
    paired_a: list[float] = []
    paired_b: list[float] = []
    for value_a, value_b in zip(samples_a[:size], samples_b[:size], strict=True):
        if math.isnan(value_a) or math.isnan(value_b):
            continue
        paired_a.append(value_a)
        paired_b.append(value_b)
    return paired_a, paired_b


def _paired_delta(
    samples_a: list[float], samples_b: list[float], picks: list[int]
) -> float:
    """Средняя разность по одной бутстрап-выборке индексов."""
    total = 0.0
    for index in picks:
        total += samples_b[index] - samples_a[index]
    return total / len(picks)


def delta_ci(
    samples_a: list[float], samples_b: list[float], *, resamples: int = RESAMPLES
) -> tuple[float, float, float, bool]:
    """Delta (b - a) with a 95% bootstrap CI and its significance verdict.

    Doc 05 §4: a delta is significant only when its interval excludes zero;
    the report must always print the interval, not only the point.
    """
    if not samples_a or not samples_b:
        raise ValueError("delta_ci needs samples on both sides")
    # Парный ресэмплинг: индексы тянутся один раз, обе половины пары берутся
    # по одному и тому же индексу. Матрица конфигураций — парный дизайн по
    # построению (те же промпты, те же сиды, различие в одной ручке), и
    # независимый ресэмплинг оценивал бы Var_A + Var_B вместо дисперсии парной
    # разности Var_A + Var_B − 2ρ·σ_A·σ_B. При положительной ρ — а она
    # положительна здесь по построению — интервал выходил шире истинного в
    # √(1/(1−ρ)) раз: 1.41 при ρ=0.5, 2.24 при ρ=0.8.
    #
    # Ошибка была направленной и потому дорогой: она порождала ложные вердикты
    # «эффекта нет», а конституция объявляет «закрыто с цифрами» полноценным
    # результатом и закрывает направление окончательно. Часть прежних
    # вердиктов о мёртвых ручках могла держаться на ширине интервала.
    #
    # Разрыв сопоставимости с прошлыми отчётами осознанный (решение владельца
    # 2026-08-26): интервалы стали уже, пересчёту прошлые отчёты не подлежат.
    paired_a, paired_b = complete_pairs(samples_a, samples_b)
    if not paired_a:
        raise ValueError("delta_ci needs at least one complete pair")
    indices = range(len(paired_a))
    rng = random.Random(_CI_SEED + 1)
    deltas = sorted(
        _paired_delta(paired_a, paired_b, rng.choices(indices, k=len(paired_a)))
        for _ in range(resamples)
    )
    point = mean(paired_b) - mean(paired_a)
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
