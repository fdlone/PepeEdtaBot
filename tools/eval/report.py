"""Markdown report and gate evaluation (doc 05 §4, §6).

The report is the decision artifact: stored under ``docs/eval_reports/`` and
cited by phase verdicts (including "closed without implementation"). Gate
verdicts are computed strictly from ``eval_thresholds.yaml`` — pass / fail /
insufficient data, with the numbers that produced them.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean
from typing import Any

from app.core.generation_telemetry import CandidateRoute

from .bootstrap import bootstrap_ci, delta_ci, distinct_delta_ci
from .metrics import distinct_n, latency_percentiles, meme_regression, metric_values
from .prompts import PromptSet
from .run import ConfigRun, DfCorpusFacts

# Phase 2 arms (doc 05 §2 + the "one arm per knob" rule): the shipped
# combination C1, each knob alone (C1a/C1b), the flat-temperature control
# (C1flat), and any calibration variant of the grid. Prefix-matched so a
# calibration run gets a per-arm verdict without editing this list.
PHASE2_ARM_PREFIX = "C1"

# Phase 3 arms (temporal blend), same prefix-matching convention: the shipped C2
# plus every calibration variant of the grid.
PHASE3_ARM_PREFIX = "C2"

# Phase 4 arms (PMI memes + collocation scoring), same prefix convention.
PHASE4_ARM_PREFIX = "C3"

# Phase 9 arms (order-3 x order-2 interpolation), same prefix convention. C6 and
# not C5: C5 is taken in matrix.yaml, and since the prefix is what routes an arm
# to its gate, reusing it would compare a Phase 9 arm against Phase 5 thresholds.
PHASE9_ARM_PREFIX = "C6"

# Canonical §3 order for the metrics table.
METRIC_ORDER = (
    "generation_success_rate",
    "candidate_accept_rate",
    "mean_response_length",
    "unique_token_ratio",
    "exact_context_copy_rate",
    "repetition_rate",
    "cycle_detection_rate",
    "cycle_harm_rate",
    "context_affinity",
    "context_affinity_without_copy",
    "seeded_present_rate",
    "seeded_win_rate_given_present",
    "freshness_reflection",
    "historical_meme_rate",
    # M3R-011: printed as a pair and adjacent on purpose — the gap between the
    # pool and the window is the finding, and a reader scanning one row must
    # land on the other.
    "structural_pool_ecb",
    "structural_window_escape",
)
INSUFFICIENT = "insufficient data"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _cell(samples: list[float] | None) -> str:
    if not samples:
        return INSUFFICIENT
    point, lo, hi = bootstrap_ci(samples)
    return f"{_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"


def _delta_cell(base: list[float] | None, other: list[float] | None) -> str:
    result = delta_ci(base, other)
    if result is None:
        return "—"
    point, lo, hi, significant = result
    marker = " *" if significant else ""
    return f"{_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]{marker}"


ROUTE_NOT_ATTEMPTED = "not attempted"


def _route_rows(config_id: str, run: ConfigRun) -> list[str]:
    """Per-route table rows of one configuration (M3R-103, reporting half).

    Two denominators printed separately: pool share (candidates of the route
    among all candidates, per generation) and presence (generations where the
    route placed at least one candidate); the win rate is conditioned on
    presence and never printed alone. Affinity and copy are over the replies
    the route won. Latency is the mean of generations with the route in the
    pool against those without — an upper bound on the route's cost, not a
    measurement of its step (design D3). Rejections come from the generator's
    telemetry, summed by failure class (M3R-021).

    A route with zero attempts and no presence reads ``not attempted``: an
    off route printed as zeros would look like "runs and never wins" (D4).
    """
    with_pool = [record for record in run.records if record.pool_routes]
    rows: list[str] = []
    for route in CandidateRoute:
        name = str(route)
        attempts = sum(
            int(snapshot.get(f"route_{name}_attempts") or 0)
            for snapshot in run.telemetry
        )
        present = [record for record in with_pool if name in record.pool_routes]
        if attempts == 0 and not present:
            rows.append(f"| {config_id} | {name} | 0 | {ROUTE_NOT_ATTEMPTED} |" + " — |" * 7)
            continue
        absent = [record for record in with_pool if name not in record.pool_routes]
        winners = [
            record
            for record in run.records
            if record.success and record.winner_route == name
        ]
        pool_share = _cell(
            [record.pool_routes.count(name) / len(record.pool_routes) for record in with_pool]
        )
        presence = _cell([float(name in record.pool_routes) for record in with_pool])
        win_given_present = _cell(
            [float(record.winner_route == name) for record in present]
        )
        winners_affinity = _cell(
            [
                record.affinity
                for record in winners
                if not record.is_copy and record.affinity is not None
            ]
        )
        winners_copy = _cell([float(record.is_copy) for record in winners])
        latency_present = (
            f"{mean(record.latency_ms for record in present):.1f}"
            if present
            else "—"
        )
        latency_absent = (
            f"{mean(record.latency_ms for record in absent):.1f}" if absent else "—"
        )
        rejected: Counter[str] = Counter()
        prefix = f"route_{name}_rejected_"
        for snapshot in run.telemetry:
            for key, value in snapshot.items():
                if key.startswith(prefix) and value:
                    rejected[key[len(prefix) :]] += int(value)
        rejected_cell = (
            ", ".join(f"{cls} {count}" for cls, count in sorted(rejected.items()))
            if rejected
            else "0"
        )
        rows.append(
            f"| {config_id} | {name} | {attempts} | {pool_share} | {presence} "
            f"| {win_given_present} | {winners_affinity} | {winners_copy} "
            f"| {latency_present} / {latency_absent} | {rejected_cell} |"
        )
    return rows


def _finite(samples: list[float] | None) -> list[float]:
    """Отбросить метки `nan` («записи в метрике нет»).

    Метрики с фильтром выровнены по списку записей, чтобы парная дельта могла
    сшивать наблюдения одного промпта. Всё, что читает такой список как
    выборку, обязано метки убрать.
    """
    if not samples:
        return []
    return [value for value in samples if not math.isnan(value)]


def metrics_summary(
    runs: dict[str, ConfigRun], context_mode: str = "ctx"
) -> dict[str, Any]:
    """Deterministic metric points per configuration, latency excluded —
    the object the bit-for-bit reproducibility check compares.

    ``context_mode`` travels INSIDE each configuration's entry rather than as a
    sibling key: the entries are what gets sliced out, quoted and compared
    between reports, and a mode that lives one level up is a mode that gets
    lost on the way (M3R-140).
    """
    summary: dict[str, Any] = {}
    for config_id, run in sorted(runs.items()):
        values = metric_values(run.records)
        replies = [r.reply_content for r in run.records if r.success]
        d2, basis2 = distinct_n(replies, 2)
        d3, basis3 = distinct_n(replies, 3)
        summary[config_id] = {
            "context_mode": context_mode,
            "shared_with": run.shared_with,
            "n_records": len(run.records),
            "metrics": {
                # `nan` — метка «этой записи в метрике нет» (метрики с
                # фильтром выровнены по списку записей, чтобы дельта строила
                # пары). В сводку она попасть не должна: сводка сравнивается
                # на побитовую воспроизводимость, а `nan != nan` — прогон
                # переставал равняться самому себе.
                name: (round(mean(clean), 9) if (clean := _finite(samples)) else None)
                for name, samples in values.items()
            },
            "distinct_2": None if d2 is None else round(d2, 9),
            "distinct_3": None if d3 is None else round(d3, 9),
            "distinct_basis_tokens": {"2": basis2, "3": basis3},
        }
    return summary


def _delta_part(
    label: str,
    base: list[float] | None,
    arm: list[float] | None,
    *,
    parts: list[str],
    missing: list[str],
) -> tuple[float, bool] | None:
    """Append one metric's delta to ``parts``; return (point, significant).

    ``None`` means the delta was not computable — the caller has already been
    told what is missing and must not treat the absence as a pass.
    """
    # Полных пар может не оказаться и на непустых списках: армы отвечают на
    # разных промптах, и после выравнивания метками пересечение бывает
    # пустым. `delta_ci` возвращает None и на это, и на пустую сторону —
    # отсутствие данных ведёт в «insufficient», а не роняет прогон.
    result = delta_ci(base, arm)
    if result is None:
        # Две разные причины и два разных действия: пустая сторона — чинить
        # прогон, пустое пересечение непустых списков — смотреть выравнивание
        # (армы ответили на непересекающихся промптах). Первая редакция
        # правки схлопнула их в одну строку, и вердикт `insufficient data`
        # перестал говорить, что именно чинить.
        missing.append(
            f"{label} samples" if not base or not arm else f"{label} paired samples"
        )
        return None
    point, lo, hi, significant = result
    parts.append(
        f"{label} Δ {_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"
        f"{' *' if significant else ''}"
    )
    return point, significant


def _delta_ci_low(
    label: str,
    base: list[float] | None,
    arm: list[float] | None,
    *,
    parts: list[str],
    missing: list[str],
) -> float | None:
    """Like ``_delta_part`` but returns the CI LOWER BOUND instead of the point.

    Phase 9 is the first gate that bounds the interval rather than the estimate:
    earlier phases ask "did this drop significantly", which fails only on a
    demonstrated move. Interpolation is expected to cost some topicality by
    construction, so the question changes to "how much drop can we not rule
    out" — and that is a statement about the bound, not about the point.
    """
    # Полных пар может не оказаться и на непустых списках: армы отвечают на
    # разных промптах, и после выравнивания метками пересечение бывает
    # пустым. `delta_ci` возвращает None и на это, и на пустую сторону —
    # отсутствие данных ведёт в «insufficient», а не роняет прогон.
    result = delta_ci(base, arm)
    if result is None:
        # Две разные причины и два разных действия: пустая сторона — чинить
        # прогон, пустое пересечение непустых списков — смотреть выравнивание
        # (армы ответили на непересекающихся промптах). Первая редакция
        # правки схлопнула их в одну строку, и вердикт `insufficient data`
        # перестал говорить, что именно чинить.
        missing.append(
            f"{label} samples" if not base or not arm else f"{label} paired samples"
        )
        return None
    point, lo, hi, significant = result
    parts.append(
        f"{label} Δ {_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"
        f"{' *' if significant else ''}"
    )
    return lo


def _latency_part(
    arm: ConfigRun,
    budget: float,
    *,
    parts: list[str],
    missing: list[str],
    failures: list[str],
) -> None:
    p95 = latency_percentiles(arm.records)["latency_p95"]
    if p95 is None:
        missing.append("latencies")
        return
    parts.append(f"p95 {p95:.1f} ms (budget {budget:.0f})")
    if p95 > budget:
        failures.append("p95 over budget")


def _verdict(
    parts: list[str], failures: list[str], missing: list[str]
) -> tuple[str, str]:
    """Shared gate resolution: a demonstrated failure outranks a missing part.

    An arm that provably raised copying does not become undecided because some
    other part of the gate lacked data. Only a gate with no failures AND a hole
    is undecided.
    """
    detail = "; ".join(parts) if parts else "nothing computable"
    if missing:
        detail = f"{detail} — missing: {', '.join(missing)}"
    if failures:
        return "fail", f"{detail} — {'; '.join(failures)}"
    if missing:
        return INSUFFICIENT, detail
    return "pass", detail


PHASE5_ARM_PREFIX = "C4"


def _phase5_corpus_precondition(
    config: dict[str, Any],
    df_facts: DfCorpusFacts | None,
    *,
    parts: list[str],
    missing: list[str],
) -> None:
    """Corpus-sufficiency precondition of the Phase 5 gate (both parts).

    Any shortfall lands in ``missing`` (=> ``insufficient data``), never in
    failures: "not enough corpus yet" is not a failed phase. Both measured
    quantities are printed next to the bars either way, so "not yet" is
    distinguishable from "measured and did not hold" without reading the DB.
    """
    floor_raw = config.get("n_docs_min")
    ceiling_raw = config.get("df_singleton_share_max")
    if floor_raw is None or ceiling_raw is None:
        # Design D4: a default here would be a bar that never went through
        # pre-registration — the verdict must be checkable against the file.
        missing.append(
            "corpus precondition not registered in the thresholds file "
            "(n_docs_min / df_singleton_share_max)"
        )
        return
    floor, ceiling = int(floor_raw), float(ceiling_raw)
    if df_facts is None:
        missing.append("prod df corpus facts were not measured for this run")
        return
    if df_facts.error is not None:
        missing.append(f"source df unreadable: {df_facts.error}")
        return
    n_docs = df_facts.n_docs or 0
    share = df_facts.singleton_share
    share_txt = f"{share:.0%}" if share is not None else "df empty"
    parts.append(
        f"prod corpus: n_docs {n_docs} (floor {floor}), "
        f"singleton share {share_txt} (ceiling {ceiling:.0%})"
    )
    if n_docs < floor:
        missing.append(f"prod n_docs {n_docs} below the floor {floor}")
    if share is None:
        missing.append("prod markov_token_df is empty for this chat")
    elif share > ceiling:
        missing.append(
            f"df singleton share {share:.0%} above the ceiling {ceiling:.0%}"
        )


def _mean_telemetry(arm: ConfigRun, key: str) -> float | None:
    """Mean of a per-seed telemetry value across an arm's snapshots.

    ``None`` when no snapshot reported it (e.g. the seeded branch never fired,
    so the win rate has no denominator on any seed)."""
    values = [
        float(v)
        for snap in arm.telemetry
        if (v := snap.get(key)) is not None
    ]
    return sum(values) / len(values) if values else None


def _phase5_arm_verdict(
    baseline: ConfigRun,
    arm: ConfigRun,
    thresholds: dict[str, Any],
    df_facts: DfCorpusFacts | None = None,
) -> tuple[str, str]:
    """Phase 5 promotion gate for one seeded arm (verdict, detail).

    Computes every automatic condition — seeded present rate, seeded win rate
    given present, the affinity-without-copy delta vs the no-seeded baseline,
    p95 — and prints them. The verdict additionally carries the
    corpus-sufficiency precondition (gate-phase5-ndocs-floor): both a
    pre-registered floor on the SOURCE snapshot's prod-accumulated ``n_docs``
    and a ceiling on its df singleton share must hold, or the verdict is
    ``insufficient data``, never pass/fail. The quantities are read from the
    source DB before the runner window-populates its working copy (design D1);
    a missing thresholds key is an unregistered bar and also reads as
    ``insufficient data``, never a code default (design D4).
    """
    if arm.shared_with is not None:
        return INSUFFICIENT, f"arm resolves to the same overrides as {arm.shared_with}"

    config = thresholds.get("phase5_promotion", {})
    base_values = metric_values(baseline.records)
    arm_values = metric_values(arm.records)
    parts: list[str] = []
    failures: list[str] = []
    missing: list[str] = []
    _phase5_corpus_precondition(config, df_facts, parts=parts, missing=missing)

    present = _mean_telemetry(arm, "seeded_present_rate")
    win = _mean_telemetry(arm, "seeded_win_rate_given_present")
    present_min = float(config.get("seeded_present_rate_min", 0.30))
    win_min = float(config.get("seeded_win_rate_given_present_min", 0.40))
    present_txt = f"{present:.0%}" if present is not None else INSUFFICIENT
    win_txt = f"{win:.0%}" if win is not None else INSUFFICIENT
    parts.append(
        f"seeded present {present_txt} (bar {present_min:.0%}), "
        f"win|present {win_txt} (bar {win_min:.0%})"
    )

    floor = float(config.get("affinity_without_copy_delta_min", 0.0))
    affinity_delta = _delta_part(
        "affinity_without_copy",
        base_values.get("context_affinity_without_copy"),
        arm_values.get("context_affinity_without_copy"),
        parts=parts,
        missing=missing,
    )
    if affinity_delta is not None:
        point, significant = affinity_delta
        if significant and point < floor:
            failures.append("affinity without copies dropped significantly")

    _latency_part(
        arm,
        float(config.get("latency_p95_ms_max", 150)),
        parts=parts,
        missing=missing,
        failures=failures,
    )
    # Failures are still reported (a p95 blowout is real regardless of df), but
    # the missing prod-df line keeps the verdict out of "pass" by construction.
    return _verdict(parts, failures, missing)


def _phase9_arm_verdict(
    baseline: ConfigRun,
    arm: ConfigRun,
    thresholds: dict[str, Any],
    solo_rating: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Six-part Phase 9 gate for one arm (verdict, detail).

    Two-sided on quality by design: this is the first phase whose target metric
    (diversity) has a mechanism to move, so the risk is no longer "nothing
    happens" but "diversity bought with incoherence". A one-sided distinct bar
    would pass exactly that trade.

    Condition 3 bounds the CI lower bound rather than the point estimate — see
    ``_delta_ci_low``. Condition 6 (connectedness) comes from the solo rating
    round — see ``_connectedness_part``.
    """
    if arm.shared_with is not None:
        return INSUFFICIENT, f"arm resolves to the same overrides as {arm.shared_with}"

    config = thresholds.get("phase9_interp", {})
    base_values = metric_values(baseline.records)
    arm_values = metric_values(arm.records)
    parts: list[str] = []
    failures: list[str] = []
    missing: list[str] = []

    base_replies = [r.reply_content for r in baseline.records if r.success]
    arm_replies = [r.reply_content for r in arm.records if r.success]
    for n_gram, key in ((2, "distinct2_delta_min"), (3, "distinct3_delta_min")):
        minimum = float(config.get(key, 0.0))
        result = distinct_delta_ci(base_replies, arm_replies, n_gram)
        if result is None:
            missing.append(f"distinct-{n_gram} basis")
            continue
        point, lo, hi, significant = result
        parts.append(
            f"distinct-{n_gram} Δ {_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"
            f"{' *' if significant else ''}"
        )
        if not (significant and point > minimum):
            failures.append(f"distinct-{n_gram} did not rise significantly")

    copy_max = float(config.get("exact_copy_delta_max", 0.0))
    copy_delta = _delta_part(
        "copy",
        base_values.get("exact_context_copy_rate"),
        arm_values.get("exact_context_copy_rate"),
        parts=parts,
        missing=missing,
    )
    if copy_delta is not None:
        point, significant = copy_delta
        if significant and point > copy_max:
            failures.append("copy rose significantly")

    affinity_floor = float(config.get("affinity_without_copy_delta_ci_low_min", -0.02))
    affinity_low = _delta_ci_low(
        "affinity_without_copy",
        base_values.get("context_affinity_without_copy"),
        arm_values.get("context_affinity_without_copy"),
        parts=parts,
        missing=missing,
    )
    if affinity_low is not None and affinity_low <= affinity_floor:
        failures.append(
            f"affinity without copies could be down to {_fmt(affinity_low)}"
        )

    repetition_max = float(config.get("repetition_delta_max", 0.0))
    repetition_delta = _delta_part(
        "repetition",
        base_values.get("repetition_rate"),
        arm_values.get("repetition_rate"),
        parts=parts,
        missing=missing,
    )
    if repetition_delta is not None:
        point, significant = repetition_delta
        if significant and point > repetition_max:
            failures.append("repetition rose significantly")

    # Safety, not a target: the Phase 6 verdict was measured on a different walk
    # topology, and interpolation changes that topology.
    cycle_max = float(config.get("cycle_detection_rate_max", 0.05))
    cycles = arm_values.get("cycle_detection_rate")
    if not cycles:
        missing.append("cycle-detection samples")
    else:
        point, lo, hi = bootstrap_ci(cycles)
        parts.append(f"cycles {_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]")
        if point >= cycle_max:
            failures.append("cycle rate at or above the safety bound")

    _latency_part(
        arm,
        float(config.get("latency_p95_ms_max", 150)),
        parts=parts,
        missing=missing,
        failures=failures,
    )

    # Condition 6. The instrument is the M3R-020 solo protocol, not a rater
    # panel (the phase spec was synced 2026-08-14 away from a panel the project
    # does not have). Three states are kept apart on purpose: no round, an
    # uncountable round, and a round that says the arm lost connectedness. Only
    # the last is a failure; the first two leave the gate at `insufficient
    # data`, never `pass` — an automatic-only verdict would be a green light
    # bought by measuring the easy half.
    _connectedness_part(
        baseline.config_id,
        arm.config_id,
        solo_rating,
        float(config.get("manual_connected_delta_floor", -0.10)),
        parts=parts,
        missing=missing,
        failures=failures,
    )
    return _verdict(parts, failures, missing)


def _connectedness_part(
    baseline_id: str,
    arm_id: str,
    solo_rating: dict[str, Any] | None,
    floor: float,
    *,
    parts: list[str],
    missing: list[str],
    failures: list[str],
) -> None:
    """Condition 6 of the Phase 9 gate, computed from the solo round aggregate."""
    if solo_rating is None:
        missing.append("connectedness round (M3R-020 solo protocol not conducted)")
        return
    if not solo_rating.get("valid", False):
        reasons = "; ".join(solo_rating.get("invalid_reasons") or ["not stated"])
        missing.append(f"connectedness round not countable ({reasons})")
        return
    arms = solo_rating.get("arms", {})
    base_share = (arms.get(baseline_id) or {}).get("connected_share")
    arm_share = (arms.get(arm_id) or {}).get("connected_share")
    if base_share is None or arm_share is None:
        missing.append(f"connectedness ratings for {baseline_id}/{arm_id}")
        return
    delta = float(arm_share) - float(base_share)
    parts.append(
        f"connected {float(arm_share):.0%} vs {float(base_share):.0%} "
        f"(Δ {_fmt(delta)})"
    )
    if delta < floor:
        failures.append("connectedness fell below the pre-registered floor")


def _phase2_arm_verdict(
    baseline: ConfigRun, arm: ConfigRun, thresholds: dict[str, Any]
) -> tuple[str, str]:
    """Four-part Phase 2 gate for one arm (verdict, detail).

    All four parts must pass. Two are "must not worsen" (copy, affinity) where
    only a *significant* move against us fails; two are "must improve"
    (distinct-2/3) where an insignificant move is a fail — an unmeasurable
    effect is not the diversity this phase promised.
    """
    if arm.shared_with is not None:
        return INSUFFICIENT, f"arm resolves to the same overrides as {arm.shared_with}"

    config = thresholds.get("phase2_entropy", {})
    base_values = metric_values(baseline.records)
    arm_values = metric_values(arm.records)
    parts: list[str] = []
    failures: list[str] = []
    missing: list[str] = []

    copy_max = float(config.get("exact_copy_delta_max", 0.0))
    copy_delta = _delta_part(
        "copy",
        base_values.get("exact_context_copy_rate"),
        arm_values.get("exact_context_copy_rate"),
        parts=parts,
        missing=missing,
    )
    if copy_delta is not None:
        point, significant = copy_delta
        if significant and point > copy_max:
            failures.append("copy rose significantly")

    base_replies = [r.reply_content for r in baseline.records if r.success]
    arm_replies = [r.reply_content for r in arm.records if r.success]
    for n, key in ((2, "distinct2_delta_min"), (3, "distinct3_delta_min")):
        minimum = float(config.get(key, 0.0))
        result = distinct_delta_ci(base_replies, arm_replies, n)
        if result is None:
            missing.append(f"distinct-{n} basis")
            continue
        point, lo, hi, significant = result
        parts.append(
            f"distinct-{n} Δ {_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"
            f"{' *' if significant else ''}"
        )
        if not (significant and point > minimum):
            failures.append(f"distinct-{n} did not rise significantly")

    floor = float(config.get("affinity_without_copy_delta_floor", 0.0))
    affinity_delta = _delta_part(
        "affinity_without_copy",
        base_values.get("context_affinity_without_copy"),
        arm_values.get("context_affinity_without_copy"),
        parts=parts,
        missing=missing,
    )
    if affinity_delta is not None:
        point, significant = affinity_delta
        if significant and point < floor:
            failures.append("affinity without copies dropped significantly")

    _latency_part(
        arm,
        float(config.get("latency_p95_ms_max", 150)),
        parts=parts,
        missing=missing,
        failures=failures,
    )
    return _verdict(parts, failures, missing)


def _phase3_arm_verdict(
    baseline: ConfigRun, arm: ConfigRun, thresholds: dict[str, Any]
) -> tuple[str, str]:
    """Five-part Phase 3 gate for one arm (verdict, detail).

    Two-sided by design (design D9): freshness must rise significantly, while
    historical memes, copying and copy-free topicality must not worsen
    significantly. A one-sided freshness bar would pass an arm that simply
    forgets the chat.
    """
    if arm.shared_with is not None:
        return INSUFFICIENT, f"arm resolves to the same overrides as {arm.shared_with}"

    config = thresholds.get("phase3_temporal", {})
    base_values = metric_values(baseline.records)
    arm_values = metric_values(arm.records)
    parts: list[str] = []
    failures: list[str] = []
    missing: list[str] = []

    fresh_min = float(config.get("freshness_delta_min", 0.0))
    fresh_delta = _delta_part(
        "freshness",
        base_values.get("freshness_reflection"),
        arm_values.get("freshness_reflection"),
        parts=parts,
        missing=missing,
    )
    if fresh_delta is not None:
        point, significant = fresh_delta
        if not (significant and point > fresh_min):
            failures.append("freshness did not rise significantly")

    meme_floor = float(config.get("historical_meme_delta_floor", 0.0))
    meme_delta = _delta_part(
        "historical_meme",
        base_values.get("historical_meme_rate"),
        arm_values.get("historical_meme_rate"),
        parts=parts,
        missing=missing,
    )
    if meme_delta is not None:
        point, significant = meme_delta
        if significant and point < meme_floor:
            failures.append("historical memes dropped significantly")

    copy_max = float(config.get("exact_copy_delta_max", 0.0))
    copy_delta = _delta_part(
        "copy",
        base_values.get("exact_context_copy_rate"),
        arm_values.get("exact_context_copy_rate"),
        parts=parts,
        missing=missing,
    )
    if copy_delta is not None:
        point, significant = copy_delta
        if significant and point > copy_max:
            failures.append("copy rose significantly")

    floor = float(config.get("affinity_without_copy_delta_floor", 0.0))
    affinity_delta = _delta_part(
        "affinity_without_copy",
        base_values.get("context_affinity_without_copy"),
        arm_values.get("context_affinity_without_copy"),
        parts=parts,
        missing=missing,
    )
    if affinity_delta is not None:
        point, significant = affinity_delta
        if significant and point < floor:
            failures.append("affinity without copies dropped significantly")

    _latency_part(
        arm,
        float(config.get("latency_p95_ms_max", 150)),
        parts=parts,
        missing=missing,
        failures=failures,
    )
    return _verdict(parts, failures, missing)


def _phase4_arm_verdict(
    baseline: ConfigRun,
    arm: ConfigRun,
    thresholds: dict[str, Any],
    manual: dict[str, Any] | None,
) -> tuple[str, str]:
    """Phase 4 gate for one arm (verdict, detail).

    The primary condition is a human one — the share of top memes rated genuine
    (doc 05 §5) — and it is deliberately not computable here. Without it the
    verdict is `insufficient data` even when every automatic condition passes:
    this phase's claim is "the bot sounds like this chat", and no metric in this
    harness can see that.
    """
    if arm.shared_with is not None:
        return INSUFFICIENT, f"arm resolves to the same overrides as {arm.shared_with}"

    config = thresholds.get("phase4_memes", {})
    base_values = metric_values(baseline.records)
    arm_values = metric_values(arm.records)
    parts: list[str] = []
    failures: list[str] = []
    missing: list[str] = []

    share_min = float(config.get("manual_real_share_min", 0.70))
    rated_min = int(config.get("manual_rated_min", 20))
    control_floor = float(config.get("manual_control_delta_floor", 0.0))
    decoy_max = float(config.get("manual_decoy_false_positive_max", 0.20))
    if manual is None:
        missing.append("manual top-meme rating (doc 05 §5)")
    else:
        rated = int(manual.get("rated", 0))
        real = int(manual.get("real", 0))
        share = real / rated if rated else 0.0
        control_rated = int(manual.get("control_rated", 0))
        control_real = int(manual.get("control_real", 0))
        control_share = control_real / control_rated if control_rated else 0.0
        decoy_rated = int(manual.get("decoy_rated", 0))
        decoy_real = int(manual.get("decoy_real", 0))
        decoy_share = decoy_real / decoy_rated if decoy_rated else 0.0
        raters = int(manual.get("raters", 0))
        agreement = manual.get("agreement")
        agreement_text = (
            f", agreement {float(agreement):.2f}"
            if agreement is not None and raters > 1
            else ", agreement unavailable (single rater)"
            if raters == 1
            else ""
        )
        parts.append(
            f"manual: meme {real}/{rated} genuine ({share:.0%}, bar {share_min:.0%}) "
            f"vs frequency control {control_real}/{control_rated} "
            f"({control_share:.0%}), Δ {share - control_share:+.0%}; "
            f"decoys {decoy_real}/{decoy_rated} ({decoy_share:.0%}); "
            f"{raters} rater(s){agreement_text}"
        )
        # Validity first: if the decoys were rated genuine this often, neither
        # share means anything and the round is undecided rather than failed.
        if decoy_rated and decoy_share > decoy_max:
            missing.append(
                f"a usable rating round (decoy false-positive {decoy_share:.0%} "
                f"over the {decoy_max:.0%} bar — the ratings are noise)"
            )
        elif rated < rated_min:
            failures.append(f"only {rated} memes rated, protocol requires {rated_min}")
        else:
            if share < share_min:
                failures.append("genuine share below the bar")
            if control_rated and (share - control_share) < control_floor:
                failures.append(
                    "meme ranking scored below the frequency selection it replaces"
                )
            if not control_rated:
                missing.append("frequency-control half of the rating round")

    copy_max = float(config.get("exact_copy_delta_max", 0.0))
    copy_delta = _delta_part(
        "copy",
        base_values.get("exact_context_copy_rate"),
        arm_values.get("exact_context_copy_rate"),
        parts=parts,
        missing=missing,
    )
    if copy_delta is not None:
        point, significant = copy_delta
        if significant and point > copy_max:
            failures.append("copy rose significantly")

    floor = float(config.get("affinity_without_copy_delta_floor", 0.0))
    affinity_delta = _delta_part(
        "affinity_without_copy",
        base_values.get("context_affinity_without_copy"),
        arm_values.get("context_affinity_without_copy"),
        parts=parts,
        missing=missing,
    )
    if affinity_delta is not None:
        point, significant = affinity_delta
        if significant and point < floor:
            failures.append("affinity without copies dropped significantly")

    _latency_part(
        arm,
        float(config.get("latency_p95_ms_max", 150)),
        parts=parts,
        missing=missing,
        failures=failures,
    )
    return _verdict(parts, failures, missing)


def _phase6_verdict(
    baseline: ConfigRun | None, thresholds: dict[str, Any]
) -> tuple[str, str]:
    """Phase 6 anti-cycle gate (verdict, detail) — two-dimensional (ADR-015).

    The gate opens only if cycles are BOTH frequent (`cycle_detection_rate`)
    AND harmful (`cycle_harm_rate`); non-exceedance of either closes the phase
    without implementation (roadmap Phase 6). So a detection arm whose whole
    confidence interval sits below its threshold makes the conjunction
    impossible, and the gate resolves to `close` WITHOUT the manual harm round —
    the same "a demonstrated miss outranks a missing part" rule the other gates
    use, applied to a conjunction (change: markov2r-phase6-anticycle-verdict).
    """
    phase6 = thresholds.get("phase6_anticycle", {})
    if baseline is None:
        return INSUFFICIENT, "no baseline run"
    rate = metric_values(baseline.records).get("cycle_detection_rate")
    if not rate:
        return INSUFFICIENT, "no cycle-detection samples"
    detect_min = float(phase6.get("cycle_detection_rate_min", 0.05))
    point, lo, hi = bootstrap_ci(rate)
    interval = f"{_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"
    if hi < detect_min:
        # The detection arm is decisively below its bar: the whole interval is
        # under the threshold, so cycles are not frequent at the protocol's
        # confidence and the AND-gate cannot open. Phase closes; the harm round
        # would only measure the harm of a ~detection-rate phenomenon and
        # cannot change this.
        return (
            "close",
            f"cycle_detection_rate {interval} wholly below the "
            f"{detect_min:.2f} threshold — cycles are not frequent, the "
            "rate×harm conjunction cannot hold, so Phase 6 closes without "
            "implementation (M2R-600/610 not built); the manual harm round is "
            "not required (ADR-015)",
        )
    # Detection is near or above its bar: the harm arm now decides, and its
    # manual component is missing.
    return (
        INSUFFICIENT,
        f"cycle_detection_rate {interval} (threshold {detect_min:.2f}) is not "
        "decisively below the bar; cycle_harm_rate has only its automatic "
        "component until a manual round (doc 05 §5) is conducted",
    )


def _structural_escape_rows(
    runs: dict[str, ConfigRun], thresholds: dict[str, Any]
) -> list[tuple[str, str, str]]:
    """One structural-escape row per configuration (M3R-011).

    Per configuration rather than per arm-against-C0: the headline claim is
    absolute ("this input yields >= 2 different trajectories in the window"),
    not comparative. A route that doubles the window count while staying under
    two has still not opened the space, and a delta would hide that.

    The two numbers are checked together on purpose — the pool floor is what
    stops a change from buying window diversity by shrinking the pool.
    """
    config = thresholds.get("structural_escape", {})
    window_min = float(config.get("window_escape_min", 2.0))
    pool_min = float(config.get("pool_ecb_min", 4.0))
    rows: list[tuple[str, str, str]] = []
    for config_id in sorted(runs):
        values = metric_values(runs[config_id].records)
        pool = values.get("structural_pool_ecb")
        window = values.get("structural_window_escape")
        parts: list[str] = []
        failures: list[str] = []
        missing: list[str] = []
        if not pool or not window:
            missing.append("structural samples")
        else:
            pool_mean, pool_lo, pool_hi = bootstrap_ci(pool)
            window_mean, window_lo, window_hi = bootstrap_ci(window)
            # Always both, never one: the gap between them is the finding.
            parts.append(
                f"window escape {_fmt(window_mean)} [{_fmt(window_lo)}, "
                f"{_fmt(window_hi)}] (min {window_min:.1f}); "
                f"pool ECB {_fmt(pool_mean)} [{_fmt(pool_lo)}, {_fmt(pool_hi)}] "
                f"(floor {pool_min:.1f})"
            )
            # Доля рядом с абсолютом. Пол `pool_ecb_min` — счёт траекторий, и
            # он тривиально берётся растущим пулом: seeded добавляет
            # кандидатов сверх `effective_target`, каждый маршрут Track B
            # добавит ещё. Порог не двигается (пре-регистрация), но читатель
            # обязан видеть величину, которую порог должен был мерить, —
            # иначе первый же маршрутный отчёт покажет зелёный ECB, ничего не
            # означающий. Имя отдельное, ровно как «доля входов ниже 2».
            share = values.get("structural_pool_ecb_share")
            if share:
                share_mean, share_lo, share_hi = bootstrap_ci(share)
                parts.append(
                    f"pool ECB share {_fmt(share_mean)} "
                    f"[{_fmt(share_lo)}, {_fmt(share_hi)}] "
                    "(доля различных траекторий в пуле; порога нет — "
                    "справочно к полу выше)"
                )
            # Distribution, not only the mean: it is what says whether the
            # verdict hangs on the overlap threshold. A run whose window count
            # is 1 almost everywhere would not change its verdict at any
            # neighbouring threshold, and a reader can see that instead of
            # having to re-run with another value.
            buckets = Counter(int(value) for value in window)
            spread = ", ".join(
                f"{count}:{buckets[count] / len(window):.0%}"
                for count in sorted(buckets)
            )
            parts.append(f"window distribution {spread}")
            if window_mean < window_min:
                failures.append("window escape below the registered minimum")
            if pool_mean < pool_min:
                failures.append("pool ECB below the safety floor")
        verdict, detail = _verdict(parts, failures, missing)
        rows.append((f"structural_escape[{config_id}]", verdict, detail))
    if not rows:
        rows.append(("structural_escape", INSUFFICIENT, "no configurations in this run"))
    return rows


def _apply_mode_requirement(
    row: tuple[str, str, str], thresholds: dict[str, Any], context_mode: str
) -> tuple[str, str, str]:
    """Downgrade a two-mode gate measured in one mode (M3R-140).

    Applied after the gate computed its numbers, not instead of it: the ctx half
    of a two-mode gate is worth printing, it is just not worth calling a pass.
    Declared per gate by ``requires_both_modes`` in the thresholds file, so the
    rule reaches only the gates registered under it — the closed phases are not
    retroactively reopened as `insufficient data`.
    """
    gate, verdict, detail = row
    name = gate.split("[", 1)[0]
    if not thresholds.get(name, {}).get("requires_both_modes"):
        return row
    absent = "noctx" if context_mode == "ctx" else "ctx"
    return (
        gate,
        INSUFFICIENT,
        f"{detail}; gate requires both context modes, this run measured "
        f"{context_mode} only ({absent} not measured)",
    )


def _meme_regression_rows(
    runs: dict[str, ConfigRun],
    prompt_set: PromptSet,
    thresholds: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """(gate, verdict, detail) rows of the meme-regression check (M3R-130).

    Relative to the baseline, per arm, over a set that carries a support floor.
    The previous form asked whether the BASELINE reproduced every meme of a
    floor-less list — luck about the set, computed for C0 only and labelled
    "informational" because it could not gate anything.
    """
    config = thresholds.get("meme_regression", {})
    tolerance_raw = config.get("reproduced_tolerance")
    min_size_raw = config.get("set_min_size")
    total = len(prompt_set.memes)
    name = "meme_regression"
    if tolerance_raw is None or min_size_raw is None:
        # Pre-registration rule: a bar that is not in the thresholds file is not
        # a bar (precedent: gate-phase5-ndocs-floor design D4).
        return [(
            name,
            INSUFFICIENT,
            "gate thresholds are not registered "
            "(reproduced_tolerance / set_min_size)",
        )]
    tolerance, min_size = float(tolerance_raw), int(min_size_raw)
    if total < min_size:
        return [(
            name,
            INSUFFICIENT,
            f"meme set has {total} entries, below the registered minimum "
            f"{min_size} (support floor "
            f"{config.get('support_min', '?')}, prompt set {prompt_set.version})"
            " — a share over this few memes is noise, not evidence",
        )]
    baseline = runs.get("C0")
    if baseline is None:
        return [(name, INSUFFICIENT, "no baseline run to compare against")]
    base_hits, _ = meme_regression(baseline.records, total)
    base_share = base_hits / total
    rows: list[tuple[str, str, str]] = []
    for arm_id in sorted(runs):
        arm = runs[arm_id]
        hits, missing = meme_regression(arm.records, total)
        share = hits / total
        detail = (
            f"{hits}/{total} memes reproduced ({share:.0%}); "
            f"C0 {base_hits}/{total} ({base_share:.0%}), "
            f"tolerance {tolerance:.0%}; prompt set {prompt_set.version}"
        )
        if arm_id == "C0":
            rows.append((f"{name}[C0]", "baseline", detail))
            continue
        if share < base_share - tolerance:
            rows.append((
                f"{name}[{arm_id}]",
                "fail",
                f"{detail}; never reproduced (indices): {missing}",
            ))
        else:
            rows.append((f"{name}[{arm_id}]", "pass", detail))
    return rows


def evaluate_gates(
    runs: dict[str, ConfigRun],
    thresholds: dict[str, Any],
    manual_rating: dict[str, Any] | None = None,
    context_mode: str = "ctx",
    df_facts: DfCorpusFacts | None = None,
    solo_rating: dict[str, Any] | None = None,
) -> list[tuple[str, str, str]]:
    """(gate, verdict, detail) rows. Phase 0: most gates lack their data by
    construction — that is reported, never guessed."""
    rows: list[tuple[str, str, str]] = []
    baseline = runs.get("C0")

    phase2_arms = sorted(
        arm for arm in runs if arm.startswith(PHASE2_ARM_PREFIX)
    )
    if baseline is None or not phase2_arms:
        rows.append(
            (
                "phase2_entropy",
                INSUFFICIENT,
                "no Phase 2 arm in this run (entropy sampling not enabled)",
            )
        )
    else:
        for arm_id in phase2_arms:
            verdict, detail = _phase2_arm_verdict(baseline, runs[arm_id], thresholds)
            rows.append((f"phase2_entropy[{arm_id}]", verdict, detail))

    phase3_arms = sorted(arm for arm in runs if arm.startswith(PHASE3_ARM_PREFIX))
    if baseline is None or not phase3_arms:
        rows.append(
            (
                "phase3_temporal",
                INSUFFICIENT,
                "no Phase 3 arm in this run (temporal blend not enabled)",
            )
        )
    else:
        for arm_id in phase3_arms:
            verdict, detail = _phase3_arm_verdict(baseline, runs[arm_id], thresholds)
            rows.append((f"phase3_temporal[{arm_id}]", verdict, detail))

    phase4_arms = sorted(arm for arm in runs if arm.startswith(PHASE4_ARM_PREFIX))
    if baseline is None or not phase4_arms:
        rows.append(
            (
                "phase4_memes",
                INSUFFICIENT,
                "no Phase 4 arm in this run (meme scoring not enabled)",
            )
        )
    else:
        for arm_id in phase4_arms:
            verdict, detail = _phase4_arm_verdict(
                baseline, runs[arm_id], thresholds, manual_rating
            )
            rows.append((f"phase4_memes[{arm_id}]", verdict, detail))

    phase5_arms = sorted(arm for arm in runs if arm.startswith(PHASE5_ARM_PREFIX))
    if baseline is None or not phase5_arms:
        rows.append(
            (
                "phase5_promotion",
                INSUFFICIENT,
                "no Phase 5 arm in this run (seeded generation not enabled)",
            )
        )
    else:
        for arm_id in phase5_arms:
            verdict, detail = _phase5_arm_verdict(
                baseline, runs[arm_id], thresholds, df_facts
            )
            rows.append((f"phase5_promotion[{arm_id}]", verdict, detail))

    phase9_arms = sorted(arm for arm in runs if arm.startswith(PHASE9_ARM_PREFIX))
    if baseline is None or not phase9_arms:
        rows.append(
            (
                "phase9_interp",
                INSUFFICIENT,
                "no Phase 9 arm in this run" if baseline else "no baseline run",
            )
        )
    else:
        for arm_id in phase9_arms:
            verdict, detail = _phase9_arm_verdict(
                baseline, runs[arm_id], thresholds, solo_rating
            )
            rows.append((f"phase9_interp[{arm_id}]", verdict, detail))

    rows.extend(_structural_escape_rows(runs, thresholds))

    rows.append(("phase6_anticycle", *_phase6_verdict(baseline, thresholds)))

    phase7 = thresholds.get("phase7_order4", {})
    shadow_eligible = 0
    shadow_selected_share: float | None = None
    if baseline is not None and baseline.telemetry:
        eligible_counts = [
            int(snapshot.get("shadow_order4_eligible") or 0)
            for snapshot in baseline.telemetry
        ]
        shares = [
            float(share)
            for snapshot in baseline.telemetry
            if (share := snapshot.get("shadow_order4_selected_share")) is not None
        ]
        shadow_eligible = sum(eligible_counts)
        if shares:
            shadow_selected_share = mean(shares)
    if shadow_selected_share is None or shadow_eligible < 1000:
        rows.append(
            (
                "phase7_order4",
                INSUFFICIENT,
                f"shadow data: {shadow_eligible} eligible steps "
                "(need >= 1000 for a verdict; estimator=window)",
            )
        )
    else:
        threshold = float(phase7.get("order4_selected_share_min", 0.10))
        verdict = "pass" if shadow_selected_share >= threshold else "fail"
        rows.append(
            (
                "phase7_order4",
                verdict,
                f"shadow order-4 share {shadow_selected_share:.1%} vs "
                f"threshold {threshold:.0%} over {shadow_eligible} eligible "
                "steps (estimator=window — conservative lower bound); the "
                "exact-copy condition is checked at Phase 7 proposal time",
            )
        )

    perf = thresholds.get("performance", {})
    budget = float(perf.get("generation_p95_ms_max", 150))
    if baseline is not None:
        p95 = latency_percentiles(baseline.records)["latency_p95"]
        if p95 is None:
            rows.append(("performance.generation_p95", INSUFFICIENT, "no latencies"))
        else:
            verdict = "pass" if p95 <= budget else "fail"
            rows.append(
                (
                    "performance.generation_p95",
                    verdict,
                    f"C0 p95 = {p95:.1f} ms (budget {budget:.0f} ms)",
                )
            )
    rows.append(
        (
            "performance.lookup_p95",
            INSUFFICIENT,
            "distribution-lookup instrumentation lands in Phase 1",
        )
    )
    return [_apply_mode_requirement(row, thresholds, context_mode) for row in rows]


def manual_summary(manual: dict[str, Any] | None) -> list[str]:
    """Report section describing the manual rating round that fed this run.

    Only the aggregate keys are read: the rated items are verbatim chat
    phrases, and a committed report carries counts, not phrases (spec:
    `generation-eval` — "Manual ratings are versioned without leaking chat
    content"). Unknown keys in the aggregate file — a `_comment`, say — are
    never echoed, so the section cannot leak whatever a note happens to hold.

    The "not conducted" wording is reachable only for a genuinely absent
    rating: a report that describes its own inputs wrongly discredits its
    other sections too, and this one used to say it unconditionally.
    """
    if manual is None:
        return ["Not conducted in this run (first required at the Phase 4 gate)."]

    def counts(rated_key: str, real_key: str) -> tuple[int, int, str]:
        rated = int(manual.get(rated_key, 0))
        real = int(manual.get(real_key, 0))
        share = f"{real / rated:.0%}" if rated else INSUFFICIENT
        return rated, real, share

    rated, real, share = counts("rated", "real")
    control_rated, control_real, control_share = counts("control_rated", "control_real")
    decoy_rated, decoy_real, decoy_share = counts("decoy_rated", "decoy_real")
    raters = int(manual.get("raters", 0))
    agreement = manual.get("agreement")
    agreement_text = (
        f"{float(agreement):.2f}"
        if agreement is not None and raters > 1
        else "unavailable (single rater)"
        if raters == 1
        else INSUFFICIENT
    )
    version = manual.get("ranking_version")

    lines = [
        f"Raters: {raters}; inter-rater agreement: {agreement_text}.",
        "",
        "| source | rated | genuine | share |",
        "|---|---|---|---|",
        f"| association ranking | {rated} | {real} | {share} |",
        f"| frequency control | {control_rated} | {control_real} | {control_share} |",
        f"| decoys | {decoy_rated} | {decoy_real} | {decoy_share} |",
    ]
    if version:
        lines.extend(["", f"Ranking version rated: `{version}`."])
    return lines


def build_report(
    *,
    runs: dict[str, ConfigRun],
    skipped: list[str],
    prompt_set: PromptSet,
    thresholds: dict[str, Any],
    snapshot_label: str,
    seeds: tuple[int, ...],
    generations: int,
    revision: str,
    date: str,
    notes: list[str],
    manual_rating: dict[str, Any] | None = None,
    context_mode: str = "ctx",
    df_facts: DfCorpusFacts | None = None,
    solo_rating: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(
        f"# Eval report {date} snapshot={snapshot_label} "
        f"prompts={prompt_set.version} seeds={','.join(map(str, seeds))} "
        f"mode={context_mode}"
    )
    lines.append("")
    lines.append(f"Revision: `{revision}`. Generations per configuration: {generations}.")
    # M3R-140: the mode is stated twice — in the title line a reader copies and
    # in a sentence they cannot skim past. Every number below belongs to this
    # mode alone; a gate that requires both is reported `insufficient data`.
    lines.append(
        f"Context mode: **{context_mode}** — "
        + (
            "the prompt is supplied to the generator as context."
            if context_mode == "ctx"
            else "no context tokens are supplied; the prompt only selects the "
            "generation and seeds the RNG."
        )
    )
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## Config matrix")
    lines.append("")
    for config_id, run in sorted(runs.items()):
        shared = f" — results shared with {run.shared_with}" if run.shared_with else ""
        lines.append(f"- **{config_id}**: {len(run.records)} generations{shared}")
    if skipped:
        lines.append(
            f"- unavailable (feature not implemented yet): {', '.join(sorted(skipped))}"
        )
    lines.append("")

    ordered = sorted(runs)
    baseline_values = metric_values(runs["C0"].records) if "C0" in runs else {}
    lines.append("## Metrics table")
    lines.append("")
    lines.append(
        "Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], "
        "`*` = significant (interval excludes 0, doc 05 §4)."
    )
    lines.append("")
    lines.append(
        "> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам "
        "и сидам, поэтому ресэмплируются пары наблюдений, а не два арма "
        "независимо. С отчётами до 2026-08-26 ширина интервалов "
        "**несопоставима** — там дельта считалась независимым ресэмплингом и "
        "интервал был шире истинного тем сильнее, чем выше корреляция армов. "
        "Точечные оценки сопоставимы: они не изменились. "
        "**distinct-2/3 парность НЕ получили** — их дельта считается по целым "
        "ответам (`distinct_delta_ci`) и остаётся непарной; там интервал "
        "по-прежнему шире истинного, то есть вердикт консервативен, но "
        "сравнивать его ширину с таблицей выше нельзя."
    )
    lines.append("")
    columns = ["metric"]
    for config_id in ordered:
        columns.append(config_id)
        if config_id != "C0":
            columns.append(f"Δ {config_id} vs C0")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * len(columns))
    per_config_values = {c: metric_values(runs[c].records) for c in ordered}
    for metric in METRIC_ORDER:
        cells: list[str] = []
        for config_id in ordered:
            samples = per_config_values[config_id].get(metric)
            cells.append(_cell(samples))
            if config_id != "C0":
                cells.append(_delta_cell(baseline_values.get(metric), samples))
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    for config_id in ordered:
        run = runs[config_id]
        replies = [r.reply_content for r in run.records if r.success]
        d2, basis2 = distinct_n(replies, 2)
        d3, basis3 = distinct_n(replies, 3)
        latencies = latency_percentiles(run.records)
        hit_rates = [
            snapshot["cache_hit_rate"]
            for snapshot in run.telemetry
            if snapshot.get("cache_hit_rate") is not None
        ]
        hit_line = (
            f"{mean([float(rate) for rate in hit_rates]):.0%}"
            if hit_rates
            else INSUFFICIENT
        )
        # M2R-100: the entropy the sampler saw and the temperature it applied.
        # The pivot is set from the first of these, so a later reader can audit
        # that it was measured rather than picked.
        entropies = [
            float(value)
            for snapshot in run.telemetry
            if (value := snapshot.get("mean_normalized_entropy")) is not None
        ]
        temperatures = [
            float(value)
            for snapshot in run.telemetry
            if (value := snapshot.get("mean_applied_temperature")) is not None
        ]
        branchings = [
            float(value)
            for snapshot in run.telemetry
            if (value := snapshot.get("mean_branching")) is not None
        ]
        entropy_line = (
            f"{mean(entropies):.3f} (branching {mean(branchings):.2f})"
            if entropies and branchings
            else INSUFFICIENT
        )
        temperature_line = (
            f"{mean(temperatures):.2f}" if temperatures else INSUFFICIENT
        )
        shadow_shares = [
            snapshot["shadow_order4_selected_share"]
            for snapshot in run.telemetry
            if snapshot.get("shadow_order4_selected_share") is not None
        ]
        # M2R-210: intent (alpha) is in the matrix; these two are the effect.
        coverages = [
            float(value)
            for snapshot in run.telemetry
            if (value := snapshot.get("blend_step_coverage")) is not None
        ]
        displacements = [
            float(value)
            for snapshot in run.telemetry
            if (value := snapshot.get("mean_blend_displacement")) is not None
        ]
        blend_line = (
            f"coverage {mean(coverages):.1%}, shift {mean(displacements):.4f}"
            if coverages and displacements
            else INSUFFICIENT
        )
        # M2R-901: same intent-versus-effect pair for the order interpolation.
        # Was collected from the first day of Phase 9 and printed nowhere, so the
        # phase's central claim — "the merge fired, the replies did not move" —
        # had to be argued from branching and entropy instead of from the
        # mechanism's own counters.
        interp_coverages = [
            float(value)
            for snapshot in run.telemetry
            if (value := snapshot.get("interp_step_coverage")) is not None
        ]
        interp_displacements = [
            float(value)
            for snapshot in run.telemetry
            if (value := snapshot.get("mean_interp_displacement")) is not None
        ]
        interp_line = (
            f"coverage {mean(interp_coverages):.1%}, "
            f"shift {mean(interp_displacements):.4f}"
            if interp_coverages and interp_displacements
            else INSUFFICIENT
        )
        shadow_line = (
            f"{mean([float(share) for share in shadow_shares]):.1%} "
            "(estimator=window)"
            if shadow_shares
            else INSUFFICIENT
        )
        lines.append("")
        lines.append(
            f"{config_id}: distinct-2 = {d2 and _fmt(d2)} (basis {basis2}), "
            f"distinct-3 = {d3 and _fmt(d3)} (basis {basis3}) — type/token ratios, "
            "comparable only at equal basis; их дельта считается НЕпарным "
            "бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — "
            "интервал шире истинного, вердикт консервативен; "
            f"latency p50/p95 = {latencies['latency_p50']:.1f}/"
            f"{latencies['latency_p95']:.1f} ms; cache_hit_rate: {hit_line}; "
            f"mean normalized entropy: {entropy_line}; "
            f"mean applied temperature: {temperature_line}; "
            f"temporal blend: {blend_line}; "
            f"order interpolation: {interp_line}; "
            f"shadow order-4 share: {shadow_line}; storage_delta: n/a."
        )
    lines.append("")

    lines.append("## Per-category breakdown")
    lines.append("")
    lines.append("| config | category | n | success | copy | repetition | affinity |")
    lines.append("|---|---|---|---|---|---|---|")
    for config_id in ordered:
        if runs[config_id].shared_with:
            continue
        by_category: dict[str, list[Any]] = {}
        for record in runs[config_id].records:
            by_category.setdefault(record.category, []).append(record)
        for category, records in sorted(by_category.items()):
            values = metric_values(records)
            lines.append(
                f"| {config_id} | {category} | {len(records)} "
                f"| {_cell(values['generation_success_rate'])} "
                f"| {_cell(values['exact_context_copy_rate'])} "
                f"| {_cell(values['repetition_rate'])} "
                f"| {_cell(values['context_affinity'])} |"
            )
    lines.append("")

    # M3R-103 (reporting half): the third axis — configuration x route — as its
    # own table rather than extra metric rows. Without it "quality rose by X%"
    # has no answer to WHICH mechanism moved it (roadmap: not accepted as a
    # result).
    lines.append("## Per-route breakdown (M3R-103)")
    lines.append("")
    lines.append(
        "Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его "
        "атрибутировал генератор при создании. Два знаменателя раздельно: "
        "**доля пула** — кандидаты маршрута среди всех кандидатов генерации; "
        "**присутствие** — доля генераций, где маршрут положил хотя бы одного "
        "кандидата; **win given present** — доля побед среди них. Affinity без "
        "копий и copy — по ответам, которые выиграл маршрут. Латентность — "
        "средняя по генерациям с маршрутом в пуле / без него: верхняя оценка "
        "цены маршрута, не измерение его шага. Отклонения — до пула, по классам "
        f"M3R-021, из телеметрии генератора. `{ROUTE_NOT_ATTEMPTED}` — механизм "
        "маршрута в этой конфигурации не запускался (не то же, что «запускался и "
        "ничего не произвёл»)."
    )
    lines.append("")
    lines.append(
        "| config | route | attempts | pool share | presence | win given present "
        "| winners' affinity w/o copy | winners' copy | latency with / without, ms "
        "| rejected before pool (F-classes) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for config_id in ordered:
        if runs[config_id].shared_with:
            continue
        lines.extend(_route_rows(config_id, runs[config_id]))
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    for gate, verdict, detail in evaluate_gates(
        runs, thresholds, manual_rating, context_mode, df_facts, solo_rating
    ):
        lines.append(f"- **{gate}**: {verdict} — {detail}")
    for row in _meme_regression_rows(runs, prompt_set, thresholds):
        lines.append(f"- **{row[0]}**: {row[1]} — {row[2]}")
    lines.append("")

    lines.append("## Manual eval summary")
    lines.append("")
    lines.extend(manual_summary(manual_rating))
    lines.append("")

    lines.append("## Verdict per phase")
    lines.append("")
    lines.append(
        "- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases "
        "measure against these numbers. Temporal metrics report "
        f"`{INSUFFICIENT}` until Phase 3 accumulates timestamps (audit §10.1)."
    )
    lines.append("")
    return "\n".join(lines)
