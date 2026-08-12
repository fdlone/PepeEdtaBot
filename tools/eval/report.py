"""Markdown report and gate evaluation (doc 05 §4, §6).

The report is the decision artifact: stored under ``docs/eval_reports/`` and
cited by phase verdicts (including "closed without implementation"). Gate
verdicts are computed strictly from ``eval_thresholds.yaml`` — pass / fail /
insufficient data, with the numbers that produced them.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from .bootstrap import bootstrap_ci, delta_ci, distinct_delta_ci
from .metrics import distinct_n, latency_percentiles, meme_regression, metric_values
from .prompts import PromptSet
from .run import ConfigRun

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
    if not base or not other:
        return "—"
    point, lo, hi, significant = delta_ci(base, other)
    marker = " *" if significant else ""
    return f"{_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]{marker}"


def metrics_summary(runs: dict[str, ConfigRun]) -> dict[str, Any]:
    """Deterministic metric points per configuration, latency excluded —
    the object the bit-for-bit reproducibility check compares."""
    summary: dict[str, Any] = {}
    for config_id, run in sorted(runs.items()):
        values = metric_values(run.records)
        replies = [r.reply_content for r in run.records if r.success]
        d2, basis2 = distinct_n(replies, 2)
        d3, basis3 = distinct_n(replies, 3)
        summary[config_id] = {
            "shared_with": run.shared_with,
            "n_records": len(run.records),
            "metrics": {
                name: (round(mean(samples), 9) if samples else None)
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
    if not base or not arm:
        missing.append(f"{label} samples")
        return None
    point, lo, hi, significant = delta_ci(base, arm)
    parts.append(
        f"{label} Δ {_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"
        f"{' *' if significant else ''}"
    )
    return point, significant


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
    if manual is None:
        missing.append("manual top-meme rating (doc 05 §5)")
    else:
        rated = int(manual.get("rated", 0))
        real = int(manual.get("real", 0))
        share = real / rated if rated else 0.0
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
            f"manual: {real}/{rated} rated genuine ({share:.0%}, bar {share_min:.0%}), "
            f"{raters} rater(s){agreement_text}"
        )
        if rated < rated_min:
            failures.append(f"only {rated} memes rated, protocol requires {rated_min}")
        elif share < share_min:
            failures.append("genuine share below the bar")

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


def evaluate_gates(
    runs: dict[str, ConfigRun],
    thresholds: dict[str, Any],
    manual_rating: dict[str, Any] | None = None,
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

    rows.append(
        (
            "phase5_promotion",
            INSUFFICIENT,
            "seeded generation does not exist before Phase 5",
        )
    )

    phase6 = thresholds.get("phase6_anticycle", {})
    if baseline is not None:
        values = metric_values(baseline.records)
        rate = values["cycle_detection_rate"]
        detail = (
            f"observed cycle_detection_rate={_cell(rate)} "
            f"(threshold {phase6.get('cycle_detection_rate_min')}); "
            "cycle_harm_rate has only its automatic component until a manual "
            "round (doc 05 §5) is conducted"
        )
    else:
        detail = "no baseline run"
    rows.append(("phase6_anticycle", INSUFFICIENT, detail))

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
    return rows


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
) -> str:
    lines: list[str] = []
    lines.append(
        f"# Eval report {date} snapshot={snapshot_label} "
        f"prompts={prompt_set.version} seeds={','.join(map(str, seeds))}"
    )
    lines.append("")
    lines.append(f"Revision: `{revision}`. Generations per configuration: {generations}.")
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
        "Value [95% CI] per configuration; delta vs C0 [95% CI], `*` = significant "
        "(interval excludes 0, doc 05 §4)."
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
            "comparable only at equal basis; "
            f"latency p50/p95 = {latencies['latency_p50']:.1f}/"
            f"{latencies['latency_p95']:.1f} ms; cache_hit_rate: {hit_line}; "
            f"mean normalized entropy: {entropy_line}; "
            f"mean applied temperature: {temperature_line}; "
            f"temporal blend: {blend_line}; "
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

    lines.append("## Gates")
    lines.append("")
    for gate, verdict, detail in evaluate_gates(runs, thresholds, manual_rating):
        lines.append(f"- **{gate}**: {verdict} — {detail}")
    if "C0" in runs:
        verdict, missing = meme_regression(runs["C0"].records, len(prompt_set.memes))
        if verdict is None:
            lines.append(f"- **meme_regression_pass**: {INSUFFICIENT} — empty meme list")
        else:
            status = "pass" if verdict else "fail"
            detail = (
                "all memes reproduced"
                if verdict
                else f"memes never reproduced (indices): {missing}"
            )
            lines.append(
                f"- **meme_regression_pass (C0, informational at Phase 0)**: "
                f"{status} — {detail} (list of {len(prompt_set.memes)} memes, "
                f"prompt set {prompt_set.version})"
            )
    lines.append("")

    lines.append("## Manual eval summary")
    lines.append("")
    lines.append("Not conducted in this run (first required at the Phase 4 gate).")
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
