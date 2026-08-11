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
# combination, each knob alone, and the flat-temperature control.
PHASE2_ARMS = ("C1", "C1a", "C1b", "C1flat")

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
    base_copy = base_values.get("exact_context_copy_rate")
    arm_copy = arm_values.get("exact_context_copy_rate")
    if base_copy and arm_copy:
        point, lo, hi, significant = delta_ci(base_copy, arm_copy)
        parts.append(
            f"copy Δ {_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"
            f"{' *' if significant else ''}"
        )
        if significant and point > copy_max:
            failures.append("copy rose significantly")
    else:
        missing.append("exact-copy samples")

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
    base_aff = base_values.get("context_affinity_without_copy")
    arm_aff = arm_values.get("context_affinity_without_copy")
    if base_aff and arm_aff:
        point, lo, hi, significant = delta_ci(base_aff, arm_aff)
        parts.append(
            f"affinity_without_copy Δ {_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"
            f"{' *' if significant else ''}"
        )
        if significant and point < floor:
            failures.append("affinity without copies dropped significantly")
    else:
        missing.append("affinity-without-copy samples")

    budget = float(config.get("latency_p95_ms_max", 150))
    p95 = latency_percentiles(arm.records)["latency_p95"]
    if p95 is None:
        missing.append("latencies")
    else:
        parts.append(f"p95 {p95:.1f} ms (budget {budget:.0f})")
        if p95 > budget:
            failures.append("p95 over budget")

    detail = "; ".join(parts) if parts else "nothing computable"
    if missing:
        detail = f"{detail} — missing: {', '.join(missing)}"
    # A demonstrated failure outranks a missing part: an arm that provably
    # raised copying does not become undecided because some other part of the
    # gate lacked data. Only a gate with no failures AND a hole is undecided.
    if failures:
        return "fail", f"{detail} — {'; '.join(failures)}"
    if missing:
        return INSUFFICIENT, detail
    return "pass", detail


def evaluate_gates(
    runs: dict[str, ConfigRun], thresholds: dict[str, Any]
) -> list[tuple[str, str, str]]:
    """(gate, verdict, detail) rows. Phase 0: most gates lack their data by
    construction — that is reported, never guessed."""
    rows: list[tuple[str, str, str]] = []
    baseline = runs.get("C0")

    phase2_arms = [arm for arm in PHASE2_ARMS if arm in runs]
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
        shadow_shares = [
            snapshot["shadow_order4_selected_share"]
            for snapshot in run.telemetry
            if snapshot.get("shadow_order4_selected_share") is not None
        ]
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
    for gate, verdict, detail in evaluate_gates(runs, thresholds):
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
