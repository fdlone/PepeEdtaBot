"""Knob census (M3R-151, change knob-census): which runtime knobs move anything.

Builds its arms from the runtime registry — the two extremes of every
measurable knob's domain (a flipped value for booleans), plus the same extremes
with the parent knob enabled for knobs another knob gates — runs them on a
snapshot in both context modes against one baseline, and classifies every knob
by the rule pre-registered in ``eval_thresholds.yaml`` (``knob_census``):

    dead    not read on any live path (static check, no run)
    gated   inert with the parent at its default, not inert with it enabled
    inert   at every extreme every metric's delta interval sits inside the
            tolerance band, in both modes
    strong  some extreme moves some metric significantly by at least the bar
    weak    everything else (including "not resolved")

Knobs the harness cannot exercise (read only by the reply pipeline or the
handlers) are listed as outside the offline measurement — never as inert.

Usage:
    python -m tools.eval.knob_census plan
    python -m tools.eval.knob_census run --db <copy> --context-mode noctx \\
        --arms C0 --out <dir>/C0_noctx.json
    python -m tools.eval.knob_census launch --db <copy> --out <dir> --workers 16
    python -m tools.eval.knob_census report --inputs <dir> \\
        --out docs/eval_reports/eval_<date>_knob-census.md

The report carries numbers only: no reply text, no n-grams.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.registry import RUNTIME_FIELDS, FieldSpec  # noqa: E402
from tools.eval.bootstrap import delta_ci  # noqa: E402
from tools.eval.config import MatrixConfig, load_thresholds  # noqa: E402
from tools.eval.metrics import latency_percentiles, metric_values  # noqa: E402
from tools.eval.prompts import PROMPTS_PATH, load_prompts  # noqa: E402
from tools.eval.run import CONTEXT_MODES, PROTOCOL_SEEDS, run_matrix  # noqa: E402

# Modules whose reads make a knob measurable by the protocol harness (design
# D1): the harness executes the generation core and nothing above it.
CORE_MODULES = (
    "response_generator",
    "markov",
    "candidate_scorer",
    "slot_mutation",
    "interpolation",
    "temporal",
    "shadow_order",
    "seed",
    "collocations",
)

# Knobs read by the core that the census still cannot sweep, with the reason.
NOT_SWEPT: dict[str, str] = {
    "length_mode_weights": (
        "composite value (three weights); sweep needs a grid of its own"
    ),
    "markov_long_compression": (
        "string enum (log|pow); measured as the parent of its beta"
    ),
    "normalize_lower": (
        "tokenization of the LEARNED corpus, not a generation knob; "
        "the copy is lowercased already"
    ),
    "markov_alpha_sleepy": (
        "mood-gated: the harness runs in the neutral mood, which reads alpha_calm"
    ),
    "markov_alpha_lively": (
        "mood-gated: the harness runs in the neutral mood, which reads alpha_calm"
    ),
    "markov_alpha_heated": (
        "mood-gated: the harness runs in the neutral mood, which reads alpha_calm"
    ),
}

# Child knob -> overrides that switch its parent mechanism on (design D3).
# Values are the ones the phase grids used, so the "parent on" arm is a
# configuration the project has already measured, not an invention.
GATED_BY: dict[str, dict[str, Any]] = {
    "markov_entropy_pivot": {"markov_entropy_temp_gain": 0.6},
    "markov_entropy_temp_min": {"markov_entropy_temp_gain": 0.6},
    "markov_entropy_temp_max": {"markov_entropy_temp_gain": 0.6},
    "markov_branching_candidate_floor": {"markov_branching_degenerate_max": 2.5},
    "markov_long_compression_beta": {
        "markov_alpha_calm": 0.5,
        "markov_long_compression": "pow",
    },
    "markov_short_half_life_days": {"markov_alpha_calm": 0.5},
    "markov_seed_branch_min": {"markov_seeded_candidate_ratio": 0.3},
    "markov_seed_branch_ideal": {"markov_seeded_candidate_ratio": 0.3},
    "markov_seed_branch_max": {"markov_seeded_candidate_ratio": 0.3},
    "markov_seed_min_support": {"markov_seeded_candidate_ratio": 0.3},
    "markov_seed_min_score": {"markov_seeded_candidate_ratio": 0.3},
    "markov_seed_min_token_len": {"markov_seeded_candidate_ratio": 0.3},
    "markov_seed_head_share": {"markov_seeded_candidate_ratio": 0.3},
    "hot_ngram_min_count": {"hot_ngram_slot_ratio": 0.4},
    "hot_ngram_recency_share": {"hot_ngram_slot_ratio": 0.4},
    "markov_hot_ngram_meme_ordering": {"hot_ngram_slot_ratio": 0.4},
}

# The frozen baseline convention: flavor and emoji silenced (matrix.yaml C0).
C0_OVERRIDES: dict[str, Any] = {"reply_flavor_strength": 0.0, "emoji_append_chance": 0.0}

CLASSIFICATION_METRICS = (
    "context_affinity_without_copy",
    "exact_context_copy_rate",
    "repetition_rate",
    "historical_meme_rate",
    "structural_window_escape",
    "structural_pool_ecb",
    "mean_response_length",
)

# How many times the default a knob with no upper bound is pushed to (D2).
OPEN_UPPER_FACTOR = 4


@dataclass(frozen=True, slots=True)
class Arm:
    arm_id: str
    knob: str
    label: str  # "min" / "max" / "flip"
    value: Any
    gated: bool  # parent enabled
    overrides: dict[str, Any]


def _read_sites() -> dict[str, list[str]]:
    """Where each registry knob is read, by module (static, whole app/)."""
    sites: dict[str, list[str]] = {spec.name: [] for spec in RUNTIME_FIELDS}
    skip = {"registry.py", "settings.py", "bot_messages.py"}
    for path in sorted((PROJECT_ROOT / "app").rglob("*.py")):
        if path.name in skip:
            continue
        text = path.read_text(encoding="utf-8")
        # runtime_state.py declares every knob, so a bare name there is not a
        # read; ``self.<name>`` is (mood config, the rare-event daily cap).
        prefix = r"self\." if path.name == "runtime_state.py" else r"\b"
        for name in sites:
            if re.search(rf"{prefix}{re.escape(name)}\b", text):
                sites[name].append(path.relative_to(PROJECT_ROOT).as_posix())
    return sites


def is_core_knob(name: str, sites: dict[str, list[str]]) -> bool:
    return any(
        site.endswith(f"app/core/{module}.py") for site in sites.get(name, ())
        for module in CORE_MODULES
    )


def extremes(spec: FieldSpec) -> list[tuple[str, Any]]:
    """The values a knob is pushed to: domain extremes, or the flipped boolean.

    Parsed from the registry's own range hint — the same string ``/help``
    prints — so the census sweeps exactly the domain the bot enforces. An
    extreme equal to the default is skipped (it would alias C0).
    """
    hint = spec.parse.hint
    default = spec.parse(spec.default)
    if hint == "true/false":
        return [("flip", not default)]
    match = re.fullmatch(r"(-?[\d.]+)\.\.(-?[\d.]+)", hint)
    if match:
        lo_s, hi_s = match.groups()
        lo = spec.parse(lo_s)
        hi = spec.parse(hi_s)
        return [(label, value) for label, value in (("min", lo), ("max", hi)) if value != default]
    match = re.fullmatch(r">= (-?[\d.]+)", hint)
    if match:
        lo = spec.parse(match.group(1))
        hi = spec.parse(str(default * OPEN_UPPER_FACTOR))
        return [(label, value) for label, value in (("min", lo), ("max", hi)) if value != default]
    if re.fullmatch(r"[\d/]+", hint):  # e.g. "2/3"
        values = [spec.parse(part) for part in hint.split("/")]
        return [("alt", value) for value in values if value != default]
    return []


def plan() -> tuple[list[Arm], dict[str, str], dict[str, list[str]]]:
    """Arms to run, knobs skipped with reasons, read sites of every knob."""
    sites = _read_sites()
    arms: list[Arm] = []
    skipped: dict[str, str] = {}
    for spec in RUNTIME_FIELDS:
        name = spec.name
        if not is_core_knob(name, sites):
            continue
        if name in NOT_SWEPT:
            skipped[name] = NOT_SWEPT[name]
            continue
        values = extremes(spec)
        if not values:
            skipped[name] = f"domain hint {spec.parse.hint!r} not sweepable"
            continue
        for label, value in values:
            base = dict(C0_OVERRIDES)
            base[name] = value
            arms.append(Arm(f"{name}__{label}", name, label, value, False, base))
            parent = GATED_BY.get(name)
            if parent:
                gated = dict(C0_OVERRIDES)
                gated.update(parent)
                gated[name] = value
                arms.append(Arm(f"{name}__{label}__gated", name, label, value, True, gated))
    return arms, skipped, sites


def _serialise_samples(
    values: dict[str, list[float] | None],
) -> dict[str, list[float | None] | None]:
    return {
        name: None if samples is None else [None if math.isnan(v) else v for v in samples]
        for name, samples in values.items()
    }


def _deserialise_samples(values: dict[str, Any]) -> dict[str, list[float] | None]:
    return {
        name: None if samples is None else [math.nan if v is None else float(v) for v in samples]
        for name, samples in values.items()
    }


async def run_arms(
    *,
    db_source: Path,
    arm_ids: list[str],
    context_mode: str,
    out: Path,
    generations: int,
    seeds: tuple[int, ...],
    chat_id: int | None,
) -> None:
    """Run C0 or a subset of arms and dump per-arm metric samples (numbers only)."""
    arms, _skipped, _sites = plan()
    by_id = {arm.arm_id: arm for arm in arms}
    prompt_set = load_prompts(PROMPTS_PATH)
    results: dict[str, Any] = {
        "context_mode": context_mode,
        "prompt_version": prompt_set.version,
        "generations": generations,
        "seeds": list(seeds),
        "arms": {},
    }
    thresholds = load_thresholds()
    edge_overlap = float(
        thresholds.get("structural_escape", {}).get("edge_overlap_similar", 0.5)
    )
    for arm_id in arm_ids:
        if arm_id == "C0":
            configs = {"C0": MatrixConfig("C0", "baseline", True, dict(C0_OVERRIDES))}
        else:
            arm = by_id[arm_id]
            configs = {
                "C0": MatrixConfig("C0", "baseline", False, dict(C0_OVERRIDES)),
                arm_id: MatrixConfig(arm_id, arm.knob, True, dict(arm.overrides)),
            }
        try:
            runs, _ = await run_matrix(
                db_source=db_source,
                chat_id=chat_id,
                configs=configs,
                prompt_set=prompt_set,
                seeds=seeds,
                generations=generations,
                context_mode=context_mode,
                edge_overlap_similar=edge_overlap,
            )
        except Exception as error:  # an arm that crashes is a finding, not a stop
            results["arms"][arm_id] = {"error": f"{type(error).__name__}: {error}"}
            _dump(out, results)
            continue
        run = runs[arm_id]
        results["arms"][arm_id] = {
            "n_records": len(run.records),
            "success_rate": (
                sum(1 for r in run.records if r.success) / len(run.records)
                if run.records
                else None
            ),
            "latency_p95": latency_percentiles(run.records)["latency_p95"],
            "metrics": _serialise_samples(metric_values(run.records)),
        }
        _dump(out, results)


def _dump(out: Path, results: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, sort_keys=True), encoding="utf-8")


def launch(
    *,
    db: str,
    out_dir: Path,
    workers: int,
    generations: int,
    modes: tuple[str, ...],
) -> list[Path]:
    """Spawn detached worker processes: C0 per mode plus the arms in chunks.

    Every worker writes its own JSON and a ``.done`` marker; the caller polls
    the markers. Detached so a supervising shell's timeout cannot kill the run.
    """
    arms, _skipped, _sites = plan()
    ids = [arm.arm_id for arm in arms]
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, list[str], str]] = []
    for mode in modes:
        jobs.append((f"C0_{mode}", ["C0"], mode))
    per_worker = max(1, math.ceil(len(ids) * len(modes) / max(1, workers - len(modes))))
    for mode in modes:
        for index in range(0, len(ids), per_worker):
            chunk = ids[index : index + per_worker]
            jobs.append((f"arms_{mode}_{index // per_worker:02d}", chunk, mode))
    markers: list[Path] = []
    flags = 0
    if os.name == "nt":
        # No console window per worker (the first launch opened one per
        # process) and a new process group, so a supervising shell's timeout
        # or Ctrl-C does not reach the workers.
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    for job_id, chunk, mode in jobs:
        out = out_dir / f"{job_id}.json"
        marker = out_dir / f"{job_id}.done"
        log = out_dir / f"{job_id}.log"
        argv = [
            sys.executable, "-m", "tools.eval.knob_census", "run",
            "--db", db, "--context-mode", mode, "--arms", ",".join(chunk),
            "--out", str(out), "--generations", str(generations),
            "--done", str(marker),
        ]
        with log.open("w", encoding="utf-8") as handle:
            subprocess.Popen(
                argv,
                cwd=PROJECT_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                start_new_session=os.name != "nt",
            )
        markers.append(marker)
    return markers


# ---------------------------------------------------------------- report


def classify_extreme(
    base: dict[str, list[float] | None],
    arm: dict[str, list[float] | None],
    thresholds: dict[str, Any],
) -> tuple[str, dict[str, tuple[float, float, float, bool] | None]]:
    """Class of one extreme against C0 plus the per-metric deltas."""
    tolerance = thresholds["tolerance"]
    strong = thresholds["strong"]
    deltas: dict[str, tuple[float, float, float, bool] | None] = {}
    inert = True
    is_strong = False
    for metric in CLASSIFICATION_METRICS:
        result = delta_ci(base.get(metric), arm.get(metric))
        deltas[metric] = result
        if result is None:
            inert = False  # not resolved is not inert
            continue
        point, lo, hi, significant = result
        tol = float(tolerance[metric])
        if lo < -tol or hi > tol:
            inert = False
        if significant and abs(point) >= float(strong[metric]):
            is_strong = True
    if is_strong:
        return "strong", deltas
    if inert:
        return "inert", deltas
    return "weak", deltas


RANK = {"inert": 0, "weak": 1, "strong": 2}


def classify_knob(per_extreme: list[tuple[bool, str]]) -> str:
    """Knob class from its extremes' classes; ``per_extreme`` = (gated, class)."""
    plain = [cls for gated, cls in per_extreme if not gated]
    gated = [cls for gated_flag, cls in per_extreme if gated_flag]
    plain_class = max(plain, key=RANK.get) if plain else "weak"
    if gated and plain_class == "inert":
        gated_class = max(gated, key=RANK.get)
        if gated_class != "inert":
            return "gated"
    return plain_class


PROPOSAL = {
    "dead": "remove: nothing reads it",
    "inert": "remove or reduce to a constant: extremes move nothing measurable",
    "gated": "decide together with the parent knob; alone it is a no-op",
    "weak": "candidate to merge or narrow: effect below the strength bar",
    "strong": "keep; check the domain ceiling (an extreme may break form)",
}


def _fmt_delta(result: tuple[float, float, float, bool] | None) -> str:
    if result is None:
        return "—"
    point, lo, hi, significant = result
    return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]{'*' if significant else ''}"


def build_report(inputs: Path, thresholds: dict[str, Any], date: str) -> tuple[str, dict[str, Any]]:
    arms, skipped, sites = plan()
    baselines: dict[str, dict[str, list[float] | None]] = {}
    baseline_meta: dict[str, Any] = {}
    measured: dict[str, dict[str, Any]] = {}
    for path in sorted(inputs.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = payload["context_mode"]
        for arm_id, entry in payload["arms"].items():
            if arm_id == "C0":
                baselines[mode] = _deserialise_samples(entry["metrics"])
                baseline_meta[mode] = {
                    "prompt_version": payload["prompt_version"],
                    "latency_p95": entry["latency_p95"],
                    "n_records": entry["n_records"],
                }
            else:
                measured.setdefault(arm_id, {})[mode] = entry
    lines: list[str] = []
    lines.append(f"# Перепись ручек (M3R-151) — {date}")
    lines.append("")
    lines.append(
        "Change `knob-census`. Правило классов — `eval_thresholds.yaml` → "
        "`knob_census`, зарегистрировано до прогона. Каждая ручка — на "
        "экстремумах домена (булева — инверсия) против C0, парные дельты, оба "
        "режима; ручки с родителем — ещё и при включённом родителе. "
        "Метрики классификации: " + ", ".join(f"`{m}`" for m in CLASSIFICATION_METRICS) + "."
    )
    lines.append("")
    for mode, meta in sorted(baseline_meta.items()):
        lines.append(
            f"- C0 `{mode}`: {meta['n_records']} записей, версия промптов "
            f"`{meta['prompt_version']}`, p95 {meta['latency_p95']:.1f} мс"
        )
    lines.append("")
    lines.append(
        "Латентность в таблице справочная: армы считались параллельно и между "
        "собой по ней не сравнимы. Классы: dead — не читается; gated — двигает "
        "только при включённом родителе; inert — интервалы всех дельт внутри "
        "полосы допуска на всех экстремумах; strong — значимая дельта не ниже "
        "планки силы; weak — остальное."
    )
    lines.append("")

    summary: dict[str, Any] = {"knobs": {}, "skipped": skipped, "thresholds": thresholds}
    knob_rows: list[tuple[str, str, str]] = []
    detail_lines: list[str] = []
    order = [spec.name for spec in RUNTIME_FIELDS]
    knobs_in_plan = sorted({arm.knob for arm in arms}, key=order.index)
    for knob in knobs_in_plan:
        per_extreme: list[tuple[bool, str]] = []
        rows: list[str] = []
        errors: list[str] = []
        for arm in [a for a in arms if a.knob == knob]:
            for mode in sorted(baselines):
                entry = measured.get(arm.arm_id, {}).get(mode)
                label = f"{arm.label}{' (parent on)' if arm.gated else ''}"
                blank = " |" * len(CLASSIFICATION_METRICS)
                if entry is None:
                    rows.append(f"| {label} | {mode} | not run |{blank}")
                    continue
                if "error" in entry:
                    errors.append(f"{arm.arm_id}/{mode}: {entry['error']}")
                    rows.append(f"| {label} | {mode} | error |{blank}")
                    continue
                cls, deltas = classify_extreme(
                    baselines[mode], _deserialise_samples(entry["metrics"]), thresholds
                )
                per_extreme.append((arm.gated, cls))
                cells = " | ".join(_fmt_delta(deltas[m]) for m in CLASSIFICATION_METRICS)
                rows.append(
                    f"| {arm.label}={arm.value}{' (parent on)' if arm.gated else ''} "
                    f"| {mode} | {cls} | {cells} |"
                )
        knob_class = classify_knob(per_extreme) if per_extreme else "not run"
        if errors and not per_extreme:
            knob_class = "error"
        summary["knobs"][knob] = {"class": knob_class, "errors": errors}
        knob_rows.append((knob, knob_class, PROPOSAL.get(knob_class, "see detail")))
        detail_lines.append(f"### `{knob}` — **{knob_class}**")
        detail_lines.append("")
        detail_lines.append(
            "| extreme | mode | class | " + " | ".join(CLASSIFICATION_METRICS) + " |"
        )
        detail_lines.append("|---|---|---|" + "---|" * len(CLASSIFICATION_METRICS))
        detail_lines.extend(rows)
        if errors:
            detail_lines.append("")
            detail_lines.append("Ошибки прогона: " + "; ".join(errors))
        detail_lines.append("")

    lines.append("## Сводка по ручкам")
    lines.append("")
    lines.append("| ручка | класс | предложение |")
    lines.append("|---|---|---|")
    for knob, cls, proposal in knob_rows:
        lines.append(f"| `{knob}` | **{cls}** | {proposal} |")
    lines.append("")
    counts: dict[str, int] = {}
    for _, cls, _ in knob_rows:
        counts[cls] = counts.get(cls, 0) + 1
    lines.append("Итого: " + ", ".join(f"{cls} {n}" for cls, n in sorted(counts.items())) + ".")
    lines.append("")

    lines.append("## Не свипуются (читаются ядром)")
    lines.append("")
    for knob, reason in skipped.items():
        lines.append(f"- `{knob}` — {reason}")
    lines.append("")

    lines.append("## Вне оффлайн-замера (не читаются ядром генерации)")
    lines.append("")
    lines.append("| ручка | где читается | статический класс |")
    lines.append("|---|---|---|")
    outside: dict[str, Any] = {}
    for spec in RUNTIME_FIELDS:
        if is_core_knob(spec.name, sites):
            continue
        where = sites.get(spec.name, [])
        cls = "dead" if not where else "outside"
        outside[spec.name] = {"class": cls, "sites": where}
        lines.append(f"| `{spec.name}` | {', '.join(where) if where else '—'} | {cls} |")
    summary["outside"] = outside
    lines.append("")
    lines.append("## Разбор по ручкам")
    lines.append("")
    lines.extend(detail_lines)
    return "\n".join(lines), summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tools.eval.knob_census", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    run_p = sub.add_parser("run")
    run_p.add_argument("--db", required=True)
    run_p.add_argument("--chat-id", type=int, default=None)
    run_p.add_argument("--context-mode", choices=CONTEXT_MODES, default="ctx")
    run_p.add_argument("--arms", required=True, help="comma-separated arm ids, or C0")
    run_p.add_argument("--out", required=True)
    run_p.add_argument("--generations", type=int, default=500)
    run_p.add_argument("--seeds", type=str, default=None)
    run_p.add_argument("--done", type=str, default=None, help="marker file written on completion")
    launch_p = sub.add_parser("launch")
    launch_p.add_argument("--db", required=True)
    launch_p.add_argument("--out", required=True)
    launch_p.add_argument("--workers", type=int, default=8)
    launch_p.add_argument("--generations", type=int, default=500)
    report_p = sub.add_parser("report")
    report_p.add_argument("--inputs", required=True)
    report_p.add_argument("--out", required=True)
    report_p.add_argument("--json-out", default=None)
    report_p.add_argument("--date", default=None)
    args = parser.parse_args()

    if args.command == "plan":
        arms, skipped, sites = plan()
        for arm in arms:
            print(f"{arm.arm_id:55s} {arm.overrides}")
        print(f"\n{len(arms)} arms, {len({a.knob for a in arms})} knobs")
        for knob, reason in skipped.items():
            print(f"skip {knob}: {reason}")
        return
    if args.command == "run":
        seeds = (
            tuple(int(s) for s in args.seeds.split(",")) if args.seeds else PROTOCOL_SEEDS
        )
        asyncio.run(
            run_arms(
                db_source=Path(args.db),
                arm_ids=[a for a in args.arms.split(",") if a],
                context_mode=args.context_mode,
                out=Path(args.out),
                generations=args.generations,
                seeds=seeds,
                chat_id=args.chat_id,
            )
        )
        if args.done:
            Path(args.done).write_text("done", encoding="utf-8")
        return
    if args.command == "launch":
        markers = launch(
            db=args.db,
            out_dir=Path(args.out),
            workers=args.workers,
            generations=args.generations,
            modes=CONTEXT_MODES,
        )
        print(f"launched {len(markers)} workers; markers: {markers[0].parent}")
        return
    if args.command == "report":
        import datetime as _dt

        thresholds = load_thresholds()["knob_census"]
        date = args.date or _dt.date.today().isoformat()
        text, summary = build_report(Path(args.inputs), thresholds, date)
        Path(args.out).write_text(text, encoding="utf-8")
        if args.json_out:
            payload = {k: v for k, v in summary.items() if k != "thresholds"}
            Path(args.json_out).write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        print(f"report: {args.out}")


if __name__ == "__main__":
    main()
