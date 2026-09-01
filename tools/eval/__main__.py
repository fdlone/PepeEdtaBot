"""CLI for the Markov 2.0R eval protocol (doc 05).

Usage:
    python -m tools.eval --db db_prod_copy/markov.db --label baseline
    python -m tools.eval --smoke
    python -m tools.eval --generate-prompts --db <snapshot> --label <snapshot-id>

``--smoke`` is the CI mode (doc 05 §7): C0 + CF on a freshly built synthetic
snapshot, 40 generations, 1 seed; fails the process on runner errors or
invariant violations (never on metric thresholds).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval.config import (  # noqa: E402
    MATRIX_PATH,
    THRESHOLDS_PATH,
    load_matrix,
    load_thresholds,
)
from tools.eval.metrics import (  # noqa: E402
    DEFAULT_EDGE_OVERLAP_SIMILAR,
    metric_values,
)
from tools.eval.prompts import (  # noqa: E402
    PROMPTS_PATH,
    generate_prompts,
    load_prompts,
    save_prompts,
)
from tools.eval.report import build_report, metrics_summary  # noqa: E402
from tools.eval.run import (  # noqa: E402
    CONTEXT_MODES,
    DEFAULT_GENERATIONS,
    PROTOCOL_SEEDS,
    SMOKE_GENERATIONS,
    read_df_corpus_facts,
    run_matrix,
)
from tools.eval.synthetic import SYNTHETIC_CHAT_ID, build_synthetic_snapshot  # noqa: E402
from tools.eval.temporal_fixture import (  # noqa: E402
    DEFAULT_FRESH_DAYS,
    build_temporal_fixture,
)
from tools.eval_prod import pick_chat_id  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "docs" / "eval_reports"


def _revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _parse_seeds(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return PROTOCOL_SEEDS
    return tuple(int(part) for part in raw.split(",") if part.strip())


async def _smoke() -> int:
    db_path, temp_dir = await build_synthetic_snapshot()
    try:
        prompt_set = generate_prompts(
            db_path,
            chat_id=SYNTHETIC_CHAT_ID,
            seed=PROTOCOL_SEEDS[0],
            snapshot_label="snapshot_synthetic",
            per_category=30,
        )
        configs = load_matrix()
        runs, _skipped = await run_matrix(
            db_source=db_path,
            chat_id=SYNTHETIC_CHAT_ID,
            configs={c: configs[c] for c in ("C0", "CF")},
            prompt_set=prompt_set,
            seeds=(PROTOCOL_SEEDS[0],),
            generations=SMOKE_GENERATIONS,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    # Invariant checks only (spec: smoke never gates on metric thresholds).
    failures: list[str] = []
    for config_id, run in runs.items():
        values = metric_values(run.records)
        success = values["generation_success_rate"]
        if not success or sum(success) == 0:
            failures.append(f"{config_id}: empty-output collapse (no successful generations)")
        for name, samples in values.items():
            # `nan` здесь — метка «этой записи в метрике нет», а не сбой:
            # метрики с фильтром выровнены по списку записей, чтобы парная
            # дельта могла сшивать наблюдения одного промпта (A2-4). Проверка
            # `not (sample == sample)` — это проверка на NaN, и до перехода на
            # парный бутстрап она была верна; после него `historical_meme_rate`
            # даёт метки всегда (категория `meme-bait` — четверть промптов), и
            # smoke падал на каждом прогоне.
            #
            # Инвариант при этом не ослаблен, а уточнён: `inf` по-прежнему
            # сбой при любых обстоятельствах, а метрика, состоящая из одних
            # меток, — тоже сбой: значит её не удалось посчитать ни разу, и
            # это ровно то «схлопывание», которое smoke обязан ловить.
            finite = [
                sample
                for sample in samples or ()
                if sample == sample and sample not in (float("inf"), float("-inf"))
            ]
            if any(
                sample in (float("inf"), float("-inf")) for sample in samples or ()
            ):
                failures.append(f"{config_id}: {name} produced inf")
            elif samples and not finite:
                failures.append(f"{config_id}: {name} has no computable value")
    if failures:
        print("SMOKE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    summary = metrics_summary(runs)
    print(json.dumps({"smoke": "ok", "summary": summary}, indent=2, sort_keys=True))
    return 0


async def _protocol(args: argparse.Namespace) -> int:
    seeds = _parse_seeds(args.seeds)
    configs = load_matrix(Path(args.matrix))
    thresholds = load_thresholds(Path(args.thresholds))
    prompt_set = load_prompts(Path(args.prompts))
    db_source = Path(args.db)
    # gate-phase5-ndocs-floor (design D1): the corpus precondition is measured
    # on the ORIGINAL snapshot, before the temporal fixture can replace
    # db_source and before any working copy window-populates df. Reading later
    # would see the retention window the runner itself writes.
    df_facts = read_df_corpus_facts(db_source, args.chat_id)
    # Phase 4 (doc 05 §5): the manual rating lives OUTSIDE the repository — it
    # is a list of verbatim chat phrases. Only its aggregate reaches the report,
    # and without it the Phase 4 gate reports insufficient data rather than
    # passing on the automatic half alone.
    manual_rating: dict[str, Any] | None = None
    if args.manual_rating:
        manual_rating = json.loads(
            Path(args.manual_rating).read_text(encoding="utf-8")
        )
    fresh_tokens: frozenset[str] | None = None
    evaluation_moment: int | None = None
    notes = [
        "IDF for affinity metrics is computed over the snapshot's retained "
        "message window (full history is not stored) — window-relative, "
        "identical across configurations (audit §3).",
    ]
    fixture_dir: tempfile.TemporaryDirectory[str] | None = None

    if args.temporal_fixture:
        # Phase 3 (M2R-210): the source snapshot has no observation times, so a
        # temporal one is reconstructed from its retained messages. The runs
        # below then measure freshness on a corpus that actually has a fresh
        # slice — see temporal_fixture.py for what this is and is not.
        fixture_dir = tempfile.TemporaryDirectory(prefix="pepe_eval_temporal_")
        target = Path(fixture_dir.name) / "temporal.db"
        chat_id = pick_chat_id(db_source, args.chat_id)
        info = await build_temporal_fixture(
            source=db_source,
            target=target,
            chat_id=chat_id,
            fresh_days=args.fresh_days,
        )
        db_source = target
        args.chat_id = chat_id
        fresh_tokens = info.fresh_tokens
        # The corpus's own last moment, not wall-clock now: the snapshot is
        # weeks old, and at a 3-day half-life "now" would read every short
        # counter as decayed to nothing.
        evaluation_moment = info.last_moment
        notes.append(info.describe())

    try:
        runs, skipped = await run_matrix(
            db_source=db_source,
            chat_id=args.chat_id,
            configs=configs,
            prompt_set=prompt_set,
            seeds=seeds,
            generations=args.generations,
            fresh_tokens=fresh_tokens,
            evaluation_moment=evaluation_moment,
            context_mode=args.context_mode,
            # M3R-011: the similarity threshold is a pre-registered gate input,
            # so it comes from the thresholds file — never from a default baked
            # into the metric module, which would let a run and its gate
            # disagree about what "substantially different" means.
            edge_overlap_similar=float(
                thresholds.get("structural_escape", {}).get(
                    "edge_overlap_similar", DEFAULT_EDGE_OVERLAP_SIMILAR
                )
            ),
        )
    finally:
        if fixture_dir is not None:
            fixture_dir.cleanup()
    date = _dt.date.today().isoformat()
    report = build_report(
        runs=runs,
        skipped=skipped,
        prompt_set=prompt_set,
        thresholds=thresholds,
        manual_rating=manual_rating,
        snapshot_label=args.label,
        seeds=seeds,
        generations=args.generations,
        revision=_revision(),
        date=date,
        notes=notes,
        context_mode=args.context_mode,
        df_facts=df_facts,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # The mode is part of the file name, not only of the header: two runs of the
    # same label differing only by mode are two different measurements, and the
    # second must not overwrite the first (M3R-140).
    out_path = REPORTS_DIR / f"eval_{date}_{args.label}_{args.context_mode}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"report: {out_path}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                metrics_summary(runs, args.context_mode), indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )
        print(f"metrics summary: {args.json_out}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tools.eval", description=__doc__)
    parser.add_argument("--db", type=str, help="snapshot DB path (protocol/prompt modes)")
    parser.add_argument("--chat-id", type=int, default=None)
    parser.add_argument("--matrix", type=str, default=str(MATRIX_PATH))
    parser.add_argument("--thresholds", type=str, default=str(THRESHOLDS_PATH))
    parser.add_argument("--prompts", type=str, default=str(PROMPTS_PATH))
    parser.add_argument(
        "--seeds", type=str, default=None, help="comma-separated; default 42,1337,2026"
    )
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--label", type=str, default="run", help="snapshot label for the report")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="CI smoke mode (doc 05 §7)")
    parser.add_argument(
        "--generate-prompts",
        action="store_true",
        help="build eval_prompts.yaml from the snapshot and exit",
    )
    parser.add_argument("--prompt-seed", type=int, default=PROTOCOL_SEEDS[0])
    parser.add_argument(
        "--manual-rating",
        type=str,
        default=None,
        help=(
            "path to the local manual top-meme rating (doc 05 §5). Keep it "
            "out of the repository: it contains verbatim chat phrases"
        ),
    )
    parser.add_argument(
        "--temporal-fixture",
        action="store_true",
        help=(
            "rebuild a temporal snapshot from the source's retained messages "
            "and run on it (Phase 3; freshness metrics need observation times)"
        ),
    )
    parser.add_argument(
        "--fresh-days",
        type=float,
        default=DEFAULT_FRESH_DAYS,
        help="window that defines the fixture's fresh slice (default 14)",
    )
    parser.add_argument(
        "--context-mode",
        choices=CONTEXT_MODES,
        default="ctx",
        help=(
            "M3R-140: 'ctx' supplies the prompt as context (the historical "
            "default); 'noctx' supplies none. Fixed per run — a report never "
            "mixes modes into one number, and a gate declared two-mode reports "
            "insufficient data until both runs exist"
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        raise SystemExit(asyncio.run(_smoke()))
    if args.generate_prompts:
        if not args.db:
            parser.error("--generate-prompts requires --db")
        from tools.eval_prod import pick_chat_id

        chat = pick_chat_id(Path(args.db), args.chat_id)
        # M3R-130: the meme support floor is a gate input, so it comes from the
        # thresholds file — a set built with a different floor is a different
        # set, and its version says so.
        meme_config = load_thresholds().get("meme_regression", {})
        prompt_set = generate_prompts(
            Path(args.db),
            chat_id=chat,
            seed=args.prompt_seed,
            snapshot_label=args.label,
            meme_support_min=int(meme_config.get("support_min", 1)),
        )
        save_prompts(prompt_set)
        print(
            f"prompts: {PROMPTS_PATH} (version {prompt_set.version}, "
            f"{len(prompt_set.memes)} memes at support floor "
            f"{meme_config.get('support_min', 1)})"
        )
        raise SystemExit(0)
    if not args.db:
        parser.error("--db is required (or use --smoke)")
    raise SystemExit(asyncio.run(_protocol(args)))


if __name__ == "__main__":
    main()
