"""High-power follow-up sweep on the prod DB copy, 2026-07-20 (pass B).

The first pass (3 seeds x 200 generations) could not separate small effects
from sampling noise: the observed per-seed spread of ``context_anchored_win_rate``
(sd 0.040) is essentially the binomial limit for n=200, and paired-by-seed
differences are just as noisy, so the noise lives in generation sampling
rather than in which contexts a seed picks. The only lever is total
generations per arm.

This pass runs 10 seeds x 400 generations = 4000 generations per arm, giving
SE ~0.008 per arm and SE ~0.011 on a difference against the live baseline --
enough to call effects of 0.03+ and to declare smaller ones null.

Arms are pruned to the questions that are actually open (see docs/OPEN.md O4);
knobs already decisive in pass A (start_bias=1.0, splice extremes, verb_pen=0,
markov_order=2) and exact no-ops (recent_reply_penalty_strength,
candidate_selection_temperature=3) are not re-run.

Runs sharded across processes: ``--shard i/n``. Work is split round-robin over
the (config, seed) task list so every arm progresses in parallel, and each
shard appends to its own JSONL to keep concurrent writes safe.

WARNING -- ``avg_generation_latency_ms`` from this pass is NOT comparable to
pass A: with several shards competing for cores, wall-clock per generation is
inflated by contention. Behavioural metrics are unaffected (generation is
deterministic given the seed; timing does not change which tokens are picked).
Use pass A, or a single-shard run, for any latency question.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval_prod import evaluate  # noqa: E402

DB = PROJECT_ROOT / "db_prod_copy" / "markov.db"
OUT_GLOB = "sweep_2026-07-20b_results*.jsonl"
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
GENERATIONS = 400

LIVE = {
    "context_jump_boost": 2.0,
    "verbatim_extension_share": 0.8,
    "order_mix_probability": 0.5,
}


def live_with(**kw):
    d = dict(LIVE)
    d.update(kw)
    return d


CONFIGS: list[tuple[str, dict]] = [
    ("baseline_live", dict(LIVE)),
    # --- O4(1): is fuzzy_context_casefold a real effect or noise? ---
    ("casefold=off", live_with(fuzzy_context_casefold=False)),
    # --- O4(3): jump_boost / ext_share were non-monotonic on 3 seeds ---
    ("jump_boost=1.0", live_with(context_jump_boost=1.0)),
    ("jump_boost=4.0", live_with(context_jump_boost=4.0)),
    ("ext_share=0", live_with(verbatim_extension_share=0.0)),
    ("ext_share=0.6", live_with(verbatim_extension_share=0.6)),
    ("ext_share=1.0", live_with(verbatim_extension_share=1.0)),
    # --- monotonicity claim for order_mix ---
    ("order_mix=0", live_with(order_mix_probability=0.0)),
    ("order_mix=0.25", live_with(order_mix_probability=0.25)),
    ("order_mix=0.75", live_with(order_mix_probability=0.75)),
    # --- knobs with a pending deploy decision (after the O1 window) ---
    ("slot_mutation=0.15", live_with(slot_mutation_probability=0.15)),
    ("slot_mutation=0.3", live_with(slot_mutation_probability=0.3)),
    ("intonation=1.0", live_with(intonation_profile_strength=1.0)),
    # --- O4(4): combined rise of several knobs, "connective tick" risk ---
    (
        "combo_mild",
        live_with(
            context_anchor_splice_probability=0.4,
            markov_jump_probability=0.2,
            order_mix_probability=0.75,
        ),
    ),
    (
        "combo_hot",
        live_with(
            context_anchor_splice_probability=0.5,
            markov_jump_probability=0.3,
            reply_context_start_bias=4.0,
            order_mix_probability=0.75,
        ),
    ),
]

# keys dropped from each stored report to keep the JSONL small
DROP = {
    "slot_mutation_samples", "slot_mutation_winner_samples",
    "db_source",
}


def already_done() -> set[tuple[str, int]]:
    """(config, seed) pairs present in ANY shard file, so restarts resume."""
    done: set[tuple[str, int]] = set()
    for path in Path(__file__).parent.glob(OUT_GLOB):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue  # a torn final line from a killed run
            done.add((row["config"], row["seed"]))
    return done


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n, round-robin over tasks")
    args = ap.parse_args()
    index, count = (int(x) for x in args.shard.split("/"))

    tasks = [(n, o, s) for n, o in CONFIGS for s in SEEDS]
    mine = [t for k, t in enumerate(tasks) if k % count == index]
    done = already_done()

    out = Path(__file__).with_name(
        f"sweep_2026-07-20b_results_s{index}.jsonl" if count > 1
        else "sweep_2026-07-20b_results.jsonl"
    )
    started_all = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, (name, overrides, seed) in enumerate(mine, 1):
            if (name, seed) in done:
                continue
            started = time.time()
            report = await evaluate(
                db_source=DB,
                chat_id=None,
                seed=seed,
                generations=GENERATIONS,
                overrides=overrides,
            )
            for key in DROP:
                report.pop(key, None)
            report["config"] = name
            fh.write(json.dumps(report, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"[shard {index} {i}/{len(mine)}] {name} seed={seed} "
                f"({time.time() - started:.0f}s, "
                f"elapsed {(time.time() - started_all) / 60:.0f}m)",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
