"""Per-knob sweep on the prod DB copy, 2026-07-20.

Baselines: registry defaults and the live config (context+chaos package
enabled 2026-07-16: boost=2.0, extension_share=0.8, order_mix=0.5).
All sweep points are deltas on top of the LIVE baseline, since that is the
pipeline actually running in prod.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval_prod import evaluate  # noqa: E402

DB = PROJECT_ROOT / "db_prod_copy" / "markov.db"
OUT = Path(__file__).with_name("sweep_2026-07-20_results.jsonl")
SEEDS = [101, 202, 303]
GENERATIONS = 200

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
    ("baseline_registry", {}),
    ("baseline_live", dict(LIVE)),
    # --- new knobs (post-2026-07-15 merges) ---
    ("slot_mutation=0.15", live_with(slot_mutation_probability=0.15)),
    ("slot_mutation=0.3", live_with(slot_mutation_probability=0.3)),
    ("slot_mutation=0.6", live_with(slot_mutation_probability=0.6)),
    ("intonation=0.5", live_with(intonation_profile_strength=0.5)),
    ("intonation=1.0", live_with(intonation_profile_strength=1.0)),
    ("jump_boost=1.0", live_with(context_jump_boost=1.0)),
    ("jump_boost=4.0", live_with(context_jump_boost=4.0)),
    ("ext_share=0", live_with(verbatim_extension_share=0.0)),
    ("ext_share=0.6", live_with(verbatim_extension_share=0.6)),
    ("ext_share=1.0", live_with(verbatim_extension_share=1.0)),
    ("order_mix=0", live_with(order_mix_probability=0.0)),
    ("order_mix=0.25", live_with(order_mix_probability=0.25)),
    ("order_mix=0.75", live_with(order_mix_probability=0.75)),
    ("splice=0", live_with(context_anchor_splice_probability=0.0)),
    ("splice=0.5", live_with(context_anchor_splice_probability=0.5)),
    # --- dead-knob recheck at extremes ---
    ("sel_temp=0", live_with(candidate_selection_temperature=0.0)),
    ("sel_temp=3", live_with(candidate_selection_temperature=3.0)),
    ("randomness=0", live_with(randomness_strength=0.0)),
    ("randomness=3", live_with(randomness_strength=3.0)),
    ("rep_pen=0", live_with(repetition_penalty_strength=0.0)),
    ("rep_pen=3", live_with(repetition_penalty_strength=3.0)),
    ("recent_pen=0", live_with(recent_reply_penalty_strength=0.0)),
    ("recent_pen=3", live_with(recent_reply_penalty_strength=3.0)),
    ("casefold=off", live_with(fuzzy_context_casefold=False)),
    ("ctx_bias=1.0", live_with(reply_context_bias=1.0)),
    ("ctx_bias=4.0", live_with(reply_context_bias=4.0)),
    # --- strong-knob confirmation in the context+chaos regime ---
    ("verb_pen=0", live_with(verbatim_penalty_strength=0.0)),
    ("verb_pen=3", live_with(verbatim_penalty_strength=3.0)),
    ("start_bias=1.0", live_with(reply_context_start_bias=1.0)),
    ("start_bias=4.0", live_with(reply_context_start_bias=4.0)),
    ("jump_prob=0", live_with(markov_jump_probability=0.0)),
    ("jump_prob=0.3", live_with(markov_jump_probability=0.3)),
    ("affinity=1.0", live_with(context_start_affinity=1.0)),
    ("affinity=10.0", live_with(context_start_affinity=10.0)),
    ("len_adapt=0", live_with(length_context_adaptation=0.0)),
    ("len_adapt=3", live_with(length_context_adaptation=3.0)),
    ("markov_order=2", live_with(markov_order=2)),
]

# keys dropped from each stored report to keep the JSONL small
DROP = {
    "slot_mutation_samples", "slot_mutation_winner_samples",
    "db_source",
}


async def main() -> None:
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done.add((row["config"], row["seed"]))
            except Exception:
                pass
    total = len(CONFIGS) * len(SEEDS)
    i = 0
    with OUT.open("a", encoding="utf-8") as fh:
        for name, overrides in CONFIGS:
            for seed in SEEDS:
                i += 1
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
                    f"[{i}/{total}] {name} seed={seed} "
                    f"({time.time() - started:.0f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    asyncio.run(main())
