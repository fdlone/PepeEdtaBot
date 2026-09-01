# tools/eval — Markov 2.0R evaluation protocol runner

Normative source: `docs/v2/05_MARKOV_2_0R_EVAL_PROTOCOL.md`. This package is the
protocol implementation; the knob-sweep harness `tools/eval_prod.py` and the
byte-identity guard `tools/generation_hash.py` stay independent.

## Usage

```bash
# Full protocol run (500 generations x seeds 42,1337,2026 per configuration):
python -m tools.eval --db db_prod_copy/markov.db --label baseline-C0-2026-07-13

# CI smoke (doc 05 §7): C0+CF, synthetic snapshot, 40 generations, 1 seed:
python -m tools.eval --smoke

# Regenerate the prompt set (a deliberate act — produces a new version):
python -m tools.eval --generate-prompts --db <snapshot> --label <snapshot-id>
```

Reports land in `docs/eval_reports/`; `--json-out` additionally writes the
deterministic metric summary used by reproducibility checks.

## Snapshots (doc 05 §1.1)

- **snapshot_full** — an anonymized copy of the production DB. Preparation:
  copy the DB file together with its `-wal`/`-shm` sidecars (the runner does
  this itself via `copy_database` and applies pending migrations to the copy,
  never to the source). The snapshot id is `<date of the dump>`; record it in
  `--label`. Snapshots never leave the developer's machine and are never
  committed (`db_prod_copy/` is gitignored). The frozen Phase 0 baseline:
  `db_prod_copy/markov.db`, dated **2026-07-13** (owner decision 2026-08-11:
  a fresher dump is not available — audit §10.2).
- **snapshot_temporal** — does not exist until Phase 3 adds timestamps and a
  real accumulation window (audit §10.1). Temporal metrics report
  `insufficient data` until then.
- **snapshot_synthetic** — built on demand, deterministically, by
  `tools/eval/synthetic.py` (seeded wrapper around `tools/seed_diverse`);
  never committed.

## Files

- `matrix.yaml` — ablation matrix C0–CF (doc 05 §2). C0 is frozen; phases add
  their configurations by editing this file.
- `eval_prompts.yaml` — versioned prompt set (doc 05 §1.2), generated from the
  snapshot, four categories ≥30 prompts each, plus the meme list for the
  §3.5 gate. The `version` field is a content hash; `load_prompts` rejects
  hand-edited files.
- `eval_thresholds.yaml` — pre-registered gate thresholds (doc 05 §4). Editing
  after fixation requires a dedicated commit with written justification.

## Per-route breakdown (M3R-103)

Every report carries a `configuration x route` table after the per-category
breakdown: for each member of `CandidateRoute` — attempts, pool share,
presence, win rate **given presence**, the affinity-without-copy and copy rate
of the replies the route won, mean latency of generations with / without the
route in the pool (an upper bound on its cost, not a step measurement), and
pre-pool rejections summed by failure class (M3R-021). Routes are attributed
by the generator at candidate creation (`GenRecord.winner_route`,
`GenRecord.pool_routes`), never inferred from text; a route whose mechanism
did not run reads `not attempted`. The bit-for-bit `metrics_summary` does not
include route fields.

## L1 hot-n-gram seeding (M3R-145)

In the `noctx` mode the runner reproduces the pipeline's L1 seed draw
(`ReplyPipeline._hot_ngram_seed`): roll `hot_ngram_seed_chance`, then pick one
of the configuration's hot n-grams (`hot_ngram_min_count`,
`hot_ngram_recency_share`) with a deterministic RNG separate from the
generation RNG (`HOT_SEED_RNG_OFFSET`), so a configuration whose hot selection
is empty stays byte-identical to a run without the draw. `ctx` never seeds:
the pipeline never seeds addressed replies. Records carry `seed_drawn` and
the winner's `start_source`; the `l1_hot_channel` gate's coverage is the share
of successful noctx generations that started from the seed. The grid lives in
`matrix_l1_grid.yaml` (arms `C7*`); the machinery line prints the draw
counters (`hot-ngram seeds: N draws, empty X%`).

## Pool composition grid (M3R-110)

`matrix_pool_grid_1.yaml` / `matrix_pool_grid_2.yaml` (arms `C8*`) sweep the
ctx-only knobs `reply_context_start_bias`, `context_anchor_splice_probability`
and `generation_attempts_with_context`. The `pool_composition` gate's coverage
is the SHIFT of the context-start share of winners against C0 (start sources
context / hidden_context / context_spliced); the winner's start source is
resolved through extension and mutation (`resolve_start_source`), so a
rewritten reply keeps its attempt's source. Run both files in both modes; C0
is deterministic and its summaries must match between the files.

