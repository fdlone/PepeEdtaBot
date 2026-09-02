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
  §3.5 gate. Since 2026-09-02 the meme list uses the channel's own
  content-n-gram predicate (`app.core.hot_ngrams.is_content_ngram`), so the
  set and the L1 route read `chat_hot_ngrams` the same way. Current version
  `bababb4b7693` (snapshot 2026-09-01); reports on `308b7deaea0f` are not
  comparable with it. The `version` field is a content hash; `load_prompts` rejects
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

## Hot route grid (M3R-230)

`matrix_l1_route_grid.yaml` (arms `C7r*`) moves `hot_ngram_slot_ratio` at the
hotness thresholds 2 / 0.25 and is judged by the same pre-registered
`l1_hot_channel` gate. Note the instrument dependency: the gate's must-improve
reads `historical_meme_rate` against the meme set of `eval_prompts.yaml`, so a
prompt set generated on an older snapshot cannot see today's hot n-grams (the
2026-09-02 run: zero overlap) — regenerating the prompt set is a version
change and a deliberate comparability break.

## Knob census (M3R-151)

`python -m tools.eval.knob_census plan | run | launch | report`. Builds arms
from the registry (both domain extremes per measurable knob, a flip for
booleans, and the same extremes with the parent enabled for gated knobs — see
`GATED_BY`), runs them against C0 in both modes (`launch` spawns detached
workers, `report` stitches the JSON samples), and classifies every knob by
the pre-registered `knob_census` rule in `eval_thresholds.yaml`: dead / gated
/ inert / weak / strong. Knobs the harness cannot exercise are listed as
outside the offline measurement. The report carries numbers only; the
owner-facing verdict is appended by hand below it.

Run of 2026-09-02 (`eval_2026-09-02_knob-census.md`, samples in
`knob-census-2026-09-02.json`): `launch --workers 9` spawned 10 windowless
workers (2 × C0 + 8 chunks of 36 arms) and finished in 34 minutes; both C0
matched the version baseline bit for bit.

## Route gate (M3R-220)

`matrix_route_gate.yaml` (arms `C11*`) is the promotion gate of a candidate
route (route-gate). One `route_gate` block for every route: the route under
test is read from the data (the single route present in the arm's pools and
absent from C0's), never from the arm id. Coverage = presence share; the
must-improve is the paired drop of the single-trajectory share (significant
and at least 5 p.p.) — the escape form M3R-011 promised, since the mean
window escape is one C0 already takes; affinity without copies, copy and
repetition must not worsen; pool ECB >= 4; p95 <= 150; connectedness from
the solo round in ctx (`rating_rounds/assoc-gate`). Two modes.

## Associative route pilot (M3R-200)

`matrix_assoc_pilot.yaml` (arms `C10*`) moves `assoc_slot_ratio`, the slot
budget of the route that grows candidates around associates of the message's
anchors (assoc-route-pilot). The `assoc_pilot` block holds VIABILITY bars,
not a promotion gate: the arm reads `viable` / `not viable` / `insufficient
data` against the pilot's four questions — presence share (Q1, below the
floor = did not exercise), paired pool-ECB and window-escape deltas (Q2, a
significant ECB drop = associates duplicate the walk), p95 (Q3), affinity
without copies printed with its CI plus copy / repetition must-not-worsen
(Q4). Two modes, as every route gate.

## Selection-window grid (M3R-100)

`matrix_selection_grid_1.yaml` / `matrix_selection_grid_2.yaml` (arms `C9*`)
move the knobs that define the selection window — `selection_score_margin`,
`context_relevance_weight` (cap follows) and `selection_diversity_bonus`
(selection-knobs). The `selection_window` gate's coverage is the paired drop
of the single-trajectory share (`window_escape < 2`, the form M3R-011
promised); then the window escape must rise significantly while affinity
without copies, copy and repetition may not worsen. Run both files in both
modes.

