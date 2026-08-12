## Why

The chain has no sense of time. Every transition carries a single integer `cnt`
that never decays and never records when it was learned, so a phrase said a
thousand times two years ago outweighs the language the chat actually speaks
today, and nothing in the schema can even express the difference. TZ §7-8
answers this with two layers — a decaying short-term counter blended with a
sublinearly compressed long-term count — and Phase 3 is where the schema finally
gets the timestamps that make any of it, including every later freshness metric,
possible.

Timing matters for a second reason: historical `first_seen` values are
unrecoverable (rebuilding the live chain is forbidden by ADR-005), so every row
that exists today will be stamped with the migration date. The temporal record
only starts once this migration lands — every week of delay is a week of
resolution the project never gets back.

## What Changes

- **Migration M2R-200**: `first_seen`, `last_seen`, `s_value`, `s_updated_at`
  added to the four chain tables (`starts`, `starts3`, `transitions`,
  `transitions3`). Historical rows get the migration timestamp and `s_value = 0`
  — recorded as a known, permanent limitation, not papered over.
- **M2R-210 — decaying short counter**: learning updates both layers atomically
  in the existing single learn transaction, using the exact-decay identity from
  TZ §7.1 (`s_value = s_value·2^(−Δt/hl) + 1`), which reproduces the sum of
  individually decayed observations at O(1) storage.
- **M2R-210 — blend**: `P = α·P_short + (1−α)·P_long` over the union of tokens,
  with the long layer sublinearly compressed (`log` or `count^β`) before
  normalization. α comes from the chat's mood.
- **Neutral by default**: every α defaults to 0, which takes an early return to
  the current sampling path — byte-identical to Markov 1.x, proven by
  `generation_hash` exactly as in Phases 1 and 2. Compression is applied only
  inside the blend path, so a disabled blend cannot silently reshape weights.
- **Half-life changes reset the short layer** with an explicit warning, because
  a decayed counter is only meaningful against the half-life it accumulated
  under (TZ §7.2).
- **Temporal eval fixture (new, owner decision 2026-08-12)**: a deterministic
  tool that replays `db_prod_copy`'s retained messages with their real
  `created_at` values into a separate eval database, producing the
  `snapshot_temporal` that doc 05 §1.1 has owed the project since Phase 0. This
  is an eval fixture built beside the chain, never a rebuild of it — ADR-005 is
  untouched.
- **Pre-registered `phase3_temporal` gate** written before any grid runs, and
  the C2 arm measured on the fixture in this change rather than weeks later.
- Not in this change (roadmap doc 03 split): M2R-215 calibration of the shipped
  α/β defaults, and M2R-220 GC.

## Capabilities

### New Capabilities
- `generation-temporal-layer`: the two-layer transition model — what the short
  counter must compute, how the layers blend, what a disabled blend guarantees,
  and what changing the half-life does.

### Modified Capabilities
- `generation-eval`: adds `snapshot_temporal` as a buildable artifact, the
  `phase3_temporal` gate, and the C2 ablation arm; removes the doc 05 v1.0.1
  amendment's "temporal metrics are impossible" clause for the fixture path.
- `generation-telemetry`: the blend's α, the short layer's effective weight and
  its coverage of the pool become trace and `/stats` numbers, so a neutral
  configuration is visibly neutral.
- `runtime-knob-validation`: bounds for the new knobs on both the env and `/set`
  paths, plus the half-life knob's reset side effect, which is the project's
  first `/set` that discards stored data and therefore needs an explicit
  contract.

## Impact

- **Schema**: migration `018` on four tables. `ALTER TABLE ADD COLUMN` is a
  metadata-only operation in SQLite; the audit rates the risk low. Storage grows
  by four columns over ~67k rows on the prod copy.
- **Hot path (learn)**: `app/infrastructure/database.py:201-349` gains a
  read-modify-write of the short counters for the transitions of one message
  (bounded by message length) inside the transaction that already exists.
- **Hot path (generate)**: `app/core/markov.py` — the sampler's weight source
  becomes a blended distribution rather than a raw `cnt`, which also feeds
  `pool_diagnostics` and therefore Phase 1 telemetry and Phase 2's entropy
  input.
- **Cache**: Phase 1's incremental transition cache now holds time-dependent
  values; `s_eff` must be resolved against an injected clock, or determinism and
  the eval protocol's bit-for-bit reproducibility both break.
- **Config**: new knobs in `app/config/registry.py`, `Settings`, `RuntimeState`,
  `.env.example`, `/config full`.
- **Docs**: `docs/v2/00_STATUS.md`, `docs/GENERATION_PIPELINE.md`,
  `docs/ARCHITECTURE.md`, a report under `docs/eval_reports/`.
- **Riding along, no functional content**: `openspec archive
  markov2r-phase2-entropy-sampling` as its own commit (owner decision
  2026-08-12: no PR just for an archive).
- **Decision this change unblocks**: whether the disabled Phase 2 code is kept
  or deleted — deferred by the owner to the Phase 3 measurements, since Phase 3
  is the one thing on the roadmap that reshapes the weight distribution whose
  bimodality made Phase 2 inert.
