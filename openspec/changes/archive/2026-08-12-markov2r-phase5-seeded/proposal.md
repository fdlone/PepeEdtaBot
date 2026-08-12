# Proposal: markov2r-phase5-seeded

## Why

The reverse index and the df aggregate landed with M2R-400; nothing reads them
yet. This change is the experiment they exist for (ADR-012 Provisional,
TZ §9.4–9.6): pick an anchor token the chat actually uses, grow a reply around
it in both directions, and let those seeded candidates compete in the pool. It
is the second and final half of Phase 5 — M2R-410 (seed score + bidirectional
generation) and M2R-420's telemetry/eval, plus the pre-registered
`phase5_promotion` gate made computable. The promotion decision itself
(M2R-430) is a follow-up: it needs df accumulated on live prod, not the
retention window.

## What Changes

- **Seed selection (TZ §9.4)**: `seed_score = normalized_idf × support_factor ×
  branching_quality`, a pure function over the df aggregate, model counts and
  forward+reverse branching. Deliberately not max-IDF — a junk unique token
  (`foobar123`) has maximal IDF and no support. Branching is a **band, not a
  threshold** (trapezoid: too little and generation stalls, too much and the
  anchor means nothing). Stopwords and tokens below a length floor are excluded
  before scoring. No candidate clears `MARKOV_SEED_MIN_SCORE` → the seeded
  branch is skipped (transparent fallback).
- **Bidirectional generation (TZ §9.5)**: from a seed token, the tail grows
  forward on the existing chain and the head grows backward on the reverse
  order-2 index; the entropy/temporal rules of §6–8 apply in both directions.
  The head/tail length split is configurable inside the existing length budget.
- **Seeded candidates in the pool**: `MARKOV_SEEDED_CANDIDATE_RATIO` (0–0.7,
  **default 0**) of the best-of-N pool are seeded candidates. They pass the
  common scorer with no priority (ADR-008).
- **Telemetry (TZ §9.6)**: `seeded_present_rate` and
  `seeded_win_rate_given_present` — separate denominators, because "a seeded
  candidate rarely appears" and "it appears but loses" are different findings.
- **Eval C4 arm + gate**: the declared-but-unavailable C4 arm becomes
  available; the `phase5_promotion` gate, currently hardcoded to "does not
  exist before Phase 5", is computed from the run.
- **Neutral default**: `ratio = 0` reads nothing seeded and changes no reply —
  `generation_hash` stays bit-identical. Raising it needs the gate to pass.

## Capabilities

### New Capabilities

- `generation-lexical-anchoring`: seed scoring, bidirectional generation from a
  seed, seeded candidates in the pool with no scoring priority, the transparent
  fallback, and the neutral-default contract.

### Modified Capabilities

- `generation-telemetry`: the two seeded denominators become observable.
- `generation-eval`: the C4 arm ships and the `phase5_promotion` gate is
  computed rather than stubbed; it reports `insufficient data` while df has not
  accumulated on prod, never `pass`.

## Impact

- **Code**: new pure seed-scoring module (`app/core/seed.py`); a reverse walk
  and seed-anchored assembly in `MarkovGenerator`; seeded-candidate wiring in
  `ResponseGenerator`; df/branching reads via the M2R-400 API; new runtime
  knobs (seed band, min score, min token len, candidate ratio, head/tail
  split) with bound validation; the eval runner's C4 df-population pass.
- **Performance**: seeded candidates cost extra walks only when the ratio is
  non-zero; the reverse walk reuses the cached-pool machinery. p95 is a gate
  criterion and is measured.
- **Storage**: none new — the reverse index and df already shipped (M2R-400);
  `storage_growth_max_share` is evaluated against that existing footprint.
- **Risk**: the df aggregate is empty until prod accumulates it after the
  migration-020 restart, so the eval df is window-approximated and the true
  promotion verdict waits for a live-accumulated run — the gate says
  `insufficient data` until then, exactly like Phase 4's manual gate. ADR-012's
  cheap-refusal path stays open: ratio back to 0, reverse structures frozen.
