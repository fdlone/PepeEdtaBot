# Design: markov2r-phase2-entropy-sampling

## Context

See `proposal.md` — Why. The design-relevant state of the code:

- **The temperature already exists, under another name.** Per-step weights are
  `weight = max(cnt, 1) ** frequency_power` (`markov.py:663`, `707`, `736`).
  A power is an inverse temperature: `power = 1/T`. `frequency_power` is derived
  from `randomness_strength` (`markov.py:1861-1865`) and then flattened by the
  exploration roll (`_roll_exploration` → `exploration_adjusted_power`). So TZ §6's
  "`RANDOMNESS_STRENGTH` remains the scale of `T_base`" is already true; what is
  missing is the per-pool term.
- **The entropy already exists, per step.** `_DiagnosticsAccumulator.note_pool`
  is called immediately before each walk-step sampling call (`markov.py:1757-1760`
  for order-3, `1786-1787` for order-2), on every attempt of every generation —
  `diagnostics` is unconditionally constructed at `markov.py:2028`. The value is
  computed and dropped. Phase 2 is therefore mostly a wiring change, not new math
  on the hot path.
- **Five call sites, two of them relevant.** `weighted_next_choice` is called at
  `markov.py:1119`, `1349`, `1410` (completing a *start* state) and at `1761`,
  `1788` (walk steps). Only the latter two have diagnostics; Phase 1 deliberately
  left start pools uninstrumented, and this change keeps that boundary.
- **Best-of-N.** `GENERATION_ATTEMPT_BUDGET = 10`, `CANDIDATE_TARGET = 5`, both
  module constants in `response_generator.py:49-51`; `target` is computed once
  (`:490`) and used as the loop's break condition (`:763`) and as a guard on slot
  mutation (`:723`).
- **Frozen references.** C0 baseline on `db_prod_copy` 2026-07-13: success 1.0,
  copy 0.222, affinity 0.280 / 0.209 without copies, p95 48 ms against a 150 ms
  budget. `tools/generation_hash.py` is the byte-identity guard.

## Goals / Non-Goals

**Goals:**

- Per-pool temperature derived from that pool's own normalized entropy, with the
  neutral configuration provably identical to 1.x.
- A candidate target that reflects how much choice the chain actually had.
- An ablation that can tell "entropy-aware" apart from "just hotter on average" —
  the confound that would otherwise make any positive result unreadable.

**Non-Goals (design level):**

- No change to start-state selection, pool membership, backoff, order-mix, jumps,
  or any acceptance gate.
- No per-chat adaptation of the pivot (one global pivot; see Open Questions).
- No attempt to make `candidate_selection_temperature` do anything — that knob is
  a different mechanism (softmax over *candidates*, culled by
  `SELECTION_SCORE_MARGIN`, measured nearly inert in 2026-07-09). Per-step token
  sampling is not culled by any margin, so the inertness result does not transfer
  in either direction.

## Decisions

### D1 — Entropy modulates the base power, before the exploration roll

`weighted_next_choice` gains one keyword-only parameter carrying the pool's
already-computed normalized entropy; when it is absent (the three start-state
call sites, and every existing test) behavior is exactly today's. Inside, the
adjustment is applied to `power` *before* `_roll_exploration`, so exploration
keeps flattening on top of the current temperature rather than replacing it.

*Alternative rejected:* applying the adjustment after the roll. It would let
entropy undo the exploration flattening, silently changing what
`randomness_strength` means at the same time.

*Alternative rejected:* recomputing entropy inside `weighted_next_choice` from
`items`. Same numbers, computed twice per step, for the convenience of not
passing a float.

### D2 — Neutrality is an early return, not a float coincidence

With the feature off or `GAIN == 0.0`, the adjustment function returns the input
power unchanged by an explicit branch taken before any arithmetic. Relying on
`x * 1.0 == x` would still be exact in IEEE754, but the clamp is not: a
`T_base = 1/power` outside `[T_min, T_max]` would be pulled to a bound even at
zero gain, and the identity would break for extreme `randomness_strength`. The
early return makes the contract structural. No RNG draw is added or removed on
either branch, so the whole downstream stream is untouched — which is why
`tools/generation_hash.py` can prove the identity instead of merely suggesting it.

### D3 — The pivot is measured, not chosen

`H_pivot` is set to the mean normalized entropy actually observed on the eval
corpus (Phase 1 telemetry already reports `mean_normalized_entropy`), so the
average step keeps its current temperature and only the tails move. A pivot
picked by hand would smuggle in a global temperature shift and contaminate every
delta in the ablation.

### D4 — Both signs of GAIN go on the grid

The two directions are opposite bets and the project has no data to prefer either:

- **`GAIN > 0`** (TZ §6 as literally written, and ADR-003's reading — "high-entropy
  states allow exploration"): open pools get hotter, confident pools get colder.
  The risk is explicit: the confident pools are exactly where verbatim replay
  lives, and sharpening them further could raise the copy metric. The gate is
  built to catch that.
- **`GAIN < 0`**: flattens precisely the near-degenerate steps that replay their
  source message, at the cost of muddying the steps the model is sure about.

Grid: `GAIN ∈ {−0.6, −0.3, +0.3, +0.6}` plus the neutral `0.0` as the identity
check. ADR open question #2 is answered by this run and its answer is recorded in
the report, not in this document.

### D5 — A flat-temperature control arm

One extra arm runs a *constant* temperature shift tuned to reproduce the mean
applied temperature of the winning entropy arm. If the entropy arm's gains are
matched by the flat arm, the effect was never about entropy and the phase says so.
This costs one arm and is the difference between a measurement and a story.

### D6 — M2R-110 reduces attempts, it does not add them

The branching-aware target moves *down* from `CANDIDATE_TARGET`, never up: on a
chain whose pools are near-degenerate, candidates 2..5 are near-duplicates of
candidate 1, and the scorer picks between copies of the same walk. The target is
recomputed after each accepted candidate from the running mean of the accepted
candidates' `mean_branching` (already in the trace); it never drops below a floor
and the attempt budget is untouched, so a chain that keeps failing the gates still
gets its ten tries.

*Alternative rejected:* raising the target on wide branching. It spends latency
exactly on the chats with the largest pools, against a budget (150 ms) that Phase
3 and Phase 5 will both eat into.

*Noted side effect:* `target` also caps slot mutation (`response_generator.py:723`),
so a reduced target reduces mutation opportunities on degenerate chains. This is
consistent (there is little to mutate there) but it is a second-order behavior
change and belongs in the report, not in a footnote.

### D7 — Matrix arms

| Arm | Content |
|---|---|
| `C1` | Phase 2 as shipped (both knobs at final defaults) — the phase's protocol C-config |
| `C1a` | entropy → temperature only |
| `C1b` | branching → candidate target only |
| `C1flat` | flat temperature shift matched to `C1a` (control, D5) |

`C1a`/`C1b` satisfy the "one arm per knob" rule this change adds to
`generation-eval`; `C1` is what the default would ship.

### D8 — Thresholds are registered before the run

Added to `eval_thresholds.yaml` in the first commit of the implementation, with
their rationale, before any Phase 2 number is produced:

- `exact_copy_delta_max: 0.0` — a *significant* increase disqualifies (same
  convention as `phase7_order4`); baseline copy is 0.222.
- `distinct2_delta_min: 0.0` and `distinct3_delta_min: 0.0` — must be positive
  **and** significant; an insignificant delta is a fail, not a pass, because the
  phase's stated benefit is diversity.
- `affinity_without_copy_delta_floor: 0.0` — must not be *significantly* below
  zero; baseline 0.209.
- `generation_p95_ms_max: 150` — reuses the existing `performance` section.

Comparisons are `distinct-2/3` within one run at equal volume only — the metric is
a type/token ratio and moves with the denominator (recorded in
`distinct_basis_tokens`, the lesson from the 2026-07-21 sweep).

## Risks / Trade-offs

- **Positive gain raises copying** → the gate's first threshold is exactly this,
  and the negative-gain arms exist because the opposite bet is plausible.
- **A gain that only shifts mean temperature** → D3 (measured pivot) + D5 (flat
  control arm).
- **M2R-110 changes attempt counts, so p95 moves** → measured in both directions
  in the same run; the phase does not ship if p95 leaves the budget.
- **The eval corpus is blind to part of the live experience** — C0 silences flavor
  and emoji by convention, and mood modifiers are not modeled. A change that reads
  fine offline can still read differently in the chat → the knobs are runtime
  settable via `/set`, so reverting is a message, not a deploy.
- **Two knobs in one phase muddy attribution** → four arms instead of one, at the
  cost of a longer run; this is the price the owner accepted when choosing to land
  M2R-100 and M2R-110 together.
- **A reduced target reduces slot mutations** (D6) → reported explicitly rather
  than discovered later as an unexplained novelty delta.

## Migration Plan

No schema change, no migration. Rollout order:

1. Merge with every knob neutral — the released build is byte-identical to 1.x and
   `tools/generation_hash.py` proves it in CI.
2. Run the grid offline on `db_prod_copy`; pick the winning arm strictly by the
   pre-registered gate.
3. Gate passes → raise the registry default in the same change, citing the report.
   Gate fails → default stays 0, phase closes with the negative result recorded in
   `docs/v2/00_STATUS.md` and the report referenced.
4. Rollback at any point: `/set markov_entropy_temp_gain 0` (and
   `markov_branching_candidates_enabled false`), no restart, no data touched.

## Open Questions

- Should `H_pivot` eventually be per-chat? Chats differ in mean branching, and a
  global pivot means a quiet chat and a busy one sit on different parts of the
  curve. Deferred: it needs per-chat telemetry history that does not exist yet, and
  it does not change any requirement or task here.
- Whether the walk should also modulate the *start* pools once Phase 5 instruments
  them. Out of scope by construction (no diagnostics there today).
