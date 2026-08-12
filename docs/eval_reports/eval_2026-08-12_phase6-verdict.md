# Phase 6 verdict — anti-cycle closed without implementation

**Date:** 2026-08-12
**Gate:** `phase6_anticycle` in `tools/eval/eval_thresholds.yaml`
(`cycle_detection_rate_min = 0.05`, `cycle_harm_rate_min = 0.02`), two-dimensional
(ADR-015), registered before any gated data was examined.
**Data:** C0 frozen config on `db_prod_copy`, **9000 generations** (3000 ×
seeds 42/1337/2026), prompt set `308b7deaea0f`. `cycle_detection_rate` measured
with the eval's own `has_token_cycle` (period-2/3 token cycle); no second
detector was built.

## Verdict

**Phase 6 closes without implementation. M2R-600 (cycle detector) and M2R-610
(cycle penalty + dynamic jump) are NOT built.** No threshold was changed.

The gate is two-dimensional: it opens only if cycles are **both** frequent
(`cycle_detection_rate ≥ 0.05`) **and** harmful (`cycle_harm_rate ≥ 0.02`), and
non-exceedance of either closes the phase. The detection arm is decisively
below its bar, so the conjunction cannot hold and the manual harm round is not
required (an AND-gate one of whose arms provably fails is already decided).

| Measure | Value | Threshold | Margin |
|---|---|---|---|
| `cycle_detection_rate` (9000 gens) | **0.0001 [0.0000, 0.0003]** | ≥ 0.05 | whole CI **167× below** the bar |
| `cycle_detection_rate` (1500 gens, phase5-c4 run) | 0.001 [0.000, 0.002] | ≥ 0.05 | consistent, also decisive |
| `cycle_harm_rate` | not measured | ≥ 0.02 | not required — detection arm already fails the conjunction |

One cyclic reply in 9000. The rate is not merely under the bar; its entire
confidence interval sits two-plus orders of magnitude below it.

## Why cycles are this rare — existing mechanisms already handle them

The gate measures cycles in *output*, after the pipeline. They are rare there
not by luck but because three existing mechanisms — built against repetition,
not cycles specifically — already suppress them, over a chain that structurally
rarely loops:

1. **Per-step repetition penalty** (`weighted_next_choice`): the immediately
   preceding token is crushed (`weight ×= max(0.01, 1 − 0.96·strength)`), the
   token two back is damped (`×= max(0.05, 1 − 0.70·strength)`), and repeats in
   the recent window divide the weight. Period-1/2 cycles are attacked before
   they can form.
2. **State dedup in the walk** (`visited_triplets`): candidates re-entering an
   already-visited order-3 state are filtered out — and re-entering a state is
   exactly what a cycle is.
3. **Candidate-level repetition penalty** (`candidate_scorer.py`): a cyclic
   candidate has maximal repeated-bigram ratio, scores low, and loses in
   best-of-N selection.

Plus the structural fact behind the Phase 2/3 verdicts: ~99% of order-3 states
have a single continuation, so the chain is mostly linear and rarely closes a
loop.

Adding M2R-600/610 would be a **third** layer on the hot generation path, aimed
at a phenomenon already caught twice and structurally rare — the exact
over-engineering the gate exists to prevent. And "cycle ≠ harm": the few cycles
that survive ("да да да") are often legitimate Pepe replies, so a hard
anti-cycle jump could hurt more than the tunable repetition penalty already in
place.

## Reversibility

Nothing is built or removed, so there is nothing to roll back. The verdict is
recorded as "closed at these numbers", not "impossible": if a future corpus made
cycles frequent, the same gate would reopen on the next run. The gate-resolution
logic (a two-dimensional AND-gate closes on a decisively-failing arm) now lives
in `tools/eval/report.py` and is covered by `tests/test_eval_protocol.py`.
