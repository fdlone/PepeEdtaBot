# Phase 3 verdict — the temporal layer ships, the blend stays off

**Date:** 2026-08-12
**Data:** `eval_2026-08-12_phase3-grid.md` (protocol volumes: 500 generations ×
seeds 42/1337/2026 per arm, reconstructed temporal snapshot built from
`db_prod_copy`'s retained messages, prompt set `308b7deaea0f`), grid definition
`tools/eval/matrix_phase3_grid.yaml`.
**Gate:** `phase3_temporal` in `eval_thresholds.yaml`, registered 2026-08-12
before the grid ran.

## Verdict

**M2R-200 (the schema and the learning path) ships enabled — it is how the chat
starts having a temporal record at all. M2R-210 (the blend) ships disabled
(`markov_alpha_* = 0`), byte-identical to Markov 1.x (`generation_hash`
`5a72e2d4`).** All nine grid arms fail the gate. No threshold was changed after
the fact.

| Arm | freshness Δ | historical meme Δ | copy Δ | affinity-без-копий Δ | p95 | verdict |
|---|---|---|---|---|---|---|
| C2a03_log | 0.000 | −0.008 | −0.003 | −0.005 | 21.7 ms | fail |
| C2a03_pow50 | 0.000 | −0.008 | −0.003 | −0.005 | 23.1 ms | fail |
| C2a03_pow75 | 0.000 | −0.011 | −0.003 | −0.005 | 23.8 ms | fail |
| C2a05_log | +0.002 | 0.000 | −0.001 | −0.005 | 26.6 ms | fail |
| C2a05_pow50 | +0.002 | 0.000 | −0.001 | −0.004 | 22.5 ms | fail |
| C2a05_pow75 | +0.002 | 0.000 | −0.001 | −0.005 | 21.6 ms | fail |
| C2a07_log | +0.004 | −0.003 | −0.002 | −0.002 | 22.4 ms | fail |
| C2a07_pow50 | +0.003 | −0.003 | −0.002 | −0.002 | 23.4 ms | fail |
| C2a07_pow75 | +0.003 | −0.003 | −0.002 | −0.002 | 22.6 ms | fail |

No interval excludes zero anywhere in the table. Every arm fails on the same
clause — freshness did not rise significantly — and nothing else worsened
either. This is not a trade the gate refused, as M2R-110 was in Phase 2. It is
inertness.

## Why the blend cannot move anything here

Measured directly on the fixture's 5507 order-3 states, blend off versus blend
on at the strongest arm (α = 0.7, log), same corpus, same states:

| pool shape | blend OFF | blend ON |
|---|---|---|
| exactly one continuation (H = 0) | **5454 (99.0%)** | **5454 (99.0%)** |
| H_norm in [0.2, 0.8) — where a weight knob has leverage | 0 (0.0%) | 23 (0.4%) |
| H_norm in [0.8, 1.0] — near-uniform | 53 (1.0%) | 30 (0.5%) |

**99.0% of steps have exactly one stored continuation, and no re-weighting
scheme can create a choice where the corpus stored one.** Both layers live on
the same rows: the blend re-weights the candidates a state already has, it never
adds candidates. That is the wall, and it is the same wall Phase 2 hit from the
other side.

## What this means for the Phase 2 code — the decision this phase was asked to unblock

The Phase 2 verdict left one hypothesis alive: that Phase 3 would flatten the
peaked pools, move the entropy distribution's mass into the middle band, and
make an entropy→temperature knob worth re-measuring. The table above is the
first direct test of it, and the answer is **partly, and only inside the sliver
that already had a choice**:

- among the 53 pools that offer a real choice, the blend moved **23 of them
  (43%)** into the band where a temperature knob has leverage — from zero;
- but that sliver is 1.0% of all steps on this corpus, so the addressable mass
  went from ~0% to ~0.4%.

**Extrapolation, labelled as such:** the full prod corpus is less degenerate
than this fixture — Phase 2 measured 78.8% single-continuation there against
99.0% here, because the fixture is built from ~1000 retained messages while the
real chain holds tens of thousands of transitions. If the 43% conversion rate
carries over to the full corpus's 20.9% multi-option steps, the workable middle
band would go from Phase 2's measured 0.3% to roughly 9% of all steps — a large
relative change on a small absolute base. **This is arithmetic on an assumption,
not a measurement, and it must not be quoted as one.**

**Recommendation to the owner: keep the Phase 2 code, do not delete it yet.**
The reasoning is that the one hypothesis that could revive it moved in its
favour rather than against, and the cost of keeping it is a knob behind an early
return with full test coverage. But the case is now specific enough to be
settled properly instead of argued: re-measure on the real corpus after a live
accumulation window, with a **new** pre-registered gate — `phase2_entropy` is
spent, and reusing it would be exactly the post-hoc move the pre-registration
rule exists to prevent.

If the owner prefers to delete it, that is a defensible call on the same
numbers: 0.4% measured, the rest extrapolated.

## Honest limits of this measurement

- **The fixture is a reconstruction, not live accumulation.** It replays the
  retention window (1000 messages, 2026-06-03..2026-07-12, 39 days), so its
  corpus is an order of magnitude smaller than the chain and more degenerate.
  Its C0 is not the frozen baseline; every delta above is computed within the
  fixture.
- **The fresh/historical split is a construction.** "Fresh" means a token first
  seen in the last 14 days of the replay — 1310 of 3137 vocabulary tokens.
- **The blend is untested against real accumulation.** Nothing here says the
  blend is useless on the live chain; it says the blend is inert on a corpus
  where 99% of states have one continuation. M2R-215 re-measures once the live
  chain has accumulated its own timestamps.
- **The reading moment does not affect the blend at all** (found during
  implementation, covered by tests): the short layer is normalized within the
  pool, so uniform decay cancels. The fixture's choice of evaluation moment
  therefore cannot have biased any arm.
