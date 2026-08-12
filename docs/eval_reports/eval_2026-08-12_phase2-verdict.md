# Phase 2 verdict — entropy sampling closes with a negative result

**Date:** 2026-08-12
**Data:** `eval_2026-08-12_phase2-grid.md` (protocol volumes: 500 generations ×
seeds 42/1337/2026 per arm, `db_prod_copy` 2026-07-13, prompt set
`308b7deaea0f`), grid definition `tools/eval/matrix_phase2_grid.yaml`.
**Gate:** `phase2_entropy` in `eval_thresholds.yaml`, registered 2026-08-12
before the grid ran.

## Verdict

**All six arms fail the gate. Both Phase 2 features ship disabled by default
(`markov_entropy_temp_gain = 0`, `markov_branching_degenerate_max = 0`), which
is byte-identical to Markov 1.x.** No threshold was changed after the fact.

| Arm | copy Δ | distinct-2 Δ | distinct-3 Δ | affinity-без-копий Δ | p95 | verdict |
|---|---|---|---|---|---|---|
| C1gain_neg06 | −0.010 | +0.005 | +0.007 | +0.004 | 29.3 ms | fail |
| C1gain_neg03 | −0.001 | −0.003 | −0.004 | +0.003 | 28.8 ms | fail |
| C1gain_pos03 | −0.001 | +0.000 | +0.001 | −0.001 | 28.8 ms | fail |
| C1gain_pos06 | +0.003 | +0.001 | +0.002 | +0.001 | 30.4 ms | fail |
| C1branch_15 | −0.023 | +0.002 | −0.000 | **−0.032 \*** | 27.0 ms | fail |
| C1branch_25 | **−0.039 \*** | +0.011 | +0.007 | **−0.040 \*** | 25.5 ms | fail |

`*` = interval excludes 0. C0 reference: copy 0.222, affinity-без-копий 0.209,
p95 28.6 ms. Full intervals in the grid report.

M2R-100 fails on "distinct-2/3 must rise significantly": every delta is inside
the noise. M2R-110 fails on the affinity floor.

## Why M2R-100 does nothing here — the mechanism, not a shrug

The per-step entropy distribution of this corpus was measured directly (7034
sampled walk steps, seed 42, 200 generations through the real pipeline):

| branching | share | | H_norm bucket | share |
|---|---|---|---|---|
| = 1 | **78.8%** | | [0.0, 0.2) | **78.8%** |
| = 2 | 7.4% | | [0.2, 0.8) | **0.3%** |
| ≥ 5 | 9.0% | | [0.8, 1.0) | **20.9%** |

mean H_norm = 0.207, mean branching 2.93.

**The signal is bimodal, and neither mode is a place where temperature can do
work.**

- **78.8% of steps have exactly one continuation.** Entropy is 0 and the choice
  is forced: no temperature, however set, changes the token.
- **Almost all of the rest sit at H_norm ≥ 0.8** — near-uniform pools. On a
  uniform pool `cnt ** power` is flat for *every* power, so the exponent has
  nearly no leverage there either.
- **The middle, where a temperature knob would actually bite, is 0.3% of steps.**

This also explains why the mean applied temperature barely moves between arms
(2.77 → 2.76): with the pivot set at the measured mean, the knob redistributes
rather than shifts — as designed — but on a bimodal distribution "redistribute
around the mean" moves the 79% that cannot respond and the 21% that respond
weakly, and nothing else.

So TZ §6's premise — the chain carries information about its own uncertainty,
and that information should drive the temperature — holds mathematically and is
**empty on this corpus**: the uncertainty here is effectively a binary
forced/free flag, not a graded signal.

Two earlier findings corroborate this rather than contradict it:

- the pre-implementation audit measured ~98% of order-3 states with exactly one
  continuation (the walk replays its source message);
- the 2026-07-20 sweep found `randomness_strength` — a **global** version of the
  same exponent, swept over its whole range 0..3 — inert on every metric. A
  per-pool modulation of an exponent whose global swing does nothing could not
  have done more. Phase 2 has now measured that directly instead of inferring it.

## What M2R-110 actually found

The branching-aware target is not inert — it trades. At bound 2.5 it cuts
copying significantly (−0.039, CI [−0.069, −0.011]) **and** cuts topicality
significantly (−0.040, CI [−0.062, −0.020]), with latency falling 28.6 → 25.5 ms.

That is a coherent story rather than noise: extra candidates are how the scorer
finds an on-topic reply, so capping them early removes both the quotes and the
hits. The gate's affinity floor exists precisely to refuse this trade, and it
did. Second-order effect noted in advance and confirmed here: a reduced target
also caps slot mutations, so degenerate chains get fewer mutation attempts.

## ADR open question #2 — answered

> "Какое отображение энтропия → температура работает лучше?"

**Neither.** On this corpus, over ±0.6 gain in both directions at protocol
volume, no mapping of normalized entropy to temperature moves any gated metric
outside the noise. The question is answered with data, and the answer is that
the mapping is not a useful lever *here* — not that one direction beat the other.

## Where this could become live again

The null result is a property of the *distribution*, not of the code. The one
thing on the roadmap that changes that distribution is Phase 3: a decayed short
layer blended with a compressed long layer changes the weights entropy is
computed from, and sublinear compression of `count` specifically flattens the
peaked pools that make up the 78.8%. If Phase 3 lands and the bimodality
softens, this knob becomes worth re-measuring — with a **new** pre-registered
gate, not this one.

Re-measuring is cheap because the code stays: it lives behind an early return at
gain 0 and is covered by unit, property and integration tests. Deleting it is a
legitimate alternative — a knob that provably earns nothing today is dead weight
— and that call belongs to the owner.
