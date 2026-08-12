# Design: markov2r-phase6-anticycle-verdict

## Context

See proposal.md — Why. Design-relevant state:

- The Phase 6 gate is two-dimensional (ADR-015): `cycle_detection_rate ≥ 0.05`
  AND `cycle_harm_rate ≥ 0.02`. Roadmap Phase 6: non-exceedance of **either**
  threshold closes the phase without implementation, numbers recorded.
- `cycle_detection_rate` is already measured (`has_token_cycle` over generated
  replies, `metrics.py` §3.2) — a period-2/3 token cycle detector. The last
  protocol run reads **0.001 [0.000, 0.002]**.
- `cycle_harm_rate` has an automatic component (candidate rejections correlated
  with cycles) plus a manual sample-rating component (doc 05 §5). The manual
  component has not been collected.
- `tools/eval/report.py` renders every gate. The Phase 6 row is hardcoded to
  `INSUFFICIENT` with a note that the harm round is pending — it never resolves
  even when detection is decisively out of range.
- The shared `_verdict` helper already encodes "a demonstrated failure outranks
  a missing part" for single-dimension gates; the Phase 6 row does not use it.

## Goals / Non-Goals

**Goals:**

- Render the Phase 6 gate as a proper close/fail when detection is decisively
  below its threshold, per the pre-registered rule.
- A decisive detection-rate interval from a larger run.
- Record the closed verdict with numbers; build nothing.

**Non-Goals:**

- M2R-600/610 (cycle detector, cycle penalty, dynamic jump). The gate is
  closed; they are not built.
- Any change to generation, schema, or thresholds.
- The manual harm round — an AND-gate whose detection arm decisively fails
  cannot open regardless of harm, so the round would measure the harm of a
  ~0.1%-of-replies phenomenon for no decision value.

## Decisions

### D1. "Decisively below" = the whole CI lies below the threshold

The detection arm fails the conjunction when its bootstrap interval's **upper**
bound is below `cycle_detection_rate_min`. This is stronger than the point
estimate missing: it says the true rate is below the bar at the protocol's
confidence, not merely that this sample landed low. At 0.001 [0.000, 0.002] vs
0.05 the upper bound is 25× below — decisive by a wide margin.

This mirrors the significance convention the other gates use (an interval that
excludes the null), applied here as "the interval excludes the threshold on the
losing side."

### D2. The gate resolution, not the thresholds, changes

`report.py`'s Phase 6 branch computes the detection interval and compares its
upper bound to the threshold:

- upper bound < threshold → **fail**, detail names the arm and the miss
  ("detection 0.001 [0.000, 0.002] wholly below 0.05; conjunction cannot hold,
  harm arm not required"). This is the closed-without-implementation verdict.
- otherwise, if the harm arm is unmeasured → **insufficient data** (unchanged
  for the case the rule is not meant to short-circuit).

The thresholds in `eval_thresholds.yaml` are untouched — changing them to force
a verdict is explicitly forbidden (ADR-017, task guard). Only the resolution
logic that was missing is added.

### D3. A larger run tightens the interval that the verdict rests on

The verdict hinges on the detection interval being decisively below 0.05. The
existing 1500-generation baseline already gives [0.000, 0.002], but a phase-
closing decision that goes in the permanent record deserves a tighter interval.
Run the protocol at increased volume for the C0 baseline the gate reads. A
cheap dedicated harness (generate N replies at the frozen C0 config, count
`has_token_cycle`) is preferred over re-running the full matrix — the seeded
C3/C4 arms are slow and irrelevant to this gate.

*ponytail:* reuse `has_token_cycle` and the C0 config resolution the eval
already has; do not build a second cycle detector.

## Risks / Trade-offs

- **Closing a phase on "rare, therefore not worth it"** → the risk is that
  cycles, though rare, are severe when they occur. Mitigated by the roadmap's
  own framing (cycle ≠ harm; "да да да" is a legitimate Pepe reply) and by the
  margin: even if every one of the ~0.1% cyclic replies were maximally harmful,
  anti-cycle machinery would address 0.1% of output at the cost of new jump
  logic on the hot path. The gate encodes exactly this trade.
- **Reversibility** → nothing is built or removed, so there is nothing to roll
  back. If a future corpus made cycles frequent, the same gate would reopen on
  the next run; the verdict is recorded as "closed at these numbers", not
  "impossible".

## Migration Plan

None — no schema, no runtime, no feature. A `report.py` logic change plus a
recorded decision.

## Open Questions

- None. The manual harm round is deliberately not run (D2 rationale); if the
  owner later wants it regardless, it is a separate, optional follow-up that
  cannot change this verdict.
