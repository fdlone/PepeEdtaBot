# Proposal: markov2r-phase6-anticycle-verdict

## Why

Phase 6 (anti-cycle + jumps) is a gated phase: its detector and cycle-penalty
(M2R-600/610) are built **only if** the two-dimensional gate opens —
`cycle_detection_rate ≥ 0.05` **and** `cycle_harm_rate ≥ 0.02` (ADR-015,
roadmap Phase 6: "непревышение любого порога ⇒ фаза закрывается без реализации
с фиксацией цифр"). The measured detection rate is **0.001 [0.000, 0.002]** —
50× below the bar, CI upper bound 25× below. A two-dimensional AND-gate with one
arm decisively below threshold cannot open, so the manual harm round cannot
rescue it and need not be run. This change produces that verdict with numbers
and closes Phase 6 without implementing anything.

## What Changes

- **No feature code.** M2R-600/610 (cycle detector, cycle penalty + dynamic
  jump) are NOT built — the gate is closed, and building against a closed gate
  would violate the project's own methodology (ADR-010/015/017).
- **Gate resolution fix** (`tools/eval/report.py`): the Phase 6 gate currently
  always reports `insufficient data`, waiting on the manual harm round. Per the
  pre-registered rule, a two-dimensional AND-gate whose detection arm is
  *significantly* below its threshold resolves to a **fail/close** verdict —
  the same "a demonstrated miss outranks a missing part" logic the other gates
  already use. The thresholds themselves are untouched.
- **Dedicated verdict run**: a protocol run at increased volume to tighten the
  detection-rate CI, so "cycles are rare" rests on a decisive interval, not one
  borderline sample. Report in `docs/eval_reports/`.
- **Recorded decision**: Phase 6 closed without implementation, numbers in
  `docs/v2/00_STATUS.md`, referencing the report.

## Capabilities

### New Capabilities

<!-- none -->

### Modified Capabilities

- `generation-eval`: the Phase 6 rate×harm gate resolves to a verdict when its
  detection arm is decisively below threshold, instead of deferring forever to
  the manual harm round.

## Impact

- **Code**: `tools/eval/report.py` gate-resolution logic only. No app code, no
  schema, no runtime behavior change — generation is untouched.
- **Risk**: low. The verdict rests on a metric 50× below its bar; a larger run
  only tightens the interval. The decision is reversible in principle — if a
  future model made cycles frequent, the gate could reopen — but nothing is
  built or removed now, so there is nothing to roll back.
- **Methodology**: this is the same negative-verdict pattern as Phases 2 and 3,
  and cleaner: a gated-to-implement phase closes with the detector never built.
