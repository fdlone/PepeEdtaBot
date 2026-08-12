# Tasks: markov2r-phase6-anticycle-verdict

Phase 6 is gated-to-implement. This change renders the gate verdict and closes
the phase — it builds no feature (M2R-600/610 are NOT implemented).

## 1. Gate resolution (generation-eval spec)

- [x] 1.1 `tools/eval/report.py`: the Phase 6 branch resolves to **fail/close** when the detection-rate CI upper bound is below `cycle_detection_rate_min` (design D1/D2); otherwise, with the harm arm unmeasured, it stays `insufficient data`. Thresholds untouched
- [x] 1.2 Detail line names the failing arm and the miss (rate, interval, threshold) so the closed verdict is auditable from its numbers
- [x] 1.3 Test: a synthetic run with detection CI wholly below the bar renders fail/close; a run with detection near/above the bar and no harm data stays insufficient

## 2. Decisive measurement

- [x] 2.1 Measure `cycle_detection_rate` at increased volume on the frozen C0 config (reuse `has_token_cycle` + the eval's C0 resolution; do not build a second detector) — record the rate and its CI — **9000 gens, 1 cyclic reply, 0.0001 [0.0000, 0.0003]**
- [x] 2.2 Confirm the interval is decisively below 0.05 (whole CI under the bar); if — unexpectedly — it is not, STOP and report (the verdict would change) — CI upper bound 0.0003, **167× below** the bar; decisive

## 3. Verdict report

- [x] 3.1 Protocol/verdict run producing the Phase 6 gate row; report in `docs/eval_reports/`, referenced from this change — `eval_2026-08-12_phase6-verdict.md` (gate-row rendering proven by `TestPhase6Gate`)
- [x] 3.2 Report shows: detection rate + CI vs 0.05, the automatic harm component, and that the manual harm round was not required (AND-gate, detection arm decisively fails)

## 4. Record the decision

- [x] 4.1 `docs/v2/00_STATUS.md`: Phase 6 row → **closed without implementation**, with the numbers and a reference to the report
- [x] 4.2 Note in the roadmap/ADR trail that M2R-600/610 stay unbuilt and why (cycles rare: whole CI 167× below the detection bar), so it is not rediscovered as an idea — ADR-015 trail
- [x] 4.3 Never adjust a pre-registered threshold to change the verdict (ADR-017) — thresholds untouched

## 5. Close-out

- [x] 5.1 `openspec validate --strict` green for this change
- [x] 5.2 Full test suite + lint/type checks; CI smoke green — 1122 tests OK, ruff/mypy clean, smoke ok
- [x] 5.3 Archive this change after merge
