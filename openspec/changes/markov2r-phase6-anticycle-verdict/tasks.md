# Tasks: markov2r-phase6-anticycle-verdict

Phase 6 is gated-to-implement. This change renders the gate verdict and closes
the phase — it builds no feature (M2R-600/610 are NOT implemented).

## 1. Gate resolution (generation-eval spec)

- [ ] 1.1 `tools/eval/report.py`: the Phase 6 branch resolves to **fail/close** when the detection-rate CI upper bound is below `cycle_detection_rate_min` (design D1/D2); otherwise, with the harm arm unmeasured, it stays `insufficient data`. Thresholds untouched
- [ ] 1.2 Detail line names the failing arm and the miss (rate, interval, threshold) so the closed verdict is auditable from its numbers
- [ ] 1.3 Test: a synthetic run with detection CI wholly below the bar renders fail/close; a run with detection near/above the bar and no harm data stays insufficient

## 2. Decisive measurement

- [ ] 2.1 Measure `cycle_detection_rate` at increased volume on the frozen C0 config (reuse `has_token_cycle` + the eval's C0 resolution; do not build a second detector) — record the rate and its CI
- [ ] 2.2 Confirm the interval is decisively below 0.05 (whole CI under the bar); if — unexpectedly — it is not, STOP and report (the verdict would change)

## 3. Verdict report

- [ ] 3.1 Protocol/verdict run producing the Phase 6 gate row; report in `docs/eval_reports/`, referenced from this change
- [ ] 3.2 Report shows: detection rate + CI vs 0.05, the automatic harm component, and that the manual harm round was not required (AND-gate, detection arm decisively fails)

## 4. Record the decision

- [ ] 4.1 `docs/v2/00_STATUS.md`: Phase 6 row → **closed without implementation**, with the numbers and a reference to the report
- [ ] 4.2 Note in the roadmap/ADR trail that M2R-600/610 stay unbuilt and why (cycles rare: 50× below the detection bar), so it is not rediscovered as an idea
- [ ] 4.3 Never adjust a pre-registered threshold to change the verdict (ADR-017)

## 5. Close-out

- [ ] 5.1 `openspec validate --strict` green for this change
- [ ] 5.2 Full test suite + lint/type checks; CI smoke green
- [ ] 5.3 Archive this change after merge
