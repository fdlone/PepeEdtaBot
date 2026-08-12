# Tasks: markov2r-phase7-order4-verdict

Phase 7 is gated-to-implement (ADR-002). This change renders the shadow-gate
verdict and closes the phase — it builds no feature (M2R-700/710 are NOT
implemented). The gate logic in `report.py` already resolves correctly; this
change adds a test lock, the measurement, and the recorded decision.

## 1. Decisive measurement

- [x] 1.1 Aggregate shadow order-4 telemetry over the three protocol seeds on C0 at increased volume (reuse `run_config_seed` + the eval's shadow selector; no second selector) — record eligible steps and selected share — **5937 eligible (1816/2124/1997), 0 selected**
- [x] 1.2 Confirm eligible ≥ 1000 (the gate's sample bar) and the selected share is below the 0.10 threshold; if — unexpectedly — order-4 is selected at ≥ 10%, STOP and report (the verdict would change) — **5937 ≥ 1000, selected share 0.0%**; decisive fail

## 2. Gate rendering lock (generation-eval spec)

- [x] 2.1 `TestPhase7Gate` in `tests/test_eval_protocol.py`: a synthetic run with ≥ 1000 shadow-eligible steps and 0% selected renders **fail**; a run below 1000 eligible stays `insufficient data`. (No `report.py` change — the logic already resolves; the test locks it)

## 3. Verdict report

- [x] 3.1 Verdict report in `docs/eval_reports/`, referenced from this change: eligible steps, per-seed breakdown, selected share vs 0.10, and why zero (structural single-continuation wall, ADR-002's "heaviest index")
- [x] 3.2 Note the exact-copy sub-condition is moot at 0% selection (no order-4 arm to measure copy against)

## 4. Record the decision

- [x] 4.1 `docs/v2/00_STATUS.md`: Phase 7 row → **closed without implementation**, with the numbers and a reference to the report
- [x] 4.2 ADR-002 trail: order-4 index (M2R-700/710) stays unbuilt, gate evaluated and failed at 5937 eligible / 0 selected, so it is not rediscovered
- [x] 4.3 Never adjust a pre-registered threshold to change the verdict (ADR-017)

## 5. Close-out

- [x] 5.1 `openspec validate --strict` green for this change
- [x] 5.2 Full test suite + lint/type checks; CI smoke green — 1124 tests OK, ruff/mypy clean, smoke ok
- [x] 5.3 Archive this change after merge
