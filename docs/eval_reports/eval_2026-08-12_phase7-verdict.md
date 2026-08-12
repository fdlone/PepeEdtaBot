# Phase 7 verdict — order-4 closed without implementation

**Date:** 2026-08-12
**Gate:** `phase7_order4` in `tools/eval/eval_thresholds.yaml`
(`order4_selected_share_min = 0.10`, `exact_copy_delta_max = 0.0`), gated per
ADR-002, needs ≥ 1000 shadow-eligible steps for a verdict.
**Data:** C0 frozen config on `db_prod_copy`, **9000 generations** (3000 ×
seeds 42/1337/2026), prompt set `308b7deaea0f`. Shadow order-4 selector
(M2R-020, measurement-only, window estimator); no order-4 index was built or
read.

## Verdict

**Phase 7 closes without implementation. M2R-700 (order-4 learning + index) and
M2R-710 (variable 4→3→2 selector) are NOT built.** No threshold was changed.

The order-4 index is the heaviest structure in the model (ADR-002), so it is
built only if the shadow selector shows order-4 would actually be chosen. It is
never chosen.

| Seed | shadow-eligible steps | order-4 would be selected |
|---|---|---|
| 42 | 1816 | 0 |
| 1337 | 2124 | 0 |
| 2026 | 1997 | 0 |
| **total** | **5937** | **0** |

`order4_selected_share = 0.0%` over 5937 eligible steps — the sample bar (1000)
is cleared 6×, and the numerator is exactly zero, not a borderline miss below
the 10% threshold. Order-4 is not "rarely" the better continuation on this
corpus — it is never the shadow selector's choice.

The exact-copy sub-condition (`exact_copy_delta_max`) is moot: it compares an
order-4 arm against the baseline, and with 0% selection there is no such arm to
measure. The selected-share dimension alone closes the phase.

## Why zero

The same structural wall behind the Phase 2/3 verdicts: ≈99% of order-3 states
already have a single continuation. Where order-3 is already deterministic, the
order-4 projection cannot disagree with it; where it could, the exact order-4
window is a verbatim replay the selector does not prefer over the recombining
order-3 walk. So the heaviest index would add storage and a 4→3→2 selector on
the hot path to reproduce, at best, what order-3 already produces — exactly the
"no proven benefit" ADR-002 guarded against.

## Reversibility

Nothing is built or removed, so there is nothing to roll back. ADR-002 stays
"Accepted, реализация gated"; the trail records that the gate was evaluated and
failed at 5937 eligible / 0 selected. If a future corpus made order-4
selectable, the same shadow gate reopens on the next run. The gate-rendering
behaviour (fail at a sufficient sample with a below-threshold share) is locked
by `TestPhase7Gate` in `tests/test_eval_protocol.py`.
