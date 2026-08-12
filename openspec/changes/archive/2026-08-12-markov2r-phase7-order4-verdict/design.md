# Design: markov2r-phase7-order4-verdict

## Context

See proposal.md — Why. Design-relevant state:

- Phase 7 gate (ADR-002): build the order-4 index only if the shadow selector
  shows order-4 would be chosen ≥ `order4_selected_share_min` (0.10) without a
  significant exact-copy rise. `eval_thresholds.yaml` also carries
  `exact_copy_delta_max = 0.0`.
- The shadow order-4 selector (M2R-020, Phase 1) is measurement-only: at each
  eligible walk step it records whether order-4 *would* have been selected, over
  the retained-message window estimator. `markov_shadow_order4_enabled` defaults
  true; the knob does not change generated output.
- `tools/eval/report.py` already renders the Phase 7 gate: `insufficient data`
  while `shadow_order4_eligible < 1000`, else pass/fail from the mean per-seed
  `shadow_order4_selected_share` vs the threshold. **This logic is already
  correct** — unlike Phase 6, no gate-resolution change is required.
- The blocker was sample size: a prior protocol run had 897 eligible steps
  (< 1000). Aggregating C0 over the three protocol seeds at increased volume
  now yields **5937 eligible, 0 selected**.

## Goals / Non-Goals

**Goals:**

- A decisive shadow sample (≥1000 eligible) confirming order-4 is never chosen.
- The gate rendering fail/close from that sample, locked by a test.
- The verdict recorded with numbers; the order-4 index never built.

**Non-Goals:**

- M2R-700/710 (order-4 learning, index, 4→3→2 selector). The gate fails; they
  are not built.
- Any gate-logic change in `report.py` — it already resolves correctly at a
  sufficient sample.
- The exact-copy sub-condition: it only matters if order-4 *were* built (there
  would be an order-4 arm to measure copy against). With 0% selected there is no
  such arm and the selected-share dimension alone closes the phase.

## Decisions

### D1. The verdict rests on selections, not just the point share

Order-4 was selected **0 times in 5937 eligible steps** across the three
protocol seeds (1816 / 2124 / 1997 eligible per seed, 0 selected each). This is
stronger than a share point estimate below 0.10: the numerator is exactly zero
at 6× the gate's sample bar. Order-4 is not "rarely" the better continuation on
this corpus — it is never the shadow selector's choice.

*Why zero:* order-4 eligibility requires an order-4 state with enough support to
project a selection, but the same structural wall behind the Phase 2/3 verdicts
(≈99% of order-3 states already have a single continuation) means the order-4
projection almost never disagrees with order-3, and when it could, the exact-4
window is a verbatim replay the selector does not prefer. This is exactly the
"heaviest index, no proven benefit" ADR-002 warned about.

### D2. No gate-logic change — only a test that locks the rendering

`report.py`'s Phase 7 branch already returns fail when eligible ≥ 1000 and the
selected share is below threshold. The change adds a `TestPhase7Gate` mirroring
Phase 6's: a synthetic run with ≥1000 eligible and 0% selected renders fail; a
run with < 1000 eligible stays insufficient. This locks the behavior the
verdict depends on without touching working code.

*ponytail:* the measurement reuses the eval's existing shadow telemetry and C0
resolution (`run_config_seed`); no second selector, no new metric.

### D3. The verdict is recorded as "closed at these numbers", not "impossible"

Like Phase 6, nothing is built or removed, so there is nothing to roll back. If
a future corpus made order-4 selectable, the same shadow gate reopens on the
next run. ADR-002 stays "Accepted, реализация gated"; the trail records that the
gate was evaluated and failed at 5937 eligible / 0 selected.

## Risks / Trade-offs

- **Closing on a shapshot's shadow projection** → the estimator is over the
  retained-message window (a conservative lower bound on order-4 support). If it
  understated order-4, the true share could be higher — but the reading is 0,
  and the window estimator is documented as conservative, so a correction moves
  away from, not toward, the 10% bar.
- **Reversibility** → none needed; nothing built. Reopens on new data.

## Migration Plan

None — no schema, no runtime, no feature. A test plus a recorded decision.

## Open Questions

- None. The exact-copy sub-condition is moot with 0% selection (D-Non-Goals).
