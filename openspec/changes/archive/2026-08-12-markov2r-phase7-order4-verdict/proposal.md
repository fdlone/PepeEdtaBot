# Proposal: markov2r-phase7-order4-verdict

## Why

Phase 7 (order-4 chain) is gated (ADR-002): the order-4 index — the heaviest
structure in the model — is built **only if** the Phase 1 shadow selector shows
order-4 would actually be chosen, without raising exact-copy. The shadow
telemetry has now accumulated a decisive sample: over **5937 shadow-eligible
steps**, order-4 would have been selected **0 times** (`order4_selected_share =
0.0%`, threshold 0.10). The gate's only remaining gap was sample size (a prior
run had 897 < 1000 eligible steps); at 5937 it is decisively met, and the
verdict is a clean fail. This change records that verdict and closes Phase 7
without implementation.

## What Changes

- **No feature code.** M2R-700 (order-4 learning + index) and M2R-710 (variable
  4→3→2 selector) are NOT built — the shadow gate fails, and building against a
  failed gate would violate the project's methodology (ADR-002/010/017).
- **Gate verdict rendered**: the Phase 7 gate in `tools/eval/report.py` already
  resolves to pass/fail once ≥1000 shadow-eligible steps exist; at 5937 eligible
  with 0% selected it renders **fail**. No gate-logic change is needed (unlike
  Phase 6) — only a test locking the fail/close rendering at volume.
- **Decisive measurement**: shadow order-4 telemetry aggregated over the three
  protocol seeds at increased volume (5937 eligible, 0 selected). Report in
  `docs/eval_reports/`.
- **Recorded decision**: Phase 7 closed without implementation, numbers in
  `docs/v2/00_STATUS.md` and the ADR-002 trail, so the order-4 index is not
  rediscovered as an idea.

## Capabilities

### New Capabilities

<!-- none -->

### Modified Capabilities

- `generation-eval`: a lock (test) on the Phase 7 shadow gate rendering fail
  when the eligible sample clears its bar and the selected share is below
  threshold — the verdict that closes the phase.

## Impact

- **Code**: a test in `tests/test_eval_protocol.py`; no app code, no schema, no
  runtime change — generation is untouched, and the shadow selector is
  measurement-only by construction (its knob does not affect output).
- **Risk**: low. The verdict rests on 0 selections in 5937 eligible steps — not
  a borderline miss. Reversible in principle: if a future corpus made order-4
  selectable, the same shadow gate would reopen on the next run; nothing is
  built or removed now.
- **Methodology**: confirms ADR-002's caution (the order-4 index is the heaviest
  structure and is not created without proven benefit) with numbers. Same
  negative-verdict pattern as Phase 6.
