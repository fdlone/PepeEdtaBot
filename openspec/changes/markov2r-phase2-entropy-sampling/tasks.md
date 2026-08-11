# Tasks: markov2r-phase2-entropy-sampling

No schema change and no migration in this phase (TZ §6 is sampling-only).

## 1. Pre-registration (must land before any Phase 2 number is produced)

- [x] 1.1 Add the `phase2_entropy` block to `tools/eval/eval_thresholds.yaml` with the numbers and the written rationale from design D8 (`exact_copy_delta_max`, `distinct2_delta_min`, `distinct3_delta_min`, `affinity_without_copy_delta_floor`; latency reuses the existing `performance` section) — separate commit, before the grid runs
- [x] 1.2 Gate evaluation in the runner: verdict computed strictly from the file, printed with the numbers that produced it, `insufficient data` when an arm is missing

## 2. Knobs

- [x] 2.1 Registry entries + `Settings`/`RuntimeState` fields + `.env.example` (established 5-step pattern; registry drift tests must stay green): `markov_entropy_temp_gain`, `markov_entropy_pivot`, `markov_entropy_temp_min`, `markov_entropy_temp_max`, `markov_branching_degenerate_max`, `markov_branching_candidate_floor` — TZ §18's `*_ENABLED` booleans dropped, see design D9
- [x] 2.2 All defaults neutral at merge time: gain 0, branching-target disabled — bounds validated on both paths (env and `/set`), per `runtime-knob-validation`
- [x] 2.3 `/config full` surfaces the two decision knobs and the pivot (clamps stay out — they are a safety net, not a voice setting)

## 3. M2R-100 — Entropy → temperature (core)

- [x] 3.1 Pure helper: `T = T_base·(1 + GAIN·(H_norm − H_pivot))` clamped to `[T_min, T_max]`, expressed over the existing frequency power (`power = 1/T`), with the explicit early return at gain 0 / feature off (design D2)
- [x] 3.2 `_DiagnosticsAccumulator.note_pool` returns the diagnostics it already computes, so the walk can pass `normalized_entropy` into sampling without recomputing it
- [x] 3.3 Adjustment applied to the `power` argument at the call site, so it lands **before** `_roll_exploration` (which runs inside `weighted_next_choice`) — no new parameter on that function, and the three start-state call sites keep passing the unadjusted power (design D1)
- [x] 3.4 Wire the two walk-step call sites (order-3 and order-2) plus the settings path: `EntropySampling` built once per generation in `ResponseGenerator`, applied to the candidate walk and to the verbatim extension

## 4. M2R-110 — Branching-aware candidate target (core)

- [x] 4.1 Running mean of accepted candidates' `mean_branching` in the response generator; target recomputed after each accepted candidate, never below the floor, never above `CANDIDATE_TARGET`, attempt budget untouched (design D6)
- [x] 4.2 Disabled knob ⇒ target is exactly the previous constant
- [ ] 4.3 Record the second-order effect on slot mutation (`target` also caps mutations) so it is visible in the report, not discovered later

## 5. Unit tests

- [ ] 5.1 Temperature mapping: pivot behavior, both signs of gain, clamp at both ends, degenerate pool (H_norm = 0), no non-positive or NaN temperature for any legal knob combination
- [ ] 5.2 Sampling: flatter weights at high entropy with positive gain, sharper at low entropy; excluded tokens stay excluded (spec: entropy never overrides gates)
- [ ] 5.3 Branching target: degenerate chain stops early; wide chain reaches the full target; a chain with zero accepted candidates still uses the whole budget (no empty reply from an early stop)
- [ ] 5.4 Knob-off paths for both features hit the early return (no arithmetic, no RNG difference)

## 6. Property / invariant tests (TZ §19)

- [ ] 6.1 For any pool and any legal knob values: weights stay ≥ 0, finite, and at least one candidate keeps positive weight
- [ ] 6.2 Determinism: same seed + same settings ⇒ identical output across runs
- [ ] 6.3 Monotonicity of the mapping in `H_norm` for a fixed sign of gain (the direction the design claims is the direction the code produces)

## 7. Neutrality proof (the phase's hard contract)

- [ ] 7.1 `python -m tools.generation_hash --db db_prod_copy/markov.db` — identical hash with the feature off, with gain 0, and against the frozen baseline
- [ ] 7.2 Characterization tests (`test_markov_generation_characterization.py`) green unchanged at neutral settings
- [ ] 7.3 Runtime revert check: gain set back to 0 via `/set` restores baseline output without a restart

## 8. Telemetry and trace

- [ ] 8.1 Mean applied temperature accumulated per generation alongside the entropy it came from; exposed in `GenerationTrace`
- [ ] 8.2 `gen_trace_log` line and `/stats` reporting; numbers only, chat ids masked (existing `log-privacy` rules)
- [ ] 8.3 Neutral configuration reads as neutral in the numbers (spec scenario)

## 9. Eval — ablation and calibration (doc 05)

- [ ] 9.1 `tools/eval/matrix.yaml`: `C1`, `C1a`, `C1b`, `C1flat` per design D7; `available: true` only for what this phase actually ships
- [ ] 9.2 Read `H_pivot` from the measured mean normalized entropy on the eval corpus and record the number in the report (design D3) — not a hand-picked value
- [ ] 9.3 Calibration grid on `db_prod_copy`: gain ∈ {−0.6, −0.3, 0, +0.3, +0.6}, protocol volumes (500 generations × seeds 42/1337/2026), bootstrap CIs
- [ ] 9.4 Flat-temperature control arm matched to the winning entropy arm's mean applied temperature (design D5)
- [ ] 9.5 Latency: p50/p95 for every arm against the 150 ms budget, both features separately (M2R-110 is expected to reduce attempts — verify rather than assume)
- [ ] 9.6 Report in `docs/eval_reports/`, referenced from this change: per-arm table with CIs, the gate verdict, the `distinct_basis_tokens` denominators, and the slot-mutation side effect from 4.3
- [ ] 9.7 CI smoke stays green on `snapshot_synthetic`

## 10. Default decision (gate)

- [ ] 10.1 Gate passes ⇒ raise `markov_entropy_temp_gain` (and the branching knob, if its own arm passed) to the calibrated defaults in this change, citing the report
- [ ] 10.2 Gate fails ⇒ defaults stay neutral; record the negative result with numbers in `docs/v2/00_STATUS.md` and reference the report — a closed phase with a documented number is a valid outcome
- [ ] 10.3 Never adjust a pre-registered threshold to make the gate pass; if a threshold turns out wrong, that is a separate commit with justification and owner approval

## 11. Documentation

- [ ] 11.1 `docs/v2/00_STATUS.md`: Phase 2 row + "next session" section pointed at Phase 3
- [ ] 11.2 `.env.example` entries with bounds and one-line meanings; `GENERATION_PIPELINE.md` / `ARCHITECTURE.md` updated where the sampling step is described
- [ ] 11.3 ADR open question #2 ("which entropy → temperature mapping works better") answered in the report and marked as answered in the change

## 12. Housekeeping riding along (no functional content)

- [ ] 12.1 `openspec archive report-pivo-mentions-to-owner` (all 16 tasks complete) — included here only to avoid a PR of its own; keep it in its own commit

## 13. Close-out

- [ ] 13.1 `openspec validate --strict` green for this change
- [ ] 13.2 Full test suite + lint/type checks as configured in the project
- [ ] 13.3 Archive this change after merge
