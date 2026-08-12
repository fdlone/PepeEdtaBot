# Tasks: markov2r-phase3-temporal-layer-core

Phase 3 core (M2R-200/210). Calibration (M2R-215) and GC (M2R-220) are separate
changes — see proposal.md.

## 1. Pre-registration (must land before any Phase 3 number is produced)

- [x] 1.1 Add the `phase3_temporal` block to `tools/eval/eval_thresholds.yaml` with the numbers and written rationale from design D9 (`freshness_delta_min`, `historical_meme_delta_floor`, `exact_copy_delta_max`, `affinity_without_copy_delta_floor`; latency reuses `performance`) — separate commit, before any grid runs
- [x] 1.2 Gate evaluation in the runner: verdict computed strictly from the file, printed with the numbers that produced it, `insufficient data` when an arm is missing

## 2. Migration (M2R-200)

- [x] 2.1 `app/migrations/018_temporal_layer.sql`: `first_seen`, `last_seen` (INTEGER unix seconds), `s_value` (REAL, default 0), `s_updated_at` (INTEGER) on `starts`, `starts3`, `transitions`, `transitions3` — NULL defaults, no data rewrite (design D6)
- [x] 2.2 Migration runs on a clean DB and on `db_prod_copy`; record the measured wall time in the change (audit §8 asks for the number, not an assurance) — **measured: 4.0 ms on the prod copy (30 720 transitions, 27 237 transitions3), 58 ms for the whole chain on an empty DB**; metadata-only as predicted, no row rewritten
- [x] 2.3 Readers tolerate NULL: NULL `first_seen` means "predates the temporal record", `s_value = 0` with NULL `s_updated_at` means an empty short layer
- [x] 2.4 `/clear confirm` leaves no temporal remnant — extend the existing orphaned-structures test rather than writing a new one

## 3. Decay arithmetic (M2R-210, core)

- [x] 3.1 One pure helper for both directions: observe (`s_value·2^(−Δt/hl) + 1`) and read (`s_eff = s_value·2^(−Δt/hl)`), taking `(s_value, s_updated_at, now, half_life)` — no second implementation in SQL (design D1)
- [x] 3.2 Guard the degenerate inputs explicitly: NULL/absent `s_updated_at`, `now` earlier than `s_updated_at` (clock skew), and a zero `s_value`
- [x] 3.3 Verify locally that no chain row has `cnt <= 0` on `db_prod_copy`, so the sampler's clamp change in 5.1 is provably behavior-preserving (design D4)

## 4. Learn path

- [x] 4.1 Extend the existing single learn transaction: read the touched rows' short pairs, compute the new values in Python, write both layers plus `first_seen`/`last_seen` atomically (design D1)
- [x] 4.2 `first_seen` is set once and never moved; `last_seen` advances on every observation
- [ ] 4.3 Measure the learn path's added cost on `db_prod_copy` and record the number — one extra indexed SELECT per message is the claim, not the finding
- [x] 4.4 Phase 1's incremental cache fold carries the temporal pair, so a cached pool and a freshly read one agree

## 5. Blend and sampling (M2R-210, core)

- [x] 5.1 Sampler takes float weights: `max(cnt, 1)` becomes `max(w, EPS)`; integer inputs keep producing identical results (design D4)
- [x] 5.2 `blend_pool(pool, alpha, now, ...)` returns its input unchanged at α = 0 — the early return that makes neutrality structural rather than incidental
- [x] 5.3 Blend over the union of tokens, long layer compressed sublinearly (`log` | `pow` with β) before normalization; an empty layer degenerates to the other
- [ ] 5.4 α resolved from the chat's mood; the value the walk actually used is what gets reported (not the configured one)
- [ ] 5.5 `now` captured once per generation and threaded into blending and cache reads; nothing below reads the clock (design D3)
- [x] 5.6 `pool_diagnostics` computed from the blended weights, so entropy describes the sampled distribution (design D5) — note the knock-on: this is Phase 2's input

## 6. Knobs

- [ ] 6.1 Registry entries + `Settings`/`RuntimeState` + `.env.example` (established 5-step pattern; registry drift tests stay green): `markov_short_half_life_days`, `markov_long_compression`, `markov_long_compression_beta`, `markov_alpha_sleepy`, `markov_alpha_calm`, `markov_alpha_lively`, `markov_alpha_heated`
- [ ] 6.2 All α default to 0 (neutral); bounds validated on both paths (env and `/set`) per `runtime-knob-validation`
- [ ] 6.3 Half-life change resets that chat's short layer with an explicit warning naming what was discarded and how long it takes to rebuild; setting the current value is a no-op; long layer untouched (design D7)
- [ ] 6.4 `/config full` surfaces the α profile, half-life and compression shape

## 7. Neutrality proof (the phase's hard contract)

- [x] 7.1 `python -m tools.generation_hash --db db_prod_copy/markov.db` — identical hash at default settings, against the frozen baseline (`5a72e2d4`)
- [ ] 7.2 Characterization tests green unchanged at neutral settings
- [ ] 7.3 Runtime revert check: α set back to 0 via `/set` restores baseline output without a restart
- [ ] 7.4 Neutrality holds with the migration applied to a database that has already learned messages under the temporal layer (schema present, short layer populated, α = 0)

## 8. Unit tests

- [x] 8.1 Decay: an observation now contributes 1; an observation one half-life old contributes 0.5; order of observations does not change the result; `s_eff` is non-increasing between observations (TZ §19)
- [x] 8.2 Compression: both shapes preserve the order of preference; a 10000-vs-20 count pair leaves the smaller non-negligible
- [x] 8.3 Blend: union coverage, token present in only one layer stays reachable, empty layer degenerates, α = 0 and α = 1 endpoints
- [ ] 8.4 Half-life reset: short layer emptied, long layer and timestamps untouched, no-op when the value is unchanged
- [ ] 8.5 Cache: a pool cached at t₀ and read at t₁ yields the same weights as an uncached read at t₁

## 9. Property / invariant tests (TZ §19)

- [x] 9.1 Blending any two valid distributions at any legal α yields a valid distribution: finite, non-negative, sums to 1
- [ ] 9.2 Determinism: same seed + same settings + same `now` ⇒ identical output, cached or not
- [x] 9.3 `s_eff` monotonically non-increasing between observations, for any legal half-life
- [ ] 9.4 Learning is atomic across both layers under a mid-transaction failure

## 10. Telemetry

- [ ] 10.1 Applied α, the short layer's step coverage and its mean mass contribution accumulated per generation; exposed in `GenerationTrace`
- [ ] 10.2 `gen_trace_log` line and `/stats`; aggregates only, no per-message timestamps, chat ids masked (existing `log-privacy` rules)
- [ ] 10.3 Neutral configuration reads as neutral, and "configured but inert" (α > 0 over an empty short layer) is distinguishable from "not configured" (spec scenario)

## 11. Temporal eval fixture (owner decision 2026-08-12)

- [ ] 11.1 `tools/eval/temporal_fixture.py`: replay a snapshot's retained messages in timestamp order through the real learning arithmetic into a separate database; source opened read-only (design D8)
- [ ] 11.2 Deterministic: two builds from the same source and parameters produce byte-identical metric results
- [ ] 11.3 Report provenance: snapshot labelled as reconstructed, with its time span and fresh-slice size printed
- [ ] 11.4 `freshness_reflection` implemented per doc 05 §3 against the fresh slice; still `insufficient data` on snapshots without observation times
- [ ] 11.5 Historical-meme check draws its n-gram list from the full snapshot's `chat_verbatim_ngrams`, not from the fixture's own old rows (design D8)

## 12. Eval — ablation and grid

- [ ] 12.1 `tools/eval/matrix.yaml`: C2 arm (`available: true` only for what this change ships); C5 stays unavailable until it has both features to combine
- [ ] 12.2 Grid on the fixture: α ∈ {0, 0.3, 0.5, 0.7} × compression ∈ {log, pow β=0.5, pow β=0.75}, protocol volumes (500 generations × seeds 42/1337/2026), bootstrap CIs
- [ ] 12.3 **All deltas computed within the fixture** (C0_fixture vs arm) — the frozen C0 baseline is a different corpus and is not a valid comparand (design D8)
- [ ] 12.4 Latency p50/p95 per arm against the 150 ms budget; report the blend's per-step cost separately from the total
- [ ] 12.5 Re-run Phase 2's entropy histogram (7034-step protocol) on the blended pools and put it beside the Phase 2 numbers — this is the evidence the Phase 2 keep-or-delete decision rests on (design D5)
- [ ] 12.6 Report in `docs/eval_reports/`, referenced from this change: per-arm table with CIs, gate verdict, `distinct_basis_tokens` denominators, fixture provenance, entropy comparison
- [ ] 12.7 CI smoke stays green on `snapshot_synthetic`

## 13. Default decision (gate)

- [ ] 13.1 Gate passed ⇒ propose the α profile as a defaults change, explicitly labelled as measured on a reconstructed snapshot and therefore provisional until live re-measurement in M2R-215
- [ ] 13.2 Gate fails ⇒ defaults stay neutral; record the negative result with numbers in `docs/v2/00_STATUS.md` and reference the report
- [ ] 13.3 Never adjust a pre-registered threshold to make the gate pass; if a threshold turns out wrong, that is a separate commit with justification and owner approval

## 14. Phase 2 decision (what this phase was asked to unblock)

- [ ] 14.1 Present the entropy comparison from 12.5 to the owner with a recommendation: keep the disabled Phase 2 code for re-measurement, or delete it as dead weight
- [ ] 14.2 Record the owner's decision in `docs/v2/00_STATUS.md`; if the answer is "re-measure", note that it requires a **new** pre-registered gate — `phase2_entropy` is spent

## 15. Documentation

- [ ] 15.1 `docs/v2/00_STATUS.md`: Phase 3 row + "next session" section pointed at M2R-215 calibration
- [ ] 15.2 `.env.example` entries with bounds and one-line meanings; `GENERATION_PIPELINE.md` / `ARCHITECTURE.md` updated where sampling and learning are described
- [ ] 15.3 Record in doc 05 that `snapshot_temporal` now has a build path, and that the v1.0.1 amendment's "physically impossible" clause applies only to retrospective live history

## 16. Housekeeping riding along (no functional content)

- [ ] 16.1 `openspec archive markov2r-phase2-entropy-sampling` — its own commit, owner decision 2026-08-12 (no PR just for an archive)

## 17. Close-out

- [ ] 17.1 `openspec validate --strict` green for this change
- [ ] 17.2 Full test suite + lint/type checks as configured in the project
- [ ] 17.3 Archive this change after merge — rides in the next phase's PR, same convention as 16.1
