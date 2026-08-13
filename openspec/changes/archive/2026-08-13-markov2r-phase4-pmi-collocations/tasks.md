# Tasks: markov2r-phase4-pmi-collocations

Phase 4 (M2R-300/310/320). The manual rating round is an operational dependency
on human raters — start lining it up at task 1.1, not at task 9.

## 1. Operational precondition (start first, it has a human in the loop)

- [x] 1.1 Agree with the owner who rates the top-20 and when (owner alone, or owner + chat participants — doc 05 §5 publishes inter-rater agreement only when there is more than one rater)
- [x] 1.2 Pre-register the `phase4_memes` gate in `tools/eval/eval_thresholds.yaml` with the ≥70% bar and the automatic must-not-worsen conditions — separate commit, before any ranking is produced

## 2. Schema

- [x] 2.1 Migration `019_markov_collocations.sql`: `chat_id, left_token, right_token, joint_count, pmi, status, updated_at`, keyed per chat and pair (TZ §13)
- [x] 2.2 Measure the migration on `db_prod_copy` and record the number — ~1.8 ms (init with 019: 4.3 ms vs no-op init 2.5 ms; one empty WITHOUT ROWID table)
- [x] 2.3 `/clear confirm` removes the chat's registry — extend the existing orphaned-structures test
- [x] 2.4 Repository for the registry following the established `BaseRepo` pattern

## 3. Analyzer (M2R-300)

- [x] 3.1 Association measures over the chat's bigrams: normalized PMI, lift, LLR
- [x] 3.2 `meme_score = normalized_pmi × support_factor × recency_factor`; support and recency factors are mandatory inputs, not optional
- [x] 3.3 **Support threshold applied in the SQL `HAVING`, not after loading** (design D1) — the measured difference is 41 ms against 88 ms, and 88% of pairs are excluded as single occurrences
- [x] 3.4 Marginals read as two aggregate queries rather than derived from the filtered pairs — the filter changes which pairs are scored, never what the probabilities mean
- [x] 3.5 Registry write: promote by score, cap at `MARKOV_COLLOCATION_MAX_ENTRIES`, never resurrect a retired entry (design D3)

## 4. Daily maintenance (design D2)

- [x] 4.1 Analyzer pass joins `run_due_maintenance` beside the flavor decays, inheriting its cadence, retry interval and alert path
- [x] 4.2 A failing pass leaves the previous registry usable and does not break message handling
- [x] 4.3 Record duration and scored-pair count per pass in telemetry
- [x] 4.4 Measure the pass on `db_prod_copy` inside the real maintenance path and record the number against the 41 ms estimate — 37.7 ms total for all 4 chats via `_run_meme_analysis` (mean 9.3 ms/pass, 1224 pairs scored), under the 41 ms single-chat estimate

## 5. Collocation scoring (M2R-320, ADR-016)

- [x] 5.1 Bonus for a candidate containing an active collocation as an adjacent pair
- [x] 5.2 Penalty for breaking one **only when the chain held a transition to the right token from that state** (design D4)
- [x] 5.3 Availability answered from the pool the walk already loaded — no query per candidate per collocation
- [x] 5.4 Active collocations read once per reply, not once per candidate
- [x] 5.5 Count applied bonuses, applied penalties and **withheld** penalties separately (the withheld count is the evidence that 5.2's guard earns its place)
- [x] 5.6 Tokenization untouched: a test asserts learning and generation tokenize identically with collocations active

## 6. Hot n-grams (M2R-310)

- [x] 6.1 Meme-score ordering added to `get_hot` behind a knob; the existing frequency path stays the default (design D5)
- [x] 6.2 Both paths runnable side by side so the ablation can compare them
- [x] 6.3 Latency of the new ordering measured against the existing one — `get_hot` runs on almost every reply, and the correlated-subquery rewrite exists because a previous version cost ~41 ms per call. Measured on the prod copy: frequency 0.14 ms, meme ordering 0.18 ms mean (+0.04 ms; both far below the old 41 ms)

## 7. Knobs

- [x] 7.1 Registry entries + `Settings`/`RuntimeState` + `.env.example` (established 5-step pattern): `markov_meme_min_joint_count`, `markov_meme_min_support`, `markov_meme_recency_factor`, `markov_collocation_max_entries`, `markov_collocation_bonus`, `markov_collocation_break_penalty`, plus the hot-ngram ordering switch
- [x] 7.2 Bonus, penalty and meme-aware ordering default to neutral; bounds validated on both paths
- [x] 7.3 `markov_meme_min_joint_count` has a lower bound above the value at which the analysis stops meaning anything (spec: `runtime-knob-validation`)
- [x] 7.4 `/config full` surfaces the scoring knobs; `/stats` surfaces the registry size by status

## 8. Tests

- [x] 8.1 Association measures: a frequent pair of frequent tokens scores below a rare pair that always co-occurs; a pair below the support threshold is never scored
- [x] 8.2 Registry lifecycle: capacity, promotion, retirement stops scoring without touching the chain
- [x] 8.3 Scoring: bonus on intact reproduction, penalty on a break with the right token available, **no penalty when it was not**
- [x] 8.4 Neutral defaults leave generated text identical (`generation_hash` unchanged) — `5a72e2d4…` on `db_prod_copy`, identical to the Phase 3 frozen baseline
- [x] 8.5 `/clear confirm` leaves no registry row
- [x] 8.6 Property: scores are finite for any legal counts, including the degenerate single-token corpus

## 9. Manual rating round (needs people — see 1.1)

- [x] 9.1 Produce the top-20 ranking and a rating sheet template (real / merely frequent / junk) — `tools/meme_rating_round.py`; the 2026-08-12 round is prepared locally in `rating_rounds/` (42 positions: 20 meme, 20 frequency control, 5 decoys, overlaps shown once)
- [x] 9.2 Rating sheet stays out of the repository — it is verbatim chat content (spec: `generation-collocations`)
- [x] 9.3 Conduct the round; record rater count and per-category counts — conducted 2026-08-12/13, 3 raters; meme half 8/20, frequency control 2/20, decoys 1/5
- [x] 9.4 Aggregate into the report: counts, shares, inter-rater agreement when there is more than one rater; explicitly "agreement unavailable" for a single rater — aggregated in `rating_rounds/2026-08-12/manual_rating.json` (out of the repo), agreement = Fleiss' kappa 0.17 over three categories. Recorded as an UPPER bound: one rater's answer arrived unnumbered with 46 labels instead of 42, and the reduction to 42 was an alignment against the other two, which inflates agreement by construction. The aggregate itself is invariant — all 60 equally-optimal alignments yield the same counts

## 10. Eval

- [x] 10.1 `tools/eval/matrix.yaml`: C3 arm (meme-aware ordering + collocation scoring), `available: true` only for what this change ships
- [x] 10.2 Protocol run C0 vs C3 at protocol volumes; the automatic gate conditions computed from it — 1500 gens/arm, copy Δ −0.033 [−0.063, −0.005]*, affinity_without_copy Δ −0.016 [−0.039, 0.006] (floor holds), p95 40.0 ms; note: repetition_rate rose 0.001→0.007* (not a gate condition, recorded for the verdict reader)
- [x] 10.3 Gate reports `insufficient data` while the manual rating is missing — never `pass` (spec: `generation-eval`) — verified on the real run: phase4_memes[C3] says `insufficient data — missing: manual top-meme rating`
- [x] 10.4 Report in `docs/eval_reports/`, referenced from this change — `docs/eval_reports/eval_2026-08-12_phase4-c3.md`
- [x] 10.5 CI smoke stays green — `python -m tools.eval --smoke` passes with the C3 arm in the matrix

## 11. Default decision (gate)

- [ ] 11.1 Gate passed ⇒ propose raising the defaults, with the numbers — **branch not taken**: the gate failed, so no defaults were proposed
- [x] 11.2 Gate fails ⇒ defaults stay neutral; record the negative result with numbers in `docs/v2/00_STATUS.md` and reference the report — gate `fail` on 2026-08-13 (`eval_2026-08-13_phase4-verdict.md`): manual 8/20 genuine (40%) against the 70% bar; all knobs stay at their neutral defaults
- [x] 11.3 Never adjust a pre-registered threshold to make the gate pass — honoured: `phase4_memes` in `eval_thresholds.yaml` is untouched since its 2026-08-12 pre-registration, and the 40%-vs-70% failure was recorded rather than tuned away

## 12. Documentation

- [x] 12.1 `docs/v2/00_STATUS.md`: Phase 4 row + next-session pointer
- [x] 12.2 `.env.example` with bounds and one-line meanings; `GENERATION_PIPELINE.md` where scoring is described
- [x] 12.3 Record that collocation gluing stays out of scope and why (ADR-016), so it is not rediscovered as an idea — ADR-016 + TZ §10.2 already normative; now also restated in `GENERATION_PIPELINE.md` at the exact place a reader meets the scoring

## 13. Housekeeping riding along (no functional content)

- [x] 13.1 `openspec archive markov2r-phase3-temporal-layer-core` — its own commit, established convention

## 14. Close-out

- [x] 14.1 `openspec validate --strict` green for this change
- [x] 14.2 Full test suite + lint/type checks — 1095 tests OK, ruff clean, mypy clean (2026-08-12)
- [x] 14.3 Archive this change after merge — rides in the next phase's PR — archived 2026-08-13 alongside the Phase 4 verdict
