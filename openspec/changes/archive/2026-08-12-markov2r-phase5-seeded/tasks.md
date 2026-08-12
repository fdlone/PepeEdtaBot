# Tasks: markov2r-phase5-seeded

Phase 5, second half (M2R-410/420). Ships disabled: `ratio = 0` leaves every
reply byte-identical, proven by `generation_hash` at close-out. The promotion
decision (M2R-430) is a follow-up — it needs prod-accumulated df.

## 1. Seed scoring (M2R-410, TZ §9.4)

- [x] 1.1 `app/core/seed.py` — pure `seed_score = normalized_idf × support_factor × branching_quality`; no SQL/clock/RNG (pattern of `collocations.py`). Reuse `collocations.support_factor`; branching_quality is a trapezoid over `min(forward_branch, reverse_branch)` (design D1)
- [x] 1.2 Stopword + `MARKOV_SEED_MIN_TOKEN_LEN` filter before scoring; degenerate df (`n_docs == 0` / unknown token) yields no score, not a crash
- [x] 1.3 Tests: junk unique token scores below a well-supported distinctive one; branching outside the band drives the score down; property — score finite for any legal counts incl. single-token corpus

## 2. Bidirectional generation (M2R-410, TZ §9.5)

- [x] 2.1 Reverse walk in `MarkovGenerator` over `get_reverse_transitions`, sampling predecessors with the **same** weighted-choice + entropy/temporal helpers as the forward walk (design D2); factor the shared step so head and tail cannot drift
- [x] 2.2 Seed-anchored assembly `head_reversed + [seed] + tail`; head/tail split by `MARKOV_SEED_HEAD_SHARE` within `max_reply_tokens`; copy-on-write pools like the forward walk
- [x] 2.3 Reverse pool absent → head stops at the anchor, candidate still assembled from the tail
- [x] 2.4 Test: reverse walk over a hand-built chain yields the expected predecessor distribution; assembled candidate places the anchor mid-reply

## 3. Seeded candidates in the pool (M2R-410, ADR-008)

- [x] 3.1 `MARKOV_SEEDED_CANDIDATE_RATIO` of the `candidate_target` slots filled by seeded assembly; seed token drawn from the top of the per-message seed ranking; read once per reply, only when ratio > 0
- [x] 3.2 Seeded candidates scored by the existing scorer with no priority; compete in the same selection
- [x] 3.3 Availability/df/reverse reads answered from the M2R-400 API; no query per candidate beyond the one seed ranking per reply

## 4. Telemetry (TZ §9.6, generation-telemetry spec)

- [x] 4.1 `note_seeded(present, won)` on `GenerationTelemetry`; `seeded_present_rate` and `seeded_win_rate_given_present` in `snapshot()` with separate denominators
- [x] 4.2 `/stats` surfaces the two rates when any seeded generation has occurred
- [x] 4.3 Test: configured-but-never-anchoring reports present-count 0, distinguishable from ratio 0

## 5. Knobs (established 5-step pattern)

- [x] 5.1 Registry + `Settings`/`RuntimeState` + `.env.example`: `markov_seed_branch_min`, `markov_seed_branch_ideal`, `markov_seed_branch_max`, `markov_seed_min_token_len`, `markov_seed_min_score`, `markov_seeded_candidate_ratio` (0–0.7), `markov_seed_head_share`
- [x] 5.2 Ratio and all seed weights default to neutral (ratio 0); bounds validated. Cross-field: `branch_min ≤ branch_ideal ≤ branch_max` rejected otherwise (spec: `runtime-knob-validation` pattern)
- [x] 5.3 `/config full` surfaces the seed knobs

## 6. Neutrality (the shipping contract)

- [x] 6.1 `ratio == 0` short-circuits before the seed ranking and before any df/reverse read — no extra RNG draw, RNG-consumption order unchanged (design D5)
- [x] 6.2 Test: reply pipeline issues no reverse/df query at the default
- [x] 6.3 `python -m tools.generation_hash --db db_prod_copy/markov.db` — identical to the frozen baseline (`5a72e2d4…`) — confirmed 2026-08-12
- [x] 6.4 Neutral defaults leave generated text identical (`generation_hash` unchanged) — the spec scenario

## 7. Eval (generation-eval spec)

- [x] 7.1 `tools/eval/run.py`: C4 arm populates df from the copy's retained messages at the fixed evaluation moment (C3 precedent, mirrored for df, design D4)
- [x] 7.2 `tools/eval/matrix.yaml`: C4 arm `available: true` with `markov_seeded_candidate_ratio` set (and neutral surface layers), `available: true` only for what this change ships
- [x] 7.3 `phase5_promotion` gate computed in `tools/eval/report.py` from the run (replace the "does not exist before Phase 5" stub): seeded present/win rates, affinity delta vs no-seeded arm, p95, storage
- [x] 7.4 Gate reports `insufficient data` while df is window-approximated — never `pass` (spec: `generation-eval`)
- [x] 7.5 Protocol run C0 vs C4 on `db_prod_copy`; report in `docs/eval_reports/`, referenced from this change — `eval_2026-08-12_phase5-c4.md`: even on window-approximated df, C4 raises `context_affinity_without_copy` **+0.064 [0.041, 0.087] \*** (the gate's own metric), copy Δ +0.016 (not significant), p95 88.7 ms (budget 150). Gate row: seeded present 81%, win|present 18% (below the 40% bar on window-df); verdict `insufficient data` pending prod-accumulated df (M2R-430)
- [x] 7.6 CI smoke stays green

## 8. Migration verification (standing rule: whole 2.0R package, one restart)

- [x] 8.1 No new migration this change; re-confirm the full 001→020 chain still applies cleanly on a copy of `db_prod_copy` after this change's code lands — whole package applies at once, 20 migrations, seed-path structures live and queryable (reverse index used, df/n_docs present)

## 9. Documentation

- [x] 9.1 `GENERATION_PIPELINE.md`: seeded branch (seed score, bidirectional assembly, ratio, transparent fallback); note §9.7 candidate IDF stays window-based (design D6)
- [x] 9.2 `docs/v2/00_STATUS.md`: Phase 5 seeded row + next-session pointer to the M2R-430 decision (needs the accumulation window)
- [x] 9.3 `.env.example` bounds and one-line meanings for the seed knobs

## 10. Default decision (gate — M2R-430, follow-up boundary)

- [ ] 10.1 Gate over prod-accumulated df passes ⇒ propose raising the ratio, with numbers; ADR-012 → Accepted
- [ ] 10.2 Gate fails ⇒ ratio stays 0, reverse/df structures frozen (not read), record the negative result with numbers in `docs/v2/00_STATUS.md`, reference the report
- [ ] 10.3 Never adjust a pre-registered threshold to make the gate pass

## 11. Close-out

- [x] 11.1 `openspec validate --strict` green for this change
- [x] 11.2 Full test suite + lint/type checks — 1119 tests OK, ruff clean, mypy clean (2026-08-12)
- [x] 11.3 Archive this change after merge
