# Tasks: markov2r-phase0-baseline-eval

## 1. Pre-flight

- [x] 1.1 Owner review of `docs/v2/MARKOV_2_0R_PRE_IMPLEMENTATION_AUDIT.md`; incorporate corrections (audit was written at proposal time)
- [x] 1.2 Resolve open question: baseline snapshot source (existing `db_prod_copy` 2026-07-13 vs fresh production dump) — record the decision in the audit §10.2
- [x] 1.3 Add `hypothesis` to `requirements-dev.txt` (approved install) and verify the suite still runs

## 2. Eval package core

- [x] 2.1 Scaffold `tools/eval/` package with `__main__` CLI (`--db`, `--config`, `--seed`, `--seeds`, `--generations`, `--smoke`, `--out`)
- [x] 2.2 `config.py`: matrix loading; commit `tools/eval/matrix.yaml` with frozen C0 definition (registry defaults, flavor/emoji silenced) and CF ≡ C0
- [x] 2.3 `run.py`: per-config × per-seed × per-category loop reusing `eval_prod` helpers (`copy_database`, instrumentation, winner attribution); per-generation RNG `Random(seed*100_000+index)`
- [x] 2.4 `metrics.py`: doc 05 §3.1–3.3, §3.6 implementations, each referencing its section number; seeded (§3.4) and temporal (§3.5) metrics return `insufficient data` markers in Phase 0
- [x] 2.5 `bootstrap.py`: stdlib percentile bootstrap (≥1000 resamples, pooled seeds, seeded resampling RNG)
- [x] 2.6 `report.py`: markdown report per doc 05 §6 (header with snapshot id / prompt version / seeds / revision, matrix, metric table with CIs and deltas vs C0, per-category breakdown, gates, manual-eval placeholder, verdict) → `docs/eval_reports/`

## 3. Prompts and thresholds

- [x] 3.1 `prompts.py --generate`: deterministic prompt generation from `snapshot_full` (generic, topical via low-df/adequate-support tokens, meme-bait via hot/verbatim n-grams, short/degenerate), ≥30 per category
- [x] 3.2 Generate, owner-review, and commit `eval_prompts.yaml` (version stamp; no real identifiers — `test_no_real_chat_ids` extended to cover it)
- [x] 3.3 Write `eval_thresholds.yaml` with proposed pre-registered gate values (Phase 5 promotion, Phase 6 rate×harm, Phase 7 shadow); owner sign-off recorded

## 4. Snapshots and baseline freeze

- [x] 4.1 `snapshot_synthetic` builder: deterministic temp-DB construction wrapping `seed_diverse` (used by smoke and tests)
- [x] 4.2 Document the local `snapshot_full` preparation procedure (copy + migrations, id = date+hash, stored outside repo)
- [x] 4.3 Freeze approved baseline snapshot; run the full protocol (500/config × 3 seeds) for C0; commit baseline report to `docs/eval_reports/`
- [x] 4.4 Verify bit-for-bit reproducibility: re-run C0 with identical inputs, assert byte-identical metrics (latency excluded)

## 5. Tests

- [x] 5.1 Unit tests for `metrics.py` (known-answer fixtures per §3 definition) and `bootstrap.py` (degenerate and known distributions)
- [x] 5.2 Unit tests for prompt generation determinism and category constraints
- [x] 5.3 Reproducibility test: two smoke runs on `snapshot_synthetic`, equal metrics (latency excluded)
- [x] 5.4 First property tests with `hypothesis` (bootstrap CI bounds ordering; metric ranges ∈ [0,1]) — establishes the §19 testing pattern for later phases
- [x] 5.5 Full existing suite green; ruff/mypy/coverage unchanged or better

## 6. CI

- [x] 6.1 Add smoke step (C0+CF, `snapshot_synthetic`, 40 generations, 1 seed) to CI, path-filtered to `app/core/**` + `tools/eval/**`; fails on errors/invariant violations only

## 7. Documentation

- [x] 7.1 Update `docs/v2/05_MARKOV_2_0R_EVAL_PROTOCOL.md` with the approved temporal-snapshot workaround (audit §10.1) as a dedicated commit
- [x] 7.2 Update `README.md` / `AGENTS.md` tooling notes (`tools/eval` usage, hypothesis dev-dep)
- [x] 7.3 `openspec validate --strict` passes; link baseline report in this change
