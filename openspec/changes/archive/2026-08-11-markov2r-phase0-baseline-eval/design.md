# Design: markov2r-phase0-baseline-eval

## Context

See `proposal.md` for motivation and `docs/v2/MARKOV_2_0R_PRE_IMPLEMENTATION_AUDIT.md` for the full code audit this design builds on. The facts that shape the approach:

- The pipeline is **reproducible under an injected seeded RNG** and this is a maintained invariant (`tools/generation_hash.py`, characterization tests, the "disabled knob consumes no draw" pattern). Bit-for-bit baseline reproducibility is therefore achievable without touching `app/`.
- `tools/eval_prod.py` already contains most of the machinery a doc-05 runner needs: DB snapshot copying (`copy_database`), trace instrumentation (`_TraceCapturingGenerator`, `_PoolCollector`), winner attribution through the extension/mutation channels, registry-default runtime state construction, and hard-won methodology (≥400 generations per arm, `distinct_basis_tokens`).
- The chain tables carry no timestamps, so `snapshot_temporal` (doc 05 §1.1) cannot exist until Phase 3 lands and accumulates data — audit §10.1 records the accepted workaround.
- The project uses stdlib `unittest`, no pytest, no numpy; CI is GitHub Actions with ruff/mypy/coverage/bandit/pip-audit.

## Goals / Non-Goals

**Goals**

- A `tools/eval` package implementing doc 05: ablation matrix, normative metrics (with §-number references), bootstrap CIs, markdown reports, gate evaluation against `eval_thresholds.yaml`.
- Frozen, reproducible C0 baseline and committed prompt/threshold files.
- CI smoke job per doc 05 §7.
- The pre-implementation audit document (written during proposal; reviewed at apply).

**Non-Goals**

- No changes to generation behavior, schema, or runtime knobs.
- No `snapshot_temporal` and no `freshness_reflection` numbers in Phase 0 (reported as `insufficient data`).
- No manual-eval round in Phase 0 (the protocol §5 machinery is exercised first at Phase 4; the report format includes the section from day one).

## Decisions

**D1 — New package `tools/eval/`, importing from `tools/eval_prod.py`, not replacing it.**
`eval_prod` stays as the knob-sweep harness; the new runner is the *protocol* implementation (matrix, categories, CIs, gates, reports). Shared helpers (`copy_database`, instrumentation, winner attribution) are imported from `eval_prod`; anything that must change for reuse is extracted, not duplicated. Alternative — growing `eval_prod` itself — rejected: its CLI contract and sweep scripts must stay stable as the refactor guard during all later phases.

Layout:

```text
tools/eval/__init__.py
tools/eval/__main__.py      # python -m tools.eval --db <snapshot> --config <matrix.yaml> --seed N
tools/eval/config.py        # matrix + thresholds loading (C0..CF definitions as knob dicts)
tools/eval/metrics.py       # doc 05 §3 implementations, each referencing its section number
tools/eval/bootstrap.py     # stdlib percentile bootstrap (≥1000 resamples, pooled seeds)
tools/eval/prompts.py       # eval_prompts.yaml loading + deterministic generation from snapshot
tools/eval/report.py        # markdown report per doc 05 §6 → docs/eval_reports/
tools/eval/run.py           # orchestration: per-config × per-seed × per-category loop
```

**D2 — Config matrix as data, not code.** `C0..CF` are named dicts of runtime-knob overrides (the exact mechanism `eval_prod` and the sweeps already use). In Phase 0 the matrix file defines C0 (registry defaults with `reply_flavor_strength=0`, `emoji_append_chance=0` — the established eval convention) and CF ≡ C0; later phases add their C-configs by editing the matrix file, not the runner. The frozen C0 definition is committed and never edited afterwards (new baselines get new IDs).

**D3 — Reproducibility contract.** Per-generation RNG is `random.Random(seed * 100_000 + index)` (the `generation_hash.py` pattern), prompt selection uses its own `random.Random(seed)`; latency metrics are excluded from the bit-for-bit comparison (the existing `test_eval_generation.py` convention). The report records snapshot id, prompt-set version, seeds, and code revision.

**D4 — Bootstrap in stdlib.** Percentile bootstrap over per-generation outcome arrays, `random.Random(fixed_seed)` for resampling so CIs are themselves reproducible. No numpy (audit §1).

**D5 — Snapshots.**
- `snapshot_synthetic`: built on demand by a deterministic script wrapping `tools/seed_diverse.py` into a temp DB — regenerated in CI, never committed.
- `snapshot_full`: an anonymized copy of the production DB, prepared by a documented local procedure (copy + migrations via `copy_database`), stored outside the repo, referenced by id (date+hash) in reports. Whether to freeze on the existing `db_prod_copy` (2026-07-13) or a fresh dump is an owner decision — audit §10.2 recommends a fresh one.
- `snapshot_temporal`: explicitly absent in Phase 0; the runner reports temporal metrics as `insufficient data` (audit §10.1).

**D6 — Prompt generation without a temporal snapshot.** `eval_prompts.yaml` is produced by `tools/eval/prompts.py --generate` from `snapshot_full` with a fixed seed: *generic* and *short/degenerate* from templated samples over the snapshot's message shapes; *topical* built around tokens with low df and adequate chain support (the seed-score precursors); *meme-bait* built around top `chat_hot_ngrams`/`chat_verbatim_ngrams` n-grams. The generated file is committed (it contains chat-derived tokens but no identifiers; the no-real-chat-ids guard applies) and versioned; regeneration is a deliberate act producing a new version.

**D7 — Threshold pre-registration now.** `eval_thresholds.yaml` is written in this change with concrete numbers for Phase 5 promotion, Phase 6 rate×harm, and Phase 7 shadow, before any gated data exists — that is the point of pre-registration (doc 05 §4). Proposed values go to the owner with the proposal; after approval they are fixed and any later edit needs a dedicated justified commit.

**D8 — CI smoke as a separate job step.** A step in the existing `test` job (one Python version) runs `python -m tools.eval --smoke` (C0+CF, `snapshot_synthetic`, 40 generations, 1 seed) and fails on runner errors or invariant violations. Full matrix runs stay manual (doc 05 §7). *Implementation note:* instead of path-filtering (which needs an extra action or workflow-level trigger surgery), the step runs on every PR but only in the `python 3.12` matrix cell — a strict superset of "PRs touching the generation core" at ~1 minute of CI time, with zero new dependencies.

**D9 — Metric implementations lean on audited code, not reimplementation.** `exact_context_copy_rate` reuses the verbatim-index conventions (`candidate_scorer` window helpers); `context_affinity` reuses `build_token_idf`-style IDF computed over the snapshot's retained messages — with the honest limitation, recorded in the report, that df comes from the retention window (full history is not stored; audit §3). `distinct-2/3` always publish their token-count basis (`distinct_basis_tokens` lesson).

## Risks / Trade-offs

- **[IDF from retention window]** df over the last 1000 messages, not full history → affinity metrics are window-relative. Mitigation: recorded in report header; consistent across configurations, so deltas remain valid; Phase 5's incremental df-aggregate will supersede it.
- **[Committed prompts derive from chat text]** meme-bait prompts contain chat n-grams. Mitigation: no identifiers (guard test), owner reviews `eval_prompts.yaml` before commit; snapshots themselves never leave the machine.
- **[Smoke flakiness]** 40 generations on synthetic data is noisy. Mitigation: smoke gates only on errors/invariants, never on metric thresholds.
- **[Runner drift vs eval_prod]** two harnesses sharing helpers. Mitigation: shared code imported from one place; `generation_hash.py` remains the bit-for-bit refactor guard.

## Migration Plan

No DB migrations, no runtime changes. Rollout = merge; rollback = revert. The frozen baseline artifacts (matrix file, prompts, thresholds, baseline report) are additive.

## Open Questions

1. Which snapshot freezes as the C0 baseline: existing `db_prod_copy` (2026-07-13) or a fresh production dump? (Owner; recommendation: fresh — audit §10.2.)
2. Threshold numbers in `eval_thresholds.yaml` — proposed values ship with this change for owner sign-off; they are parameters of the protocol, not of this design.
