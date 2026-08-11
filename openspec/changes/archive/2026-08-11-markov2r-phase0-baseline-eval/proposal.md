# Proposal: markov2r-phase0-baseline-eval

## Why

Markov 2.0R (normative package `docs/v2/01–05`, v1.1) upgrades the bot's generation core to an entropy-aware, time-adaptive, lexically anchored — but still purely Markov/statistical — system. Its governing principles P7 (empiricism over intuition) and P8 (ablation) make every later phase conditional on measurements against a frozen baseline. Phase 0 builds that foundation: a pre-implementation audit of the actual code against the target spec, and the offline eval infrastructure (ablation matrix, frozen baseline, pre-registered gate thresholds) that every subsequent phase uses as its acceptance gate. No generation behavior changes in this phase.

## What Changes

- **Pre-implementation audit document** `docs/v2/MARKOV_2_0R_PRE_IMPLEMENTATION_AUDIT.md`: environment & tooling inventory, current Markov architecture (state format, orders, backoff, sampling, scoring, jumps, mood, hot n-grams, flavor), normalization/tokenization in detail, SQLite schema and query patterns, privacy/retention machinery, Telegram-reactions availability (Phase 8 gate), gap analysis against TZ v1.1, do-not-break points, migration risks, per-phase complexity estimates, and any prompt/docs/code contradictions found.
- **Offline eval runner** `tools/eval/` (`python -m tools.eval --db <snapshot> --config <matrix> --seed N`, doc 03 M2R-001): runs the ablation matrix C0–CF from doc 05 §2, computes the metrics of doc 05 §3 (implementation referencing section numbers), prints deltas vs baseline with bootstrap 95% CIs (doc 05 §4), emits a markdown report to `docs/eval_reports/` (doc 05 §6).
- **Prompt set** `eval_prompts.yaml` (doc 05 §1.2): four categories (generic / topical / meme-bait / short-degenerate), ≥30 prompts each; topical and meme-bait built from the temporal snapshot by a seeded script, not by hand.
- **Threshold pre-registration** `eval_thresholds.yaml` (doc 05 §4): gate thresholds for Phase 5 promotion, Phase 6 rate×harm, Phase 7 shadow — fixed before any gated data is looked at; later edits require a dedicated commit with justification.
- **Frozen baseline C0** (doc 03 M2R-000): snapshot set (`snapshot_full`, `snapshot_temporal`, `snapshot_synthetic`), baseline metrics recorded, bit-for-bit reproducible under fixed snapshot + prompts + seeds.
- **CI smoke** (doc 05 §7): C0 + CF on `snapshot_synthetic`, 40 generations, 1 seed, on PRs touching the generation core.
- Dev-dependency addition: `hypothesis` (property tests required by TZ §19 for later phases; added now so the test scaffolding pattern is settled). No other new dependencies; bootstrap CIs are implemented in pure stdlib.

## Capabilities

### New Capabilities

- `generation-eval`: offline evaluation harness for the generation pipeline — ablation configuration matrix, normative metric definitions, seeded reproducibility, pre-registered gate thresholds, report format, and CI smoke behavior.

### Modified Capabilities

None. Phase 0 does not change bot behavior; existing specs (`chat-scoped-settings`, `command-rate-limits`, `in-memory-state-eviction`, `log-privacy`, `pivo-*`, `runtime-knob-validation`) are untouched.

## Impact

- **New code**: `tools/eval/` package (runner, metrics, report writer, snapshot tooling), prompt/threshold YAML files, snapshot-preparation script(s).
- **Existing code**: read-only reuse of the generation pipeline (the runner drives the real generation entry point against a snapshot DB, as `tools/eval_prod.py` does today); no changes to `app/` except, if strictly needed, non-behavioral seams for deterministic seeding — any such seam is listed in design.md.
- **Docs**: new audit document; `docs/eval_reports/` directory; AGENTS.md tooling table if dev-deps change.
- **Dependencies**: `hypothesis` in `requirements-dev.txt` (dev-only). No runtime dependencies.
- **Out of scope**: any change to live generation, schema migrations, new runtime knobs. Phase 1 starts only after this change is archived (master prompt §4).
