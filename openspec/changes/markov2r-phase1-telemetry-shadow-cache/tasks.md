# Tasks: markov2r-phase1-telemetry-shadow-cache

## 1. Knobs and scaffolding

- [x] 1.1 Registry entries + `Settings`/`RuntimeState` fields + `.env.example`: `markov_cache_max_entries`, `markov_cache_incremental`, `markov_shadow_order4_enabled` (5-step pattern; drift tests must stay green)
- [x] 1.2 `GenerationTelemetry` accumulator on `MarkovGenerator` (cache hits/misses, entropy sums, shadow counters), injected into handlers via existing DI

## 2. M2R-010 — Distribution diagnostics

- [x] 2.1 Pool diagnostics helper (entropy bits, normalized entropy, branching, confidence) computed from raw-count proportions; unit tests incl. degenerate pools (B=0, B=1)
- [x] 2.2 Wire into walk steps and start selection; aggregate per winning attempt into `GenerationTrace` new fields; per-step lines in `gen_trace_log` when enabled
- [x] 2.3 Prove behavior neutrality: characterization tests byte-identical; no new RNG draws (seeded-run hash unchanged)

## 3. M2R-020 — Shadow order selector

- [x] 3.1 Window-based order-4 support index in `LearningService` (4-token key → continuation counts; built lazily, folded incrementally like sibling caches; TTL/max-chats eviction applies)
- [x] 3.2 Shadow decision per eligible step (support ≥ `MARKOV_ORDER4_MIN_COUNT`, confidence rule from TZ §5), counters into telemetry; knob-gated; `estimator=window` label in output
- [x] 3.3 Tests: known corpus with a planted high-support 4-gram → shadow selects; disabled knob → zero cost path; behavior neutrality (hash unchanged)

## 4. M2R-030 — Incremental cache

- [x] 4.1 `apply_learning_deltas` on `MarkovGenerator`: fold message deltas into `_cache3`/`_cache2`/`_cache_starts3`/`_cache_starts2` with bisection insert preserving SQL ordering; `markov_cache_incremental=false` falls back to full wipe; `/clear` still wipes
- [x] 4.2 Incremental update for `ContextStateMatcher` index (exact + casefolded buckets)
- [x] 4.3 `LearningService.record_message` passes deltas (already computed by the learning write) instead of calling blanket invalidation
- [x] 4.4 `cache_limit` → `markov_cache_max_entries` knob; hit/miss counters in every cache read
- [x] 4.5 Equivalence tests: fold-vs-fresh cache contents after randomized message sequences (property test); ordering test against a fresh SQL read; generation byte-identity with warm vs cold caches at fixed seed
- [x] 4.6 `tools/generation_hash.py` on the prod snapshot: identical hashes before/after the change, in both knob positions

## 5. /stats and observability

- [x] 5.1 `/stats` telemetry block (Russian text in `bot_messages.py`): mean entropy/branching, cache hit-rate, shadow order-4 share with `estimator=window` caveat
- [x] 5.2 Log lines: masked chat ids only, numbers/labels only (log-privacy conventions)

## 6. Eval gate and docs

- [x] 6.1 Full protocol run on the frozen snapshot: content metrics byte-identical to C0 baseline; latency/hit-rate section recorded; report committed to `docs/eval_reports/` and linked here
- [x] 6.2 Full suite + ruff + mypy + coverage green
- [x] 6.3 Docs: `docs/GENERATION_PIPELINE.md` (diagnostics + cache invalidation sections), `docs/ARCHITECTURE.md` (cache policy), `.env.example` comments
- [x] 6.4 `openspec validate --strict`; archive after merge (archive pending)
