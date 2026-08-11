# Proposal: markov2r-phase1-telemetry-shadow-cache

## Why

Phase 1 of Markov 2.0R (roadmap doc 03: M2R-010/020/030) instruments the generator before any behavior changes: later phases need distribution diagnostics (entropy drives Phase 2 sampling), gated Phase 7 needs shadow order-4 selection data, and the audited cache pathology — all four Markov LRU caches wiped per learned message while `respond` runs before `learn`, so an active chat generates from cold caches — must be fixed before Phase 3/5 multiply lookups (layers + reverse + best-of-N make the p95 budget unreachable without a working cache; TZ §14). Generation output MUST NOT change: this phase is measurement plus performance.

## What Changes

- **M2R-010 — Distribution diagnostics.** Entropy (`H = -Σ p·log2 p`), normalized entropy (`H/log2 B`, 0 when B ≤ 1), branching factor, and confidence (`1 − H_norm`) computed for every transition pool the walk actually samples from (TZ §6 formulas). Aggregated per generation into the existing `GenerationTrace` (mean/last-step values), emitted in `gen_trace_log`, and accumulated into per-process telemetry counters surfaced by `/stats`. No behavior change; no extra RNG draws.
- **M2R-020 — Shadow order selector.** At each order-3 step, computes whether a variable-order 4→3→2 selector (TZ §5 criteria: support ≥ `ORDER4_MIN_COUNT`, confidence threshold) *would have* chosen order 4 — estimated from the retained message window, since no order-4 index exists (that is the point of the gate, ADR-002). Counters (`shadow_order4_eligible`, `shadow_order4_selected_share`) go to telemetry and `/stats`; this data later feeds the pre-registered Phase 7 gate. Zero influence on generation.
- **M2R-030 — Bounded cache with incremental invalidation.** The existing per-chat full wipe on every learned message is replaced by folding the learned message's exact deltas into the cached structures (transitions/starts caches and the context-matcher index), preserving SQL ordering invariants. Bounds stay (LRU, `markov_cache_max_entries` knob replaces the hardcoded 1024). Hit-rate counters exposed in `/stats`. Behavior-preservation is proven, not assumed: `tools/generation_hash.py` byte-identity before/after, plus fold-vs-fresh equivalence tests. A kill-switch knob (`markov_cache_incremental=false`) restores the old wipe behavior (ADR-010).
- Eval: a protocol run comparing latency (p50/p95, cache hit-rate) against the frozen C0 baseline; content metrics must be byte-identical to C0.

## Capabilities

### New Capabilities

- `generation-telemetry`: what the generator must be able to report about its own distributions (entropy, branching, confidence, shadow order selection, cache hit-rate), where those numbers surface (`/stats`, trace log), and what must never leak into them (raw text, unmasked chat ids).

### Modified Capabilities

- `in-memory-state-eviction`: the "no full rebuild per message" requirement currently binds derived chat caches (LearningService); it is extended to the Markov distribution caches, whose current invalidation-by-wipe guarantees a cold cache exactly when it is needed. Adds the requirement that incremental invalidation MUST NOT change observable generation output.

## Impact

- **Code**: `app/core/markov.py` (diagnostics computation, cache fold, hit counters), `app/core/context_state_matcher.py` (incremental index update), `app/core/gen_trace_log.py` (new fields), `app/services/learning_service.py` (pass learned deltas instead of blanket invalidation), `app/handlers/common.py` + `app/presentation/bot_messages.py` (`/stats` telemetry block), `app/config/registry.py` + `.env.example` (knobs: `markov_cache_max_entries`, `markov_cache_incremental`, `markov_shadow_order4_enabled`).
- **No schema changes, no migrations, no new dependencies.**
- **Behavior**: generation output byte-identical (verified); `/stats` output gains a telemetry block (user-visible text change, Russian).
- **Out of scope**: entropy influencing sampling (Phase 2), any order-4 table (Phase 7), s_value caching semantics (Phase 3 — the cache design leaves room for raw-triple caching per TZ §14).
