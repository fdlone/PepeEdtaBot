# Design: markov2r-phase1-telemetry-shadow-cache

## Context

See proposal.md for motivation; audit (`docs/v2/MARKOV_2_0R_PRE_IMPLEMENTATION_AUDIT.md` §2, §6) for the code facts this design leans on:

- All walk sampling flows through `weighted_next_choice` / `weighted_start*_choice` (`app/core/markov.py`); pools are `list[tuple[str, int]]` with raw counts as weights.
- `GenerationTrace` already aggregates per-attempt counters and is logged at DEBUG; `gen_trace_log` is the opt-in detailed channel.
- The four Markov LRU caches (`_cache3`, `_cache2`, `_cache_starts3`, `_cache_starts2`) plus the `ContextStateMatcher` index are wiped per chat on every learned message (`learning_service.py:188` → `invalidate_chat_cache`); `respond` precedes `learn`, so replies in an active chat run on cold caches.
- Learning knows the exact deltas: `save_message_and_update_model` builds `trans2_counter`/`trans3_counter` (distinct n-grams of the message) and the start tuples — the same data a fold needs.
- The seeded-RNG consumption order is a maintained contract (`tools/generation_hash.py`); byte-identity is the acceptance instrument.
- Byte-identity guard for this phase: `generation_hash` run before/after on the prod snapshot must produce the same SHA-256 per seed.

## Goals / Non-Goals

**Goals**: diagnostics per sampled pool; shadow order-4 statistics; warm caches with exact-delta invalidation; `/stats` telemetry block; all provably behavior-neutral.

**Non-Goals**: entropy-driven sampling (Phase 2); any order-4 storage (Phase 7); caching of time-dependent `s_eff` (Phase 3 — but the cache API leaves room for raw-triple values per TZ §14); persistence of telemetry (process-lifetime counters only); changes to `eval_prod`/sweeps.

## Decisions

**D1 — Diagnostics ride on `GenerationTrace`, not a new dataclass.** TZ §4.3 sketches `GenerationDiagnostics`; the codebase already has `GenerationTrace` doing that job (master prompt: existing code beats prescribed shapes). New fields: `mean_entropy_bits`, `mean_normalized_entropy`, `mean_branching`, `min_confidence`, `diagnostic_steps` — aggregated over the steps of the *winning* attempt. Computation happens on the pool actually sampled, before sampling — no extra SQL, no RNG draws, O(B) math per step (pools are tens of entries). *Implementation note:* the opt-in `gen_trace_log` carries the per-attempt summary in the accepted-candidate route line rather than a line per step — per-step lines would multiply hot-path logging by the walk length for no analytical gain (the accumulator already exposes the mean and the minimum).

**D2 — Entropy is computed over raw-count proportions, not tempered weights.** `H(S)` characterizes the *model's* distribution (TZ §6 defines it on transition probabilities), not the temperature-adjusted sampling weights; Phase 2 will then modulate temperature from it. The tempered distribution stays untouched.

**D3 — Shadow order-4 support from the retained message window.** No order-4 index exists, and building one is exactly what the Phase 7 gate decides. The estimator: for a step with history `(w0,w1,w2,w3)`, order-4 support = number of occurrences of the 5-token continuation pattern in the retained normalized messages (`LearningService` text window, already cached in memory). Selector shadow rule per TZ §5: support ≥ `MARKOV_ORDER4_MIN_COUNT` (default 3) AND order-4 confidence ≥ threshold. Properties: zero storage, zero SQL on the hot path (in-memory window scan bounded by a per-generation step budget), and **conservatively biased low** (window ⊂ full history) — if even the window-estimate exceeds the gate threshold, the signal is strong; recorded in telemetry as `estimator=window`. Alternative rejected: a shadow count table — it pre-builds the storage whose justification is the question, and adds hot-path writes.
Window occurrence scans are pre-indexed once per window generation (a dict from 4-token keys to continuation counters, folded incrementally like the other LearningService caches), so the per-step cost is a dict lookup, not a scan. *Implementation note:* the shadow analysis runs post-hoc over the winning reply's token sequence (consecutive walk states ≡ consecutive reply tokens) instead of inside the walk loop — zero touch on the loop, zero RNG risk, and it measures exactly the replies that reach the chat. Replies containing jumps/splices/extensions/mutations are skipped: their token adjacency crosses splice boundaries. The selector constants (support ≥ 3, confidence ≥ 0.35 — TZ §5 defaults) stay module constants until Phase 7 promotes them to knobs.

**D4 — Cache fold instead of wipe.** `invalidate_chat_cache(chat_id)` grows a sibling `apply_learning_deltas(chat_id, deltas)` where `deltas` carries the message's start tuples and n-gram counters (already computed by the learning write). Fold rules:
- `_cache3`/`_cache2`: for each learned n-gram whose state key is cached, increment the matching `(token, cnt)` entry or insert it **preserving the SQL `ORDER BY` position** (lists are ordered by token columns, not by count — insertion by bisection keeps the contract sampling relies on).
- `_cache_starts3`/`_cache_starts2`: same fold for the message's start tuple.
- `ContextStateMatcher` index: increment/insert the affected state entries (exact and casefolded buckets).
- Keys not touched by the message stay warm — that is the entire point.
`markov_cache_incremental=false` short-circuits to the old wipe. `/clear` still wipes unconditionally. LRU bounds unchanged; `cache_limit` becomes the `markov_cache_max_entries` knob (default 1024, the current constant).

**D5 — Equivalence is enforced by construction and by test.** Because caches only mirror SQL state, fold-vs-fresh equivalence is testable directly: property/unit tests learn message sequences and compare folded cache contents against cold reads after every step (the eviction spec's equivalence scenario). On top: `generation_hash` byte-identity run (prod snapshot, both knob positions) is a task-level acceptance gate.

**D6 — Telemetry counters live on the generator instance; `/stats` reads them via DI.** A small `GenerationTelemetry` accumulator (hits/misses per cache class, entropy sums, shadow counters) owned by `MarkovGenerator`, injected into the `/stats` handler through the existing `dp["generator"]` dependency. Process-lifetime, reset on restart — matches TZ §20's `/stats` list without new storage. Strings in `bot_messages.py` (Russian, as all user-facing text).

**D7 — Knobs.** `markov_cache_incremental` (bool, default true) and `markov_shadow_order4_enabled` (bool, default true) are registry entries, runtime-settable via `/set`. *Implementation note:* `MARKOV_CACHE_MAX_ENTRIES` (default 1024, ≥64) became an env-only setting alongside `TEXT_CACHE_MAX_MESSAGES`/`SQLITE_*` instead of a registry knob — a process cache bound is fixed at generator construction; making it runtime-mutable would add resize plumbing with no operational payoff. All three documented in `.env.example`.

**D8 — Eval config.** The matrix gains no new C-config: Phase 1 changes no content behavior. Acceptance run = full protocol on the frozen snapshot; content metrics must equal C0 byte-for-byte; latency section (p50/p95, hit-rate) is the phase's measured result and lands in the phase eval report.

## Risks / Trade-offs

- **[Fold bug corrupts a cached distribution]** → equivalence property tests over randomized message sequences; kill-switch knob; `/clear` and LRU eviction still rebuild from SQL; `generation_hash` gate before merge.
- **[Ordering violation on insert]** cached lists must keep SQL `ORDER BY` (sampling relies on it) → bisection insert with an explicit test comparing against a fresh SQL read.
- **[Shadow scan cost on the hot path]** → pre-indexed window (dict lookup per step); knob to disable; latency budget verified by the eval run.
- **[Window estimator misread as ground truth]** → telemetry labels it `estimator=window`; audit and Phase 7 gate documentation repeat the caveat.
- **[Telemetry text leaks]** → numbers/labels only; the log-privacy guard test conventions apply to new lines.

## Migration Plan

No DB changes. Rollout: merge → restart. Rollback: `markov_cache_incremental=false` (runtime, no restart) or revert. Phase 3 note: when `s_value` arrives, cached values become raw `(count, s_value, s_updated_at)` triples per TZ §14 — the fold API is designed to carry a value tuple per token so that change is additive.

## Open Questions

None blocking. Threshold values for the Phase 7 gate are already pre-registered (`eval_thresholds.yaml`); this phase only produces the data.
