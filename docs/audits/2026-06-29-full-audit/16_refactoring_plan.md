# 16 — Refactoring Plan (Phased Roadmap)

> Independent audit, consolidation milestone. A phased, dependency-ordered roadmap that sequences every actionable finding from [07]–[15]. Phases follow the TZ structure (critical → architecture → performance → security → DX → scalability) but are ordered by **value-per-effort and risk**, not dogmatically. No production code modified during the audit.
>
> Cross-refs: same-day items in [18_quick_wins.md](18_quick_wins.md); debt table in [15_technical_debt.md](15_technical_debt.md); exposure framing in [17_risk_register.md](17_risk_register.md).

## Guiding principles

1. **Preserve what works** — the registry pattern, privacy logging, atomic write path, and no-spawned-tasks async model are assets ([15] §3). Do not "refactor" them.
2. **Tests are the safety net** — 366 passing tests gate every change; add tests *before* the riskier extractions (markov, volume).
3. **Cheap risk-closers first** — the only Medium-exposure risks (R1/R2) are XS–S effort.

---

## Phase 1 — Critical fixes & hygiene (≈1.5 days) → all of [18]
Close the real (if low-severity) exposures and re-enable safety nets. **No architectural change.**

- **Security/config:** QW2 (CI `pip-audit` gate), QW3 (reject placeholder/equal secrets) — closes R1, R2.
- **Type safety:** QW1 (type `state`, drop `ignore_errors`) — restores strict on 10 modules.
- **Hygiene:** QW4 (log swallowed exception), QW5 (resolve dead `get_random_pivo_message` — ✅ done 2026-07-07, removed with `build_pivo_mentions` + tests), QW6 (unify logger name), QW8 (document single-instance invariant).
- **DB write-path (cheap half):** QW7 (drop 8 redundant indexes via migration `008`).

**Exit criteria:** CI trio green; `mypy app/` strict with no overrides; `pip-audit` gate active; new `test_settings` rejections + `test_migrator` plan assertion pass.

## Phase 2 — Architecture & maintainability cleanup (≈3–5 days)
Reduce friction in the two hot spots without behavior change.

- **`database.py` god-object ([11] Q6, [07] §5):** extract a single `_require_init()` guard (collapses ~12 repeats); optionally split the facade into connection / markov-store / message-store responsibilities behind the same public API.
- **`markov.py` complexity ([11], R8):** extract pure sub-steps from `_generate_text_once` (C901=36) and `_select_contextual_state` (29). **Add characterization tests first**, then refactor under green.
- **DRY:** QW-adjacent — consolidate `parse_bool` ([11] Q7) to the registry parser; inject an RNG into `pivo_message_builder`/`_helpers` ([11] Q9) so tests stop monkeypatching the global.
- **Config validation:** C3 cross-field checks (`backoff_min_order < markov_order`, reply-context bounds) in `registry.validate_cross_fields` ([13]).

**Exit criteria:** complexity counts down (target C901 ≤ ~15 on the two functions); behavior identical (tests unchanged & green); duplication removed.

## Phase 3 — Performance improvements (≈2–3 days)
Only the items that scale poorly with data; everything else is already fast ([09]).

- **D2/P2 — incremental volume:** stop the per-message double `SUM(cnt)`; maintain a per-chat volume counter from the upsert deltas, or compute lazily at the `MIN_TOKENS_FOR_MODEL` check. Add a volume-correctness test ([12] T4).
- **P5/S4 — cache chat-admin lookups** (short TTL per chat) to cut latency + API-rate use.
- **P3 (optional) — per-generation transition preload** for very active chats if cold-cache spikes show up in metrics.
- **P6 — reuse `ResponseGenerator`** instead of per-message instantiation.

**Exit criteria:** write cost no longer grows with model size; admin commands no longer make a Telegram call each time.

## Phase 4 — Security hardening (≈1 day)
Beyond the Phase-1 quick fixes, the consistency items.

- **S3 — HKDF for the Fernet key** in `PivoSecurity`, mirroring `log_masking`'s HKDF use (consistency; not urgent).
- Re-run the M3 tooling (`bandit`, `pip-audit -r`, `uvx semgrep`) as a regression gate after Phases 1–3.

**Exit criteria:** key derivation consistent; security tooling clean on the changed code.

## Phase 5 — Developer experience & observability (≈2–4 days)
Turn "good by inspection" into "defended by tooling".

- **T3 — coverage in CI:** `coverage run -m unittest` + report artifact (and optionally a threshold ratchet).
- **T1/T2 — fill test gaps:** direct `registry` round-trip tests; `throttling` TTL/overflow pruning tests.
- **L3 — basic metrics:** counters for messages learned / replies generated / rejects / `/pivo` calls / throttle drops (map directly from the existing `GenerationTrace`). Optional JSON log formatter for aggregators.
- **L4 — liveness** tied to last successful update.

**Exit criteria:** coverage measured and visible; the two gap modules unit-tested; basic runtime counters available.

## Phase 6 — Future scalability (only when warranted)
Defer until traffic/topology demands it ([09] P1, [10] A4).

- **P1 — read concurrency:** dedicated read-only connection(s) leveraging WAL concurrent readers, or a connection pool.
- **Multi-instance:** externalize the in-memory throttle/cooldown/cache state (e.g., shared store) — only if moving beyond single-instance; honor the invariant documented in QW8 until then.

**Trigger:** sustained lock contention or a decision to run multiple workers/instances.

---

## Sequencing dependencies
- Phase 1 is independent and unblocks confidence for everything else (strict typing + active gates).
- Phase 2 `markov` extraction should precede Phase 3 perf changes touching the same code.
- Phase 6 depends on a real scaling decision; do not pre-build it.

## Effort rollup
| Phase | Focus | Effort | Risk |
|---|---|---|---|
| 1 | Critical/hygiene/quick wins | ~1.5 d | Very low |
| 2 | Architecture/maintainability | 3–5 d | Low (test-gated) |
| 3 | Performance (scaling items) | 2–3 d | Low |
| 4 | Security consistency | ~1 d | Very low |
| 5 | DX & observability | 2–4 d | Very low |
| 6 | Scalability | — | Deferred |

Phases 1 + 4 alone resolve all Medium-exposure risks ([17]). Phases 2–3 are the substantive engineering; Phase 5 is the long-term investment ([19]).
