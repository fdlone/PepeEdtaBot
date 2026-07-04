# 19 — Long-Term Strategy

> Independent audit, consolidation milestone. The forward-looking view: where the architecture should hold, where it will strain, and how to evolve it without losing the qualities that make it good today. Strategic (quarters), not tactical (the tactical plan is [16_refactoring_plan.md]). No production code modified.

## 0. Thesis

PepeEdtaBot is a **small, sharp, single-purpose system** — a long-poll Telegram bot with a Markov generation core over SQLite. Its long-term value is in **staying small and correct**, not in growing a platform. The strategy is therefore: **defend the current quality with tooling, harden the few soft spots, and add complexity only when a concrete need (scale, feature, monitoring) forces it.** Resist speculative generality (YAGNI) — the codebase already models this well.

## 1. Architectural invariants to preserve

These are the load-bearing decisions that make the system maintainable; treat changing them as a deliberate, reviewed event:

1. **Layered, Telegram-agnostic core** ([03]/[05]): `core/` generation has no aiogram dependency; handlers are thin. Keep generation testable in isolation.
2. **Registry as single source of truth** ([13]): every runtime-tunable field is one spec line that flows to `Settings`, `RuntimeState`, and `/set`. New tunables must go through it — never add a parallel config path.
3. **No spawned concurrency** ([10]): request-driven only. If background work is ever needed, it must be a tracked, cancellable task — not fire-and-forget.
4. **Privacy by construction** ([08]/[14]): no message text stored (`messages.text` dropped), `author_id` anonymized, `chat_id` masked in logs, `/pivo` opt-in + encrypted. Every new feature must clear this bar.
5. **Atomic, batched DB writes** ([07]): one transaction per learned message. Keep it.

## 2. Where the system will strain (and the planned response)

| Pressure | Symptom | Strategic response |
|---|---|---|
| **Traffic growth** | Lock contention on the single connection ([09] P1) | Introduce WAL read connections / pool **only then**; the write path stays single-writer (SQLite-appropriate). |
| **Data growth per chat** | Slower writes (per-write SUM, index maintenance) | Land D1 (drop redundant indexes) + D2 (incremental volume) early — these are cheap now and compound later. |
| **Multi-instance / HA desire** | In-memory throttle/cooldown/cache diverge ([10] A4) | Externalize that state (shared store) as a *deliberate* migration; until then, enforce single-instance and document it (QW8). |
| **Operational maturity** | "Why is it slow/quiet?" hard to answer ([14] L3/L4) | Add counters + liveness in Phase 5; consider structured logs only if a log aggregator is adopted. |
| **Generation feature velocity** | `markov.py` complexity slows safe edits ([11], R8) | Keep extracting pure sub-steps under characterization tests; never let a single function regrow past ~C901 15. |

## 3. Storage trajectory

SQLite is the **right choice** for a single-instance bot and should remain so for the foreseeable future — it is embedded, transactional, WAL-capable, and the data model is a set of per-chat counter tables that index well. **Do not migrate to a client/server DB** unless multi-instance HA becomes a hard requirement; that decision should be driven by the topology need (R4/A4), not by data size alone. If it ever comes, the repository layer ([05]) already isolates SQL, so the blast radius is contained.

## 4. Dependency & supply-chain posture

- Keep the runtime dependency set **minimal** (currently 4 direct: aiogram, aiosqlite, python-dotenv, cryptography). Each addition is a supply-chain liability ([08]).
- Make the CI dependency gate real (QW2) and keep it green; periodically refresh `requirements.lock` per its documented procedure.
- Consider **hash-pinning** the lock (`pip-compile --generate-hashes` or `uv`) if/when supply-chain assurance matters more than the current simple `pip freeze` flow ([06] note). Not urgent for a hobby-scale bot.

## 5. Testing & quality strategy

- The 366-test suite is the project's primary asset. **Measure** it (T3) so breadth becomes a defended baseline; consider a light coverage ratchet so new code arrives with tests.
- Keep `unittest` (no pytest dependency) — it works, runs everywhere, and avoids a dependency. The presence of `pytest.exe` is incidental; don't formalize a pytest migration without a reason.
- Tie any DB schema change to an `EXPLAIN QUERY PLAN`/correctness test (T4) so index/aggregation work can't silently regress.

## 6. 3-horizon summary

- **Now (this quarter):** Phase 1–2 of [16] — close R1/R2, restore strict typing, drop redundant indexes, tame the two complex functions. Low risk, high hygiene return.
- **Next (1–2 quarters):** Phase 3 + 5 — incremental volume, admin caching, coverage in CI, basic metrics. Makes the bot observable and keeps writes flat as data grows.
- **Later (only if triggered):** Phase 6 — read concurrency / multi-instance. Driven by a real scaling decision, with the repository layer and documented invariants as the safety rails.

The north star: **a bot a single engineer can fully understand, run, and evolve confidently** — which the current codebase nearly achieves and the audit docs ([01]–[20]) are designed to sustain.
