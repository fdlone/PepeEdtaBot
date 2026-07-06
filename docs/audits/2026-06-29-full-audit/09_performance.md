# 09 — Performance

> Independent audit, source-only + live query-plan evidence. Looks for blocking I/O in async, sync-in-async, repeated/N+1 queries, allocations, caching/batching, locking, startup/shutdown latency. Impact estimated for the actual workload (a single-instance, long-poll Telegram bot). No production code modified.
>
> Cross-refs: lock/concurrency model in [10_async_review.md](10_async_review.md); index/query detail in [07_database.md](07_database.md); admin-call caching links to [08_security.md] S4.

## 0. Summary

For its workload the bot is **comfortably fast**: message-driven, no fan-out, hot paths are indexed and cached. There are no blocking calls in the request path, the write path is properly batched, and two layers of in-memory LRU caching cover generation. The performance ceiling is **architectural, not hot-loop**: a single SQLite connection behind a single `asyncio.Lock` serializes all DB work (P1). The two concrete hot-path costs are the per-message double aggregation (P2) and redundant-index write amplification (P4). Everything else is minor.

| # | Finding | Severity | Type |
|---|---|---|---|
| P1 | Single global lock + single connection serializes all DB I/O | Medium (scalability) | concurrency bottleneck |
| P2 | Double `SUM(cnt)` per learned message (= [07] D2) | Medium | repeated query / scaling |
| P3 | Per-step transition queries during generation (cold cache) | Low–Medium | N+1-ish (cache-mitigated) |
| P4 | 8 redundant indexes maintained on every write (= [07] D1) | Medium | write amplification |
| P5 | `get_chat_administrators` uncached per admin command (= [08] S4) | Low | network latency / API rate |
| P6 | `ResponseGenerator` instantiated per message | Low | allocation |
| P7 | Blocking `path.read_text` during migrations at startup | Negligible | startup latency |

## 1. What is already good (evidence)

- **No blocking I/O in the request path.** Grep for `time.sleep`, `requests.`, sync `open`, etc. in `app/` returns only `migrator.py:82` (startup, §P7). Typing delay uses `await asyncio.sleep` (`handlers/_helpers.py:24`).
- **Batched writes.** `save_message_and_update_model` uses `executemany` + `ON CONFLICT DO UPDATE` for `transitions{,3,1}` and a single `commit()` per message (`database.py:122-179`) — no per-token round-trips.
- **Two-tier caching.** `MarkovGenerator._cache3` (LRU 1024) caches per-state transitions; `ContextStateMatcher` caches per-chat state indexes (LRU 128). Both invalidated on write (`invalidate_chat_cache`). Warm generation is mostly cache hits.
- **Indexed hot lookups.** `EXPLAIN QUERY PLAN` confirms the n-gram lookups use the PK index (see [07] §3) — O(log n), not scans.

## 2. Findings

### P1 — Global lock + single connection serialize all DB access · **Medium (scalability)**
`Database` holds one `aiosqlite` connection and one `asyncio.Lock` shared into every repository (`database.py:23,46-49`); every read and write does `async with self._lock`. Under concurrent updates from many chats, all DB operations queue behind this single lock — WAL's concurrent-reader benefit is unused because there is only one connection.
- **Impact:** caps throughput to one-DB-op-at-a-time. Fine for current single-instance, modest-traffic use; it is the dominant ceiling if traffic grows. Matches the M1 single-instance observation.
- **Fix (only if needed):** separate read connection(s) for read-only queries (WAL allows concurrent readers), or move to a connection pool / `WITHOUT ROWID` tuning. Not urgent. Confidence: **High** on the mechanism; impact depends on load. See [10] for correctness (the lock is correct, just coarse).

### P2 — Double `SUM(cnt)` aggregation on every learned message · **Medium**
Each `save_message_and_update_model` runs two `SELECT COALESCE(SUM(cnt),0) … WHERE chat_id=?` (`database.py:162-177`) to compute readiness "volume". Cost is O(rows-per-chat) and grows as a chat's model grows; it runs on the hottest write path (every learned message). Same root as [07] **D2**.
- **Fix:** maintain volume incrementally (the upserts already know the deltas) or compute it lazily only where `MIN_TOKENS_FOR_MODEL` is checked. Confidence: **High**.

### P3 — Per-step transition queries during generation · **Low–Medium**
Generation walks the chain calling `get_transitions*(chat_id, …)` per step. On a **cold** cache (e.g., right after a write invalidated the chat), best-of-N generation issues many small indexed queries, each taking the global lock. The LRU cache makes steady state cheap, so this is a warmup cost, not a steady-state N+1.
- **Fix (optional):** for very active chats, preload the chat's transitions once per generation instead of per-step. Confidence: **Medium** (depends on cache hit rate under real traffic).

### P4 — Redundant indexes = write amplification · **Medium**
8 secondary indexes duplicate PK-index prefixes and are never chosen by the planner (proven in [07] **D1**), yet are maintained on every INSERT/upsert. Dropping them speeds the write path with zero read cost.

### P5 — Uncached `get_chat_administrators` per admin command · **Low**
`admin_or_owner.py:24` makes a live Telegram API call on every non-owner admin-gated command. Adds network latency and consumes API rate. Cooldowns bound abuse; a short-TTL per-chat cache removes the repeated call. (= [08] S4.)

### P6 — `ResponseGenerator` per-message instantiation · **Low**
A new `ResponseGenerator` is built per incoming message (noted in [05]). Cheap objects, but on the hottest path it is avoidable allocation; it could be constructed once and reused with per-call request state. Confidence: **Medium** on the allocation, **Low** on impact.

### P7 — Blocking migration read at startup · **Negligible**
`migrator.py:82` `path.read_text` is synchronous, but runs once at boot over tiny migration files. No action needed (cosmetic `pathlib`/async note in [11]).

## 3. Startup / shutdown latency

- **Startup:** connect → PRAGMAs → run pending migrations → instantiate repos → `cleanup_pivo_daily_usage` (one bounded DELETE) → `get_me`/`delete_webhook`/`set_my_commands` (a few Telegram API calls) → `start_polling`. All linear and fast; dominated by the Telegram round-trips, not local work.
- **Shutdown:** `finally: db.close(); bot.session.close()` (`main.py:122-126`). Clean and prompt; no flush/drain loops needed (each message is already committed). No lingering tasks to await (see [10]).

## 4. Recommendations (prioritized)

| Priority | Item | Effort | Payoff |
|---|---|---|---|
| **P2** | P4 — drop redundant indexes (= [07] D1) | S | Faster writes, smaller DB |
| **P2** | P2 — incremental/lazy volume (= [07] D2) | M | Removes scaling write cost |
| **P3** | P5 — cache chat-admin lookups | S | Lower latency, API-rate safety |
| **P3** | P3 — optional per-generation transition preload | M | Smooths cold-cache spikes |
| **P4** | P6 — reuse `ResponseGenerator` | S | Minor allocation |
| **—** | P1 — read connection / pool | L | Only if traffic outgrows single-instance |

No premature optimization warranted: the design is correct and the hot paths are cached/indexed. Focus on D1/D2 (cheap, real) before anything architectural.
