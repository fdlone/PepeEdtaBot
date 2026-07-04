# 07 — Database

> Independent audit, source-only + live-schema introspection of the local `markov.db` (authoritative for indexes/query plans). Covers schema, models, migrations, indexes, constraints, transactions, connection lifecycle, query efficiency. No production code or data modified.
>
> Cross-refs: write/lock concurrency in [10_async_review.md](10_async_review.md); query cost in [09_performance.md](09_performance.md); the `Database` god-object note in [11_code_quality.md](11_code_quality.md).

## 0. Summary

The data layer is a **single SQLite database accessed through one `aiosqlite` connection**, fronted by a `Database` facade (`app/infrastructure/database.py`) that owns the connection, an `asyncio.Lock`, the migrator, and four repositories (markov / messages / chat_members / pivo_usage). It is **WAL-mode, foreign-keys=ON, fully parameterized, and well-indexed for its lookup patterns** — the query planner uses the primary-key indexes for the hot n-gram lookups (verified by `EXPLAIN QUERY PLAN`).

Two concrete findings: **D1 — ~7 redundant secondary indexes** that the planner never uses (pure write amplification), and **D2 — two `SUM(cnt)` full-chat aggregations on every learned message** whose cost grows with the per-chat model. Both are write-path efficiency issues, not correctness bugs.

## 1. Engine & connection model

- **Engine:** SQLite via `aiosqlite` (async wrapper running sqlite calls on a per-connection worker thread).
- **One connection per process** (`Database._conn`), opened in `init()` (`database.py:38`), with `PRAGMA journal_mode=WAL` and `PRAGMA foreign_keys=ON` (`:41-42`).
- **No pool** — single connection by design. All access is serialized by a single `asyncio.Lock` shared into every repository (`:23,46-49`). See [10_async_review.md] for the concurrency analysis and [09_performance.md] **P1** for the throughput implication.
- **Lifecycle:** `init()` → connect → PRAGMAs → `migrator.run()` → instantiate repos → `cleanup_pivo_daily_usage()`. `close()` closes the connection and nulls the repos (`:52-59`). Wired from `main.py` (`init` before polling; `close()` in the `finally` of `run_bot`). Graceful.

## 2. Schema (live `markov.db`, reflects all 7 migrations)

| Table | Key | Purpose |
|---|---|---|
| `messages` | `id` INTEGER PK (rowid) | Raw learned messages. `text` column was **dropped** (migration 005, privacy); `normalized_text` retained; `author_id` anonymized to 0 (migration 003). |
| `starts` | PK(`chat_id,w1,w2`) | Order-2 sentence-start bigrams + `cnt`. |
| `starts3` | PK(`chat_id,w1,w2,w3`) | Order-3 sentence-start trigrams + `cnt`. |
| `transitions` | PK(`chat_id,w1,w2,w3`) | Order-2 transition counts (state=w1,w2 → w3). |
| `transitions3` | PK(`chat_id,w1,w2,w3,w4`) | Order-3 transition counts. |
| `transitions1` | PK(`chat_id,w1,w2`) | Order-1 fallback transition counts. |
| `chat_members` | PK(`chat_hash,user_hash`) | `/pivo` members (HMAC-hashed keys, Fernet-encrypted PII). Renamed from `pivo_chat_members` in migration 007. |
| `pivo_daily_usage` | PK + `usage_day` | `/pivo` per-day quota counters (added migration 006). |
| `schema_migrations` | PK(name) | Applied-migration ledger. |

**Models:** there is no ORM. Rows are read into plain tuples and adapted in the repositories (`str(...)`/`int(...)` coercion). This is lightweight and appropriate; it also means no lazy-loading / N+1 ORM traps.

**Constraints:** composite PKs, `NOT NULL`, sensible `DEFAULT`s (`cnt=0`, `created_at=datetime('now')`). `foreign_keys=ON` is set but **no FK relationships are declared** — `chat_id`/`chat_hash` are logical keys, not enforced references. For an append-only counter store this is acceptable; cross-table integrity is maintained by the single atomic write path (§4).

## 3. Indexes — finding D1 (redundant secondary indexes) · **Medium** (write amplification)

Live `sqlite_master` shows, **on top of** the implicit PK indexes (`sqlite_autoindex_*`), these explicit indexes:

| Index | Columns | PK on table | Verdict |
|---|---|---|---|
| `idx_starts_lookup` | (chat_id,w1,w2) | (chat_id,w1,w2) | **Exact duplicate of PK** → redundant |
| `idx_starts_chat_id` | (chat_id) | (chat_id,w1,w2) | Prefix of PK → redundant |
| `idx_starts3_chat_id` | (chat_id) | (chat_id,w1,w2,w3) | Prefix of PK → redundant |
| `idx_transitions_lookup` | (chat_id,w1,w2) | (chat_id,w1,w2,w3) | Prefix of PK → redundant |
| `idx_transitions3_lookup` | (chat_id,w1,w2,w3) | (…,w4) | Prefix of PK → redundant |
| `idx_transitions1_lookup` | (chat_id,w1) | (chat_id,w1,w2) | Prefix of PK → redundant |
| `idx_chat_members_chat_hash` | (chat_hash) | (chat_hash,user_hash) | Prefix of PK → redundant |
| `idx_messages_chat_id` | (chat_id) | rowid | Prefix of `idx_messages_normalized_lookup` → redundant |
| `idx_messages_normalized_lookup` | (chat_id,normalized_text) | rowid | **Keep** (only index serving verbatim-copy checks) |

**Evidence (`EXPLAIN QUERY PLAN` on the hot read):**
```
SELECT w3,cnt FROM transitions WHERE chat_id=? AND w1=? AND w2=? ORDER BY w3
→ SEARCH transitions USING INDEX sqlite_autoindex_transitions_1 (chat_id=? AND w1=? AND w2=?)
```
The planner picks the **PK autoindex**, not `idx_transitions_lookup`. SQLite serves leftmost-prefix lookups from the composite PK index, so every index in the table above (except `idx_messages_normalized_lookup`) is dead weight that is still maintained on every INSERT/upsert.

- **Impact:** each learned message updates several duplicate b-trees for zero read benefit. Modest at current scale (hundreds of rows) but unbounded write overhead as chats grow.
- **Fix:** drop the 8 redundant indexes in a new migration; keep PK autoindexes + `idx_messages_normalized_lookup`. Verify with `EXPLAIN QUERY PLAN` for each repo query first. Confidence: **High** (planner output is authoritative).

## 4. Transactions

- **Atomic write path:** `save_message_and_update_model` (`database.py:61-180`) acquires the lock once, performs the message INSERT + all `starts/starts3/transitions{,3,1}` upserts (batched with `executemany` + `ON CONFLICT DO UPDATE`), then the two volume SUMs, then a single `commit()`. The whole model update for one message is **one transaction** — correct and crash-safe.
- **Migrations:** `migrator.run` wraps each migration in `executescript("BEGIN;\n{sql}\nCOMMIT;")` (`migrator.py:85`) and records it in `schema_migrations` — idempotent and transactional. (Minor: a blocking `path.read_text` at `migrator.py:82` during startup — negligible, one-shot; see [09] P7.)
- **Autocommit elsewhere:** read methods don't open explicit transactions (fine). `clear_chat` issues 6 DELETEs under the lock; it does **not** wrap them in an explicit `commit()`-guarded block in the snippet — verify it commits (aiosqlite autocommits outside a transaction, so the DELETEs persist, but bundling them in one transaction would be cleaner). Low.

## 5. Query efficiency

- **Hot lookups are indexed** (PK autoindex), O(log n) — good. The generator further caches per-state transitions in an in-memory LRU (`markov.py` `_cache3`, limit 1024) and the context matcher caches per-chat state indexes (LRU 128), so warm generation is mostly cache hits. See [09] P3 for the cold-cache per-step query pattern.
- **D2 — double `SUM(cnt)` per write** · **Medium**: `save_message_and_update_model` runs `SELECT COALESCE(SUM(cnt),0) FROM transitions3 …` and `… FROM transitions …` (`database.py:162-177`) on **every** learned message to compute the readiness "volume". This is an O(rows-per-chat) range-scan-and-sum that grows with the model; `get_stats` repeats the pattern across 9 counts/sums. **Fix options:** maintain a per-chat volume counter incrementally, or compute volume lazily only when needed (it gates `MIN_TOKENS_FOR_MODEL`). Confidence: **High**.
- `get_states` (markov_repo) does a `GROUP BY … SUM(cnt)` full-chat scan to build the context-matcher index — acceptable because it is cached (LRU 128) and invalidated on write.

## 6. Recommendations

| Priority | Item | Effort | Note |
|---|---|---|---|
| **P2** | D1 — migration dropping the 8 redundant indexes | S | Verify each query's plan first; pure write-path win |
| **P2** | D2 — incremental/lazy per-chat volume instead of per-message double SUM | M | Scales write cost with model size today |
| **P4** | wrap `clear_chat` DELETEs in one explicit transaction | XS | Cleanliness |
| **P4** | `pathlib`/non-blocking migration read (cosmetic) | XS | One-shot startup only |

Schema design, migration discipline, parameterization, and WAL are all sound — no structural rework needed. The findings are index hygiene and one hot-path aggregation.
