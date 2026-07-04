# 10 — Async Review

> Independent audit, source-only. Checks asyncio usage, task cancellation/leaks, race conditions, shared mutable state, locks, background workers, scheduling, and graceful shutdown. No production code modified.
>
> Cross-refs: lock throughput in [09_performance.md](09_performance.md) P1; write transaction in [07_database.md](07_database.md) §4; execution flow in [04_execution_flow.md](04_execution_flow.md).

## 0. Summary

The async model is **simple and correct**. The bot is purely **request-driven through the aiogram dispatcher** — there are **no custom `create_task`, no background workers, no schedulers, no queues**. Everything runs inside one event loop; the only concurrency primitive is a single `asyncio.Lock` guarding the one SQLite connection. Because there is no spawned concurrency and all shared mutable state is touched only on the loop, there are **no task leaks, no cancellation gaps, and no data races**. The lock is coarse-grained (a throughput trade-off, see [09] P1) but **deadlock-free** — verified that the locked write path does not re-enter the non-reentrant lock.

| Aspect | Verdict |
|---|---|
| Task spawning / leaks | None — no `create_task`/`ensure_future`/`gather` in `app/` (A1) |
| Cancellation handling | Delegated to aiogram `start_polling`; clean `finally` shutdown (A2) |
| Locks / deadlock | Single `asyncio.Lock`; no re-entrant acquisition → safe (A3) |
| Shared mutable state | Loop-confined; safe today, fragile under future parallelism (A4) |
| Graceful shutdown | DB + bot session closed in `finally` (A5) |
| Blocking-in-async | None in request path (see [09]); one startup read (negligible) |

## 1. Async architecture

`main.run_bot` builds singletons, then `await dp.start_polling(bot)` (`main.py:121`). aiogram's dispatcher awaits each update and invokes the matched handler/middleware chain. Handlers (`learning`, `pivo`, `admin`, `common`, `errors`) are coroutines that `await` DB and Telegram calls. There is **no second source of concurrency**: no scheduler, no periodic task, no worker pool. The single periodic-ish action — `cleanup_pivo_daily_usage` — runs once at `init()`, not on a timer.

This means the entire system is cooperatively scheduled on one loop, and the only place two logical operations can interleave is at `await` points.

## 2. Findings / observations

### A1 — No task spawning → no task leaks · **Info (good)**
Grep for `asyncio.create_task`, `ensure_future`, `gather`, `Semaphore`, `Queue`, `to_thread` across `app/` and `main.py` returns **nothing** (only `asyncio.Lock` and `asyncio.sleep`). There are no detached tasks to leak, orphan, or forget to cancel. This is the safest possible posture and should be preserved; if background work is ever added (e.g., periodic `cleanup_pivo_daily_usage`), it must be a tracked task cancelled in shutdown.

### A2 — Cancellation handled by the framework · **Info (good)**
Shutdown/cancellation is owned by aiogram's `start_polling`, which installs signal handlers and stops polling on SIGINT/SIGTERM. The app's `try/finally` (`main.py:120-126`) then runs cleanup. No handler swallows `asyncio.CancelledError`. The one broad `except Exception` (`_helpers.py:25`, [08] S6) is narrow in scope (a chat-action send) and does **not** catch `CancelledError` semantics incorrectly — though logging it would be better.

### A3 — Single lock, no deadlock · **Info (verified)**
One `asyncio.Lock` is created in `Database.__init__` and shared into all four repositories (`database.py:23,46-49`). `asyncio.Lock` is **not reentrant**, so the risk would be a locked method calling another locked method. Verified this does **not** happen:
- Read delegates on the facade (`get_starts`, `get_transitions`, …, `database.py:184-`) only null-check and forward to the repo, which acquires the lock **once**.
- The write path `save_message_and_update_model` (`:89`) acquires the lock itself and uses inline `db.execute`/`executemany` — it does **not** call the lock-taking repo methods.
No nested acquisition path exists → deadlock-free. The lock is held across `fetchall()` (coarse) but result sets are small; correctness over granularity is the right call here.

### A4 — Shared mutable state is loop-confined · **Low (latent)**
Mutable state lives in plain Python containers mutated without locks:
- `RuntimeState` (counters, `last_reply_ts`, `learned_messages`, `recent_short_replies`),
- `ThrottlingMiddleware._last_used` dict + `_cleanup_tick` (`throttling.py:35-36`),
- the two LRU caches (`markov._cache3`, `ContextStateMatcher._cache`).

Today this is **safe**: a single event loop with no `create_task` means no two coroutines mutate these concurrently (mutations happen between `await`s, never preempted mid-statement). The risk is **latent**: if the project ever adds background tasks, multiple workers, or multiple processes, these become unsynchronized shared state. The DB lock would still protect the DB, but in-memory throttle/cooldown/cache state would race or diverge (the single-instance assumption from M1). **Recommendation:** document the single-loop/single-instance invariant near these structures so a future change doesn't silently break it. Confidence: **High** that it's safe now; the note is about future-proofing.

### A5 — Graceful shutdown · **Info (good)**
`finally: await db.close(); await bot.session.close()` (`main.py:122-126`). The DB connection closes cleanly; because every message is committed within its own transaction (§[07] 4), there is no unflushed write buffer to lose on shutdown. No tasks to await/cancel (A1).

## 3. Race-condition analysis

- **DB consistency:** the multi-statement model update is atomic under the lock + single transaction → no torn reads/writes (A3, [07] §4).
- **Cache vs. write:** writes call `invalidate_chat_cache` so a stale cached transition set is dropped after learning. Because invalidate and read both run on the loop and the write holds the DB lock, there is no interleaving that serves stale-then-committed data within a single logical operation. A read that started before a write simply sees pre-write data — acceptable for a probabilistic text model.
- **Throttle dict:** `__call__` reads-then-writes `_last_used[key]` across `await`s, but since only one coroutine runs between awaits and the key is per (chat,user,command), there is no lost-update under the single-loop model. Safe today (A4 caveat applies).

## 4. Recommendations

| Priority | Item | Effort | Note |
|---|---|---|---|
| **P4** | A4 — document single-loop/single-instance invariant by the shared mutable state | XS | Prevents a future regression |
| **P4** | A2/S6 — log the swallowed chat-action exception | XS | = [08] S6 / [11] Q1 |
| **—** | If background work is ever added, make it a tracked, cancellable task | — | Preserve A1/A2 guarantees |

The async layer needs no remediation — it is correct by construction. The only real lever is the coarse DB lock, which is a deliberate simplicity/throughput trade-off covered in [09] P1.
