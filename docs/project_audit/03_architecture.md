# 03 — Architecture

> Independent audit, source-only. See [04_execution_flow.md](04_execution_flow.md) for runtime lifecycles, [06_dependency_graph.md](06_dependency_graph.md) for the import graph, [05_module_inventory.md](05_module_inventory.md) for module APIs.

## High-level shape

A **single-process async application**. There is no web server, no scheduler,
no background worker pool, no message queue. One asyncio event loop runs an
aiogram long-polling `Dispatcher`; every Telegram update is dispatched to a
handler, which calls into services → repositories → a single shared SQLite
connection.

```
Telegram ──long poll──▶ aiogram Dispatcher
                          │  (ThrottlingMiddleware on message)
                          ▼
        ┌─────────── Routers (handlers) ───────────┐
        │ common │ admin │ pivo │ learning │ errors │
        └───┬────────┬───────┬────────┬─────────────┘
            │        │       │        │
            ▼        ▼       ▼        ▼
                 Services / Core
   LearningService   PivoService   ResponseGenerator
   MarkovGenerator   (PivoMessageBuilder, PivoParser)
            │        │       │
            ▼        ▼       ▼
              Repositories (per domain)
   MarkovRepo  MessagesRepo  ChatMembersRepo  PivoUsageRepo
            │        │       │        │
            └────────┴───┬───┴────────┘
                         ▼
              Database facade (app/infrastructure/database.py)
                  one aiosqlite.Connection + one asyncio.Lock
                         ▼
                    SQLite (WAL)
```

## Layers and their boundaries

| Layer | Packages | Allowed to depend on | Telegram-aware? |
|---|---|---|---|
| Composition root | `main.py` | everything | yes (builds Bot/Dispatcher) |
| Presentation/handlers | `app/handlers`, `app/presentation`, `app/filters`, `app/middlewares` | services, core, config, infrastructure types | **yes** (aiogram `Message`, `Router`) |
| Services | `app/services` | core, domain, repositories (via `Database`) | mostly no (PivoService takes aiogram `User`) |
| Core | `app/core` | infrastructure (`Database`), config (`RuntimeState`) | no |
| Domain | `app/domain` | `cryptography`, services (lazy import) | partial (parser/service touch aiogram types) |
| Repositories | `app/repositories` | `aiosqlite` only | no |
| Infrastructure | `app/infrastructure` | repositories, core/text, migrator | no |
| Config | `app/config` | dotenv, registry | no |

The intended dependency direction is **inward**: handlers depend on services,
services on repositories, repositories on the DB connection. This is mostly
respected. Known boundary observations (carried into [11_code_quality.md] M2):

- **`app/core/markov.py` imports `app/infrastructure/database.py`** (`markov.py:13`).
  Core depends on infrastructure — but only on the `Database` *facade type* and
  its read methods, used as a repository port. Pragmatic, not a cycle, but it
  couples the "pure" generation layer to the concrete DB facade.
- **`app/infrastructure/database.py` imports `app/core/text.py`** (`database.py:10`)
  for `sanitize_text` on write. So core↔infrastructure import edges exist in
  both directions (different modules, so no true import cycle, but the layering
  is not strictly acyclic at package granularity). Confirmed acyclic at module
  granularity in [06_dependency_graph.md](06_dependency_graph.md).
- **`app/domain/pivo.py:119`** lazily imports `app/services/pivo_message_builder`
  inside a function — a deliberate cycle-break (domain↔services). The function
  `get_random_pivo_message` appears unused by handlers (PivoService calls the
  builder directly); flagged as possible dead code in [11_code_quality.md] M2.

## Dependency injection

There is no DI framework. `main.py:run_bot` constructs all singletons and
publishes them into aiogram's **workflow data** dict
(`dp["db"] = db`, etc., `main.py:49-57`). aiogram injects them into handler
parameters by name. Filters receive `bot`/`settings` the same way
(`AdminOrOwner.__call__(self, message, bot, settings)`,
`app/filters/admin_or_owner.py:32`). This is clean and testable (tests build
the same objects directly), at the cost of stringly-typed keys.

Lifetimes:
- **Singletons (process-lifetime):** `Database`, `MarkovGenerator`,
  `PivoService`, `LearningService`, `RuntimeState`, `Settings`,
  `ThrottlingMiddleware`. All shared across chats and updates.
- **Per-request:** `ResponseGenerator` is instantiated **per message** inside
  the learning handler (`app/handlers/learning.py:231`) — it is a thin
  stateless wrapper around the shared generator/service/state, so this is cheap
  but slightly wasteful (noted in [09_performance.md] M4).

## State management

Mutable in-memory state lives in two places, both with TTL + max-size pruning:

1. **`RuntimeState`** (`app/config/runtime_state.py`) — runtime-mutable config
   values (mirrors the registry) plus per-chat ephemeral state:
   `last_reply_ts`, `learned_messages`, `recent_short_replies` (deque),
   `_last_chat_activity`. Pruned by `note_chat_activity` every 64 ticks or on
   overflow (`runtime_state_max_chats`, TTL `runtime_state_ttl_sec`).
2. **`ThrottlingMiddleware._last_used`** — `(chat,user,command) → ts`, pruned
   every 64 ticks / on overflow (`throttle_state_max_keys`, TTL).
3. **Caches:** `MarkovGenerator` holds LRU `OrderedDict` caches for
   transitions/starts (`cache_limit=1024`); `ContextStateMatcher` caches a
   per-chat state index; `LearningService._text_cache` caches recent normalized
   messages per chat. All invalidated on new messages / `/clear`
   (`learning_service.py:41-42`, `markov.py:525`).

All state is **process-local** — the design assumes a **single bot instance**.
Running two replicas would diverge throttling/quota-by-time and duplicate
replies; the per-day `/pivo` quota is the only DB-backed limiter and would stay
consistent. Scalability implications in [19_long_term_strategy.md] (M6).

## Concurrency model

- One aiogram event loop. Handlers are coroutines.
- **All DB access is serialized** through a single `asyncio.Lock` shared by the
  `Database` facade and every repository (`database.py:23`, passed into each
  repo). Only one SQLite operation runs at a time; there is no connection pool.
  This trades throughput for simplicity and correctness on a single connection.
  See [10_async_review.md] (M4) for re-entrancy/locking analysis (e.g.
  `save_message_and_update_model` holds the lock across many statements).
- `reply_humanized` sleeps a randomized "typing" delay before replying
  (`_helpers.py:23`), serializing nothing but adding per-reply latency by design.

## Error handling

- A global aiogram error router logs and swallows handler exceptions
  (`app/handlers/errors.py`). `/pivo` adds compensating logic: it refunds the
  daily quota if delivery fails after consumption (`handlers/pivo.py:78-92`).
- Startup validation is fail-fast: `load_settings` raises `ValueError` on bad
  config; `mask_chat_id` raises if `init_masking` was not called.

## Security architecture (summary; full review in M3)

- **Secrets** (`BOT_TOKEN`, `PIVO_HMAC_SECRET`, `PIVO_ENCRYPTION_SECRET`,
  `OWNER_ID`) come only from env; secrets are min-length validated and never
  runtime-mutable (`registry.py` docstring, `settings.py:90-95`).
- **PII redaction** in `core/text.sanitize_text` → `core/privacy_filter`
  (emails/phones/secrets) runs before any text is persisted, cached, or used as
  context.
- **`/pivo`** stores HMAC-SHA256 hashes as keys and Fernet ciphertext as
  payload; mentions are HTML-escaped before `parse_mode=HTML` send.
- **Authorization**: `/set`, `/setprob`, `/clear` require `AdminOrOwner`;
  `/pivo` enforces per-day quotas; bots rejected from `/pivo_on`.

## Deployment topology

Single container (`python:3.14-slim`), long-poll outbound to Telegram, SQLite
on a bind-mounted `./data` volume. Runs as non-root `bot` (UID 1000) after an
entrypoint chowns the data dir. Liveness healthcheck only opens the SQLite file
(`Dockerfile:35`). No external services. Details + ops in
[14_logging.md]/[13_configuration.md] and `docs/OPERATIONS.md`.
