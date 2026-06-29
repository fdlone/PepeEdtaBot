# 04 — Execution Flow

> Independent audit, source-only. Cross-refs: [03_architecture.md](03_architecture.md), [05_module_inventory.md](05_module_inventory.md). Async/locking risks deferred to [10_async_review.md] (M4).

## Startup sequence (`main.py:run_bot`)

1. `load_settings()` — read `.env`, validate required + cross-field invariants
   (`settings.py:69`, `registry.validate_cross_fields`). **Fail-fast** on bad config.
2. `log_masking.init_masking(settings.pivo_hmac_secret)` — derive HKDF key for
   `chat_id` masking (`main.py:68`). Must precede any `mask_chat_id` call.
3. `logging.basicConfig(...)` at `settings.log_level`; aiogram logger pinned to WARNING.
4. `Database(db_path)` → `await db.init()`:
   - open `aiosqlite` connection, `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`
     (`database.py:38-42`);
   - `migrator.run(db)` applies pending migrations in order (`database.py:44`);
   - construct the four repositories sharing `self._get_conn` + `self._lock`;
   - `cleanup_pivo_daily_usage()` deletes quota rows older than 7 days.
5. Build singletons: `MarkovGenerator(db)`, `PivoSecurity(hmac, enc)`,
   `PivoService(db, security)` + `configure_call_limits(...)`,
   `LearningService(db, generator, ...)`, `runtime_state_from_settings(settings)`.
6. `Bot(token)`; `await bot.get_me()` → `bot_username`, `bot_id`;
   `await bot.delete_webhook(drop_pending_updates=False)` (switch to polling,
   keep queued updates); `set_my_commands(...)` from `TELEGRAM_COMMANDS`.
7. `configure_dispatcher(...)` — register `ThrottlingMiddleware` on
   `dp.message`, publish singletons into `dp[...]`, include routers in order:
   **common → admin → pivo → learning → errors** (`main.py:58-62`).
8. `await dp.start_polling(bot)` — blocks until cancelled.

**Router order matters.** aiogram tries routers/handlers in registration order.
Command handlers (common/admin/pivo) are matched before the catch-all
`learning` router's `F.text` handler, so `/commands` don't fall through to
learning. The `learning` handler additionally early-returns on text starting
with `/` (`learning.py:127`).

## Shutdown sequence (`main.py:120-126`)

`start_polling` exits (cancellation / SIGINT). The `finally` block:
`await db.close()` (close connection, null out repos), `await bot.session.close()`.
`asyncio.run` wrapper swallows `KeyboardInterrupt`/`SystemExit` (`main.py:130-133`).
There is **no explicit signal handling** beyond aiogram's own and no in-flight
drain — graceful-shutdown nuance noted in [10_async_review.md] (M4).

## Telegram update lifecycle (a normal group text message)

This is the hottest path. Handler: `app/handlers/learning.py:on_text_message`.

1. **Middleware** (`ThrottlingMiddleware.__call__`): non-`Message` or non-`/`
   text passes through untouched. Command messages may be throttled
   (only `/clear` has a configured cooldown today; `main.py:24`).
2. **Router matching**: command routers don't match plain text; learning's
   `@router.message(F.text)` handles it.
3. **Guards** (`learning.py:120-128`): must be group/supergroup, have a
   non-bot `from_user`, and not start with `/`.
4. **Activity bookkeeping**: `runtime_state.note_chat_activity(chat_id, now)`
   (may trigger pruning of inactive chats).
5. **Mention detection** *before* learnability: `bot_is_mentioned(...)`
   (`reply_policy.py:27`) — username `@mention`, text alias (`pepe`/`пепе` or
   `BOT_TEXT_ALIASES`), mention entity, or a reply to the bot's own message.
6. **Learnability**: strip a leading "`<alias>, `" vocative
   (`strip_leading_bot_vocative`), `sanitize_text` (PII/link/mention redaction),
   `tokenize`; learnable if length ≤ 500 chars and ≥ 2 tokens.
7. **Gating decisions** (all early-returns):
   - not learnable + not mentioned → skip;
   - mentioned + not enough model data → "поболтайте ещё" reply;
   - not enough data → skip;
   - compute cooldown + `should_reply_to_message(mentioned, cooldown_ok,
     reply_probability, random)` → mentioned always replies; otherwise
     probabilistic and cooldown-gated.
8. **Reply context** (if `use_reply_context`): `extract_context_tokens` from the
   replied-to message and/or current message, truncated to
   `reply_context_max_tokens`. When mentioned without a reply, the current
   message is used as context.
9. **Generation**: build a per-message `ResponseGenerator` and call
   `generate(GenerationRequest(chat_id, context_tokens, seed=None,
   current_message_normalized))`. See best-of-N flow below.
10. **Delivery**: if a reply was produced, set `last_reply_ts`, send via
    `reply_humanized` (typing action + randomized delay + `message.reply`).
    Short replies are remembered to avoid near-term repeats.
11. **Learning (always, in `finally`)**: if learnable,
    `learning_service.record_message(chat_id, learn_source, tokens)` →
    `Database.save_message_and_update_model` updates n-gram tables atomically and
    invalidates caches. Progress logged on the 1st and every 25th message.

### Best-of-N generation (`core/response_generator.py:generate_with_result`)

- Loop up to `GENERATION_ATTEMPT_BUDGET = 10` attempts; collect up to
  `CANDIDATE_TARGET = 5` distinct accepted candidates, then stop.
- First `GENERATION_ATTEMPTS_WITH_CONTEXT = 5` attempts pass the reply context;
  later attempts drop it. Randomness escalates per attempt
  (`escalated_randomness_strength`).
- Each attempt calls `MarkovGenerator.generate_text(..., attempt_budget=1)`.
- Candidate rejected if: equals the current message; is a recently-used short
  reply; or (for longer replies) is a verbatim copy of a training sample
  (`LearningService.is_verbatim_copy`).
- Accepted candidates are scored by `candidate_scorer.score_candidate`
  (completion quality + lexical diversity + natural length + context relevance −
  repetition penalty). The **max-total** candidate wins; optional
  `auto_capitalize_replies` post-processing.

### Single Markov attempt (`core/markov.py:_generate_text_once`)

1. Derive exploration/power knobs from `randomness_strength`.
2. Load per-chat `starts3`/`starts2` (cached). If both empty → reject `no_starts`.
3. Pick a start triplet by priority: explicit **seed** → **contextual start**
   (probability `(bias-1)/bias`, via exact → casefold → prefix matching in
   `ContextStateMatcher`) → weighted global `starts3`/`starts2`.
4. Walk up to `max_steps = 90` tokens: choose next token from order-3
   transitions, backing off to order-2 then order-1 (gated by `enable_backoff`/
   `backoff_min_order`). `weighted_next_choice` applies context bias, repetition
   penalty, and seen-ngram penalties; degraded windows break early.
5. Post-process: trim repetitive tail → trim to sentence boundary → fix bad
   endings → strip leading punctuation → detokenize within `max_chars`.
6. Reject if too short / low diversity / short-context-copy / context-heavy;
   otherwise return text + `GenerationTrace`.

## `/pivo` command lifecycle (`handlers/pivo.py:cmd_pivo`)

1. `GroupOnly` filter; require `from_user`.
2. `parse_pivo_command(message)` → `(planned_time, target, explicit_mentions)`
   (UTF-16-aware entity parsing + plain `@username` regex; mentions HTML-escaped).
3. `pivo_service.build_call_message(...)`:
   - explicit mentions path (capped by `explicit_mentions_limit`), **or**
   - subscriber fan-out: load `chat_members`, decrypt, drop the caller, cap by
     `subscriber_fanout_limit`; fall back to a canned phrase if empty.
   - Body assembled from random template pools (`pivo_message_builder`).
4. **Quota** (skipped for OWNER): determine admin/owner via
   `is_admin_or_owner` (calls `bot.get_chat_administrators`), then
   `consume_daily_call_quota` (atomic DB upsert; limit 3 user / 5 admin). If
   exhausted → inform and return.
5. `message.reply(text, parse_mode=HTML)`. On send failure, **refund** the quota
   (`refund_daily_call_quota`) and re-raise.

`/pivo_on` subscribes (rejects bots), `/pivo_off` unsubscribes, `/pivo_privacy`
prints the privacy notice. Subscription writes Fernet-encrypted user fields keyed
by HMAC hashes (`pivo_service.subscribe`, `chat_members_repo.upsert`).

## Admin/config command flows (`handlers/admin.py`, `common.py`)

- `/config [full]` — render runtime values (no auth; read-only).
- `/set <key> <value>` — `GroupOnly + AdminOrOwner`; `try_apply` parses +
  cross-validates on a copy before mutating `RuntimeState` (effective until
  restart). A second `/set` handler without `AdminOrOwner` is the deny fallback.
- `/setprob <0..1>` — fast path for `reply_probability` (same auth + deny pattern).
- `/clear [confirm]` — `GroupOnly + AdminOrOwner`; without `confirm`, prints
  instructions; with `confirm`, wipes the chat's tables, invalidates generator
  cache, forgets runtime state. Throttled to once/hour (`main.py:24`).
- `/stats`, `/help`, `/ping` — `common` router (stats is group-only).

## Migration lifecycle (`infrastructure/migrator.py`)

At startup, `run(conn)` ensures `schema_migrations`, lists `NNN_*.sql|.py` not
yet applied, and applies each in sorted order inside a transaction (`.sql` wrapped
in `BEGIN..COMMIT` via `executescript`; `.py` modules expose `async apply(conn)`).
Each success is recorded; any failure triggers rollback and re-raise (aborting
startup). Idempotent across restarts.
