# 05 — Module Inventory

> Independent audit, source-only. Per-module purpose, public surface, dependencies, callers. Import edges verified programmatically (see [06_dependency_graph.md](06_dependency_graph.md)). Behavior flows in [04_execution_flow.md](04_execution_flow.md).

Legend: **Pub** = primary public symbols. **Deps** = internal app deps. **Callers** = who uses it.

## Composition root

### `main.py`
- **Purpose:** build all singletons, wire dispatcher, run long-polling, clean shutdown.
- **Pub:** `configure_dispatcher(dp, *, db, generator, pivo_service, learning_service, runtime_state, settings, bot_username, bot_id)`, `run_bot()`, `COMMAND_COOLDOWNS_SECONDS`.
- **Deps:** config.settings/runtime_state, core.markov, domain.pivo, all handlers, infrastructure.database, middlewares, presentation.bot_messages, services, log_masking.
- **Callers:** `__main__`.

## `app/config/` — configuration

### `registry.py` — single source of truth for runtime-mutable fields
- **Pub:** `FieldSpec`, `RUNTIME_FIELDS` (23 specs), `get_spec`, `runtime_field_names`, `validate_cross_fields(obj)`, `try_apply(state, name, value)`; parsers `_parse_bool/_int_*/_float_in_range`.
- **Deps:** none. **Callers:** settings, runtime_state, runtime_config.
- **Note:** the anti-duplication keystone — adding a `/set` field is a one-line edit here.

### `settings.py` — env loading + immutable settings
- **Pub:** `Settings` (frozen-ish dataclass, slots), `load_settings(load_env=True)`, `_load_runtime_fields`.
- **Deps:** registry, core.reply_policy (`DEFAULT_BOT_TEXT_ALIASES`). **Callers:** main, runtime_state, handlers (admin/pivo via injection), filters.
- **Note:** non-mutable secrets (`BOT_TOKEN`, `OWNER_ID`, `DB_PATH`, PIVO secrets, `LOG_LEVEL`) parsed/validated here, not in the registry.

### `runtime_state.py` — mutable per-process state
- **Pub:** `RuntimeState` (registry fields + `last_reply_ts`, `learned_messages`, `recent_short_replies`, activity tracking + pruning), `runtime_state_from_settings(settings)`.
- **Deps:** registry, settings. **Callers:** main, handlers, response_generator.

### `runtime_config.py` — `/set` application + errors
- **Pub:** `ALLOWED_RUNTIME_KEYS`, `UNKNOWN_RUNTIME_KEY_MESSAGE`, `UnknownRuntimeSettingError`, `InvalidRuntimeSettingValueError`, `apply_runtime_setting(state, key, value)`, `parse_bool`.
- **Deps:** registry. **Callers:** handlers.admin.
- **Note:** `parse_bool` here duplicates `registry._parse_bool` (different return contract); flagged for [11_code_quality.md] (M2).

## `app/core/` — generation pipeline (Telegram-agnostic)

### `markov.py` — the n-gram engine (1438 LOC)
- **Pub:** `MarkovGenerator` (`generate_text`, `generate_text_with_trace`, `invalidate_chat_cache`), `tokenize`, `detokenize`, `is_short_generated_reply`, `escalated_randomness_strength`, many pure helpers (`build_windows`, `content_tokens`, `trim_*`, `finalize_reply_ending`, `weighted_*_choice`, `GenerationTrace`).
- **Deps:** context_state_matcher, lexicon, **infrastructure.database** (used as read port). **Callers:** response_generator, learning handler, reply_policy, candidate_scorer, services.learning_service, tools.
- **Note:** largest module; mixes ~30 pure functions with the stateful generator + LRU caches. Refactor candidate ([11_code_quality.md], [15_technical_debt.md]).

### `context_state_matcher.py` — fuzzy context→state matching
- **Pub:** `ContextStateMatcher.match(chat_id, window, order, include_prefix)`, `ContextStateMatch`, `invalidate_chat_cache`.
- **Deps:** infrastructure.database. **Callers:** markov.
- **Note:** exact/casefold/prefix matching with cached per-chat state index; Cyrillic-aware prefix heuristics. Off by default (`fuzzy_context_*`).

### `response_generator.py` — best-of-N orchestration
- **Pub:** `ResponseGenerator` (`generate`, `generate_with_result`), `GenerationRequest`, `ResponseGenerationResult`, `VerbatimCopyChecker` (Protocol), `was_recent_short_reply`.
- **Deps:** config.runtime_state, candidate_scorer, markov, text, log_masking. **Callers:** learning handler (per-message instantiation).

### `candidate_scorer.py` — reply scoring
- **Pub:** `score_candidate(text, tokens, context_tokens)`, `CandidateScore`, component fns (`completion_quality`, `lexical_diversity`, `natural_length`, `context_relevance`, `repetition_penalty`), `meaningful_tokens`.
- **Deps:** lexicon, markov (`content_tokens`). **Callers:** response_generator.

### `reply_policy.py` — when/whether to reply
- **Pub:** `bot_is_mentioned`, `should_reply_to_message`, `cooldown_allows_reply`, `has_enough_model_data`, `text_contains_bot_alias`, `DEFAULT_BOT_TEXT_ALIASES`.
- **Deps:** markov (`tokenize`). **Callers:** learning handler, settings (default aliases).

### `text.py` — normalization + capitalization
- **Pub:** `sanitize_text`, `capitalize_reply_sentences`, `remove_links/mentions`, `normalize_repeats`.
- **Deps:** privacy_filter. **Callers:** database (write), messages_repo, learning_service, response_generator, handlers, migration 002.

### `privacy_filter.py` — PII/secret redaction
- **Pub:** `redact_sensitive_data`, `redact_emails/phones/secrets`.
- **Deps:** none. **Callers:** text.
- **Note:** regex + Shannon-entropy generic-secret detector; runs before persistence/caching. Security-relevant — validated in [08_security.md] (M3).

### `lexicon.py` — stopword/bad-ending word sets
- **Pub:** `STOPWORDS`, `BAD_ENDING_WORDS` (frozensets, EN+RU). **Callers:** candidate_scorer, markov.

## `app/domain/` — `/pivo` domain

### `pivo.py` — crypto + mention assembly
- **Pub:** `PivoSecurity` (`hmac_value`, `encrypt_value`, `decrypt_value`), `PivoMember`, `build_pivo_mention(s)`, `collect_pivo_mentions`, `normalize_username`, `display_name_from_user`, `get_random_pivo_message`, `PIVO_FALLBACK_MENTIONS`, `PIVO_PRIVACY_MESSAGE`.
- **Deps:** services.pivo_message_builder (**lazy**, in `get_random_pivo_message`). **Callers:** main, services.pivo_service, handlers.pivo.
- **Note:** `get_random_pivo_message` has no production caller (PivoService builds messages directly) — possible dead code ([11_code_quality.md] M2).

### `pivo_templates.py` — template string pools (399 LOC, data only)
- **Pub:** `PIVO_DEFAULT_*` / `PIVO_TARGET_*` part tuples, `PIVO_NOTIFICATION_LINES`. **Callers:** pivo_message_builder. ruff `E501` ignored here.

## `app/services/` — business logic

### `learning_service.py`
- **Pub:** `LearningService` (`record_message`, `get_token_volume`, `is_verbatim_copy`).
- **Deps:** markov, text, infrastructure.database. **Callers:** main, learning handler; satisfies `VerbatimCopyChecker`.

### `pivo_service.py`
- **Pub:** `PivoService` (`subscribe`, `unsubscribe`, `build_call_message`, `consume_daily_call_quota`, `refund_daily_call_quota`, `configure_call_limits`), `PivoQuotaResult`, `PivoCallLimitError`, daily-limit constants.
- **Deps:** domain.pivo, infrastructure.database, pivo_message_builder. **Callers:** main, handlers.pivo.

### `pivo_message_builder.py`
- **Pub:** `build_pivo_message(...)`, `build_pivo_message_context(...)`, `PivoMessageGenerator`, `PivoMessageContext`, time/target formatters.
- **Deps:** domain.pivo_templates. **Callers:** pivo_service, domain.pivo (lazy).

### `pivo_parser.py`
- **Pub:** `parse_pivo_command(message) -> PivoCommandArgs`.
- **Deps:** aiogram types only. **Callers:** handlers.pivo.
- **Note:** UTF-16 offset→index conversion for entity spans; HTML-escapes mentions.

## `app/repositories/` — SQL by domain (each holds `conn_provider` + shared `Lock`)

| Module | Pub | Tables |
|---|---|---|
| `markov_repo.py` | `MarkovRepo` (get_starts/3, get_start*_if_exists, get_transitions/1/3, get_states, get_chat_token_volume) | starts, starts3, transitions, transitions3, transitions1 |
| `messages_repo.py` | `MessagesRepo` (exists, get_recent_normalized) — depends on core.text | messages |
| `chat_members_repo.py` | `ChatMembersRepo` (upsert, remove, list_members) | chat_members |
| `pivo_usage_repo.py` | `PivoUsageRepo` (consume_daily_call, refund_daily_call, delete_usage_before) | pivo_daily_usage |

All read-only repos (markov/messages) and write repos commit within the lock. Callers: `Database` facade.

## `app/infrastructure/`

### `database.py` — SQLite facade (385 LOC)
- **Pub:** `Database` (`init`, `close`, `save_message_and_update_model`, many repo delegates, `get_stats`, `clear_chat`, pivo-usage methods, `cleanup_pivo_daily_usage`), `PIVO_DAILY_USAGE_RETENTION_DAYS`.
- **Deps:** core.text, infrastructure.migrator, repositories. **Callers:** main, core (markov/context_matcher), services, handlers.
- **Note:** owns the single connection + `asyncio.Lock`; mixes facade + delegation + cross-domain queries + raw n-gram write logic. God-object tendency ([11_code_quality.md] M2). Repeated `if self.x is None: raise` guards (~10×).

### `migrator.py` — migration runner
- **Pub:** `run(conn)`. **Deps:** none (dynamic import of `.py` migrations). **Callers:** database.init.

## `app/handlers/` — aiogram routers

| Module | Router / commands | Deps |
|---|---|---|
| `learning.py` | `F.text` catch-all (learn + maybe reply) | runtime_state, markov, reply_policy, response_generator, text, _helpers, log_masking, services |
| `admin.py` | `/config`, `/set`(+deny), `/setprob`(+deny), `/clear`(+deny) | runtime_config, runtime_state, settings, markov, filters, _helpers, database, presentation |
| `pivo.py` | `/pivo`, `/pivo_on`, `/pivo_off`, `/pivo_privacy` | runtime_state, settings, domain.pivo, filters, _helpers, services, pivo_parser, pivo_service |
| `common.py` | `/ping`, `/help`, `/stats` | runtime_state, filters, _helpers, database, presentation |
| `errors.py` | `@router.error()` global handler | — |
| `_helpers.py` | `is_group_message`, `reply_humanized` | aiogram only |

## `app/filters/`, `app/middlewares/`, `app/presentation/`, `app/log_masking.py`

| Module | Pub | Notes |
|---|---|---|
| `filters/admin_or_owner.py` | `is_admin_or_owner(message, bot, settings)`, `AdminOrOwner` | owner short-circuit, else `get_chat_administrators`; swallows API errors → deny |
| `filters/group_only.py` | `GroupOnly` | group/supergroup only |
| `middlewares/throttling.py` | `ThrottlingMiddleware` | per (chat,user,command) cooldown, TTL/max-key pruning, notify subset |
| `presentation/bot_messages.py` | `TELEGRAM_COMMANDS`, `format_*` | pure string builders; `format_config_message(state: object)` is untyped |
| `log_masking.py` | `init_masking`, `mask_chat_id`, `reset_masking`, `LogMaskingNotInitialized` | process-global HKDF key; fail-fast if uninitialized |

## `tools/` (not shipped at runtime)
- `seed_db.py`, `seed_diverse.py` — synthetic corpus seeders.
- `eval_generation.py` — synthetic selection-path eval.
- `eval_prod.py` (+ `eval_prod_baseline.json`) — offline best-of-N eval vs prod-copy DB (copies DB first). New/untracked.

## Empty / trivial
`app/__init__.py`, `app/handlers/__init__.py`, `app/infrastructure/__init__.py`, `app/migrations/__init__.py` are empty (0 LOC). `__init__.py` re-export hubs: `config`, `core`(empty), `domain`(empty), `filters`, `middlewares`, `presentation`(empty), `repositories`, `services`.
