# Технический аудит проекта PepeEdtaBot

**Дата актуализации:** 2026-06-20, пятнадцатая редакция (STRUCT-001: перенос корневых модулей в пакет `app/`).
**Текущая ветка:** `refactor/struct-001` (не слита). Phase 0, Phase 1 и оба `chore(deps)` — уже в `main`.
**Тесты / проверки:** `unittest discover tests` — 278 OK; `ruff check app/ tests/ tools/ main.py` — clean; `mypy app/` — clean (45 files); harness воспроизводит baseline.
**Backlog:** P0/P1/P2/P3 — пусто. `STRUCT-001` — **выполнен** (flat-раскладка: `app/config`,`app/core`,`app/infrastructure`,`app/domain`,`app/presentation`; seed → `tools/`). Security-backlog (LOW): AUD-010, CA-F11. Отложено по внешним решениям: структурированные JSON-логи, Prometheus-метрики.

Полная история редакций — в конце файла (после session updates). Подробные log-style записи по каждой сессии — в секциях `## 16` — `## 24` (новые сверху-вниз по порядковому номеру).

---

## Session update - 2026-06-20 (generation Phase 1)

Реализована Фаза 1 дорожной карты на ветке `feat/generation-phase1`. Phase 0 и
обновление зависимостей (`idna` 3.18, общий `chore(deps)`) уже слиты в `main`
(PR #30, #31).

### Completed
- **1.1 Injected RNG + детерминизм:** через весь генерационный контур проброшен
  `random.Random` (хелперы `weighted_*_choice`, contextual starts); публичный
  `generate_text(..., rng=None)` создаёт свежий RNG по умолчанию, контракт `str`
  сохранён. Недетерминированный `set.pop()`-eviction заменён на FIFO через
  `OrderedDict` (`remember_bounded`). SQL-пулы переходов и weighted-populations
  получили стабильную сортировку (`app/repositories/markov_repo.py`).
- **1.2 Trace:** добавлен frozen `GenerationTrace` (attempts, фактический order,
  jump_count, rejection reason, token count) + `generate_text_with_trace()`;
  DEBUG-лог trace без текста/идентификаторов.
- **1.3 Offline-харнесс:** `tools/eval_generation.py` на синтетическом корпусе
  (`tools/fixtures/synthetic_generation_corpus.txt`) с фикс-seed; зафиксирован
  post-Phase-0 baseline (`tools/generation_baseline.json`).

### Changed files
- `markov.py`, `app/repositories/markov_repo.py`
- `tools/eval_generation.py`, `tools/fixtures/synthetic_generation_corpus.txt`,
  `tools/generation_baseline.json`
- `tests/test_markov_and_text.py`, `tests/test_db_logic.py`,
  `tests/test_eval_generation.py`
- `docs/RESPONSE_GENERATION_ROADMAP.md` (запись STRUCT-001 между Фазой 1 и 2)

### Tests/checks run
- `unittest discover tests` — **278 OK**.
- `ruff check app/ tests/ tools/` — clean; `mypy app/` — clean.
- Независимая проверка детерминизма харнесса: один seed → идентичные контентные
  метрики (различается лишь wall-clock latency); разный seed → различие.

### Remaining work
- **STRUCT-001** (техдолг структуры) — следующий шаг, ДО Фазы 2.
- Затем Фазы 2–4 дорожной карты.

---

## Session update - 2026-06-20 (generation Phase 0)

Работа ведётся по согласованной дорожной карте `docs/RESPONSE_GENERATION_ROADMAP.md`
(совместный анализ Claude + Codex). Реализована Фаза 0 на ветке
`feat/generation-phase0`.

### Completed
- **0.1+0.2 (атомарно):** в `markov.py` отключена ветка «прыжков»
  (`jump_probability = 0.0`) и удалена forced-jump-отбраковка длинных no-context
  ответов (бывшая проверка `... and jump_count == 0: return ""`). Устранён разрыв
  связности (jump-splice не дописывал стартовые токены в вывод).
- **0.3 generate-before-learn:** в `app/handlers/learning.py` ответ генерируется
  по состоянию модели **до** обучения на входящем сообщении; гейт «достаточно
  данных» использует pre-message объём; `record_message` вынесен в `try/finally`
  → каждое learnable-сообщение персистится ровно один раз на всех путях (мало
  данных / cooldown / `should_reply=false` / провал генерации / успех /
  исключение); добавлен guard, отклоняющий кандидат, равный нормализованному
  текущему сообщению.
- **0.4:** post-generation-валидация в `markov.py` выполняется по фактически
  усечённому тексту (`tokenize(result)`), а не по полному буферу `generated`.

### Changed files
- `app/handlers/learning.py`, `markov.py`
- `tests/test_handlers.py`, `tests/test_markov_and_text.py`
- `docs/RESPONSE_GENERATION_ROADMAP.md` (+ входные доки
  `GENERATION_IMPROVEMENT_PROPOSAL.md`, `RESPONSE_GENERATION_IMPROVEMENT_PLAN.md`)

### Tests/checks run
- `unittest discover tests` — **271 OK** (вкл. новые регресс-тесты:
  threshold-crossing без self-reply, отклонение копии текущего сообщения,
  persist при cooldown/провале, отключённые jumps, отсутствие D1-отбраковки,
  валидация усечённого текста).
- `ruff check app/ tests/` — clean.

### Not run / limitations
- `mypy` не запускался (в окружении не сконфигурирован под текущий `.venv`).
- `markov.py` всё ещё вне strict `ruff`/`mypy` (15 пред­существующих `UP045`);
  включение запланировано как отдельный пункт DoD дорожной карты.

### Remaining work
- Фаза 1 (P0): injected RNG, инструментация (trace), offline-харнесс + baseline.
- Далее по дорожной карте: Фазы 2–4.

---

## 0. Статус проекта

### 0.1. Ветка `refactor/structure` — слита в `main` (2026-05-08)

Все шесть фаз рефакторинга, запланированные после первоначального аудита, выполнены. Ветка была признана готовой к merge в третьей редакции аудита.

| Фаза | Содержание | Статус |
|---|---|---|
| 1 | Разбить `main.py` на routers + сервисы + репозитории | ✅ |
| 2 | Версионирование миграций (`schema_migrations` + `app/migrations/`) | ✅ |
| 3 | `chat_member_profiles` + `MembersService` + HKDF derivation | ✅ (затем удалено session 21 при унификации в `chat_members`) |
| 4 | Privacy: удаление `messages.text`, дедупликация генерации | ✅ |
| 5 | Фильтры (`GroupOnly`, `AdminOrOwner`) + throttling middleware | ✅ |
| 6 | `ruff`/`mypy strict` для `app/`, `requirements.lock`, CI с линтерами, тесты хендлеров | ✅ |

Все 200 тестов зелёные локально (3.12/3.13/3.14). `ruff check` и `mypy app/` проходят без замечаний.

### 0.2. Ветка `codex/pivo-daily-quota` — слита в `main` (2026-05-09)

После merge `refactor/structure` добавлена фича суточной квоты `/pivo` и закрыт ряд runtime-багов. Полный трекинг — в `PROJECT_AUDIT_CODEX.md`.

| Коммит | Содержание |
|---|---|
| `e7ec2fb` | feat(pivo): add daily call quota |
| `73aa324` | ci: run checks on codex branches |
| `6edb0ca` | Revert «ci: run checks on codex branches» |
| `5ee676c` | fix(pivo): harden daily quota flow |
| `84693d9` | fix(clear): allow confirm under cooldown |
| `dd43b13` | fix(runtime): avoid fsm state injection conflicts |

### 0.3. Ветка `refactor/audit-p2-batch` — слита в `main` (2026-05-10)

P2-batch: единый реестр настроек (`config_registry.py`), migrator → `executescript` + rollback, Dockerfile hardening (pin minor, non-root, HEALTHCHECK), `LOG_LEVEL` из `.env`, README/ARCHITECTURE обновлены, debug-лог успешной генерации. Подробности — section 18.

### 0.4. Ветка `fix/audit-followups` — слита в `main` (2026-05-10)

Audit follow-ups: атомарность .sql-миграций через `BEGIN; ... COMMIT;`-обёртку (P1 от ревью), Docker bind-mount writability через root-entrypoint с `chown` + `runuser` (P1 от ревью), `BOT_TEXT_ALIASES` в `.env` с safe fallback (P2-5). Подробности — section 19.

### 0.5. PR #5 `polish/audit-session-20` — слит в `main` (2026-05-10)

P3-полировка: `matches_training_prefix` (rename + docstring + drop dead branch), магические числа → константы, `MENTION_RE` для email, `notify_on_throttle` UX для `/clear`. Подробности — section 20.

### 0.6. PR #6 `docs/audit-sync` — слит в `main` (2026-05-10)

Только документ — синхронизация устаревших ссылок в самом `PROJECT_AUDIT.md` после session 20.

### 0.7. PR #7 `feat/unify-chat-members` — слит в `main` (2026-05-10)

Унификация участников: миграция 007 переносит `pivo_chat_members` → `chat_members` (без `is_bot`); `chat_member_profiles` / `MembersService` / `MembersRepo` / `app/security/key_derivation.py` удалены; `PivoRepo` → `ChatMembersRepo`. Подробности — section 21.

### 0.8. PR #8 `feat/log-masking-and-p3-tests` — слит в `main` (2026-05-10)

Маскирование `chat_id` в логах (`app/log_masking.py`), тесты на ошибки Telegram API, унификация стиля `cursor.fetchone()`. Подробности — section 22.

### 0.9. Live smoke 2026-05-10

Запуск на тестовом чате: миграция 007 применилась чисто, `/pivo_on` / `/pivo` / `/pivo_off` работают, в логах `chat=1f6bfe93` (8-hex маска, не сырой `chat_id`). Бэкап `data/markov.db.backup-before-migration-007` сохранён.

### 0.10. PR #24 `audit-security-debt-cleanup` — слит в `main` (2026-05-15)

Security debt cleanup (commit `47bd749`): закрыты AUD-007, AUD-011, AUD-004 doc, AUD-006.

| Закрытый пункт | Что сделано |
|---|---|
| AUD-007 | Добавлен `app/handlers/errors.py` — централизованный error router, логирует `TelegramAPIError` и неожиданные исключения; зарегистрирован в `configure_dispatcher()` |
| AUD-011 | `assert` в prod-коде заменены на явные `RuntimeError`/`ValueError`; в `db.py` и сервисах — только runtime-исключения |
| AUD-004 | `docs/OPERATIONS.md` расширен секцией «Database Retention»: ручная очистка `messages` per-chat, проверка размера через `dbstat`, рекомендуемый schedule |
| AUD-006 | HEALTHCHECK в `Dockerfile` улучшен: теперь проверяет открытие SQLite-файла, а не только интерпретатор; ограничения probe задокументированы в `docs/OPERATIONS.md` |

### 0.11. PR #25 и #26 `improve-generation-quality` — слиты в `main` (2026-05-15)

Улучшение качества генерации и рефакторинг prefix-cache (commits `182e157`, `bc3a182`, `1ea631d`, `64d15f6`):

| Изменение | Файл | Суть |
|---|---|---|
| Переработан prefix-cache | `app/services/learning_service.py` | `matches_training_prefix` → `is_verbatim_copy`: теперь проверяет дословное совпадение нормализованного текста с последними N сообщениями (bounded, `text_cache_max_messages`), а не prefix-эвристику. Закрывает AUD-003. |
| Bounded text cache | `learning_service.py` | Кэш per-chat загружает только `LIMIT N` строк из БД вместо `fetchall()`; инвалидируется при записи сообщения |
| Новый метод в MessagesRepo | `app/repositories/messages_repo.py` | `get_recent_normalized(chat_id, limit)` — возвращает последние N нормализованных строк |
| Удалён старый метод | `messages_repo.py` | `get_all_normalized` удалён (использовал `fetchall()` без лимита) |
| Исправлен E741 | `markov.py` | Переменная `l` → `length` |
| Обновлены тесты | `tests/test_learning_service.py` | Тесты переписаны под `is_verbatim_copy` |

### 0.12. CA-F13 closed — db.py get_stats refactor (2026-05-15)

Рефакторинг `get_stats` и `clear_chat` helpers в `db.py` (CA-F13 / AUD-012):

- Добавлен приватный `_fetch_int(db, sql, params) -> int` — устраняет повторение паттерна `(await (await db.execute(...)).fetchone())[0]`
- `get_stats` упрощён до серии `await f(db, sql, p)` вызовов — читаемость резко улучшилась, логика не изменилась
- `ruff check app/ tests/` и `mypy app/` — чисто до и после изменения

---

## 1. Краткое резюме проекта

PepeEdtaBot — Telegram-бот для группового чата на `aiogram v3`, который обучается на сообщениях участников и генерирует ответы по цепям Маркова variable-order (n=3 → n=2 → n=1, с backoff). Внешние LLM не используются. Дополнительно есть opt-in команда `/pivo` для шуточного созыва участников в Discord; данные подписок хранятся в таблице `chat_members` с HMAC-индексированием и Fernet-шифрованием (`PIVO_HMAC_SECRET` / `PIVO_ENCRYPTION_SECRET`).

После всех итераций проект перешёл из «всё в `main.py`» к слоистой архитектуре `handlers / services / repositories / filters / middlewares / migrations / infrastructure` (плюс корневой `app/log_masking.py` для HKDF-маскирования `chat_id` в логах). Версионированные миграции запускаются атомарно через `BEGIN; ...; COMMIT;` обёртку поверх `sqlite3.executescript`. Throttling-middleware с `notify_on_throttle`-фолбэком для админ-команд. Единый реестр runtime-настроек (`config_registry.py`). Hardened Dockerfile (pin `python:3.14.0-slim`, non-root через root-entrypoint + `runuser`, HEALTHCHECK). CI на матрице Python 3.12/3.13/3.14.

**Серьёзных уязвимостей нет. Открытого техдолга P0/P1/P2/P3 нет.** Prefix-cache переработан на `is_verbatim_copy` с bounded загрузкой (AUD-003 закрыт). `db.py get_stats` рефакторирован (CA-F13 закрыт). Error middleware добавлен (AUD-007 закрыт). Отложены два пункта, заблокированные внешними решениями: структурированные JSON-логи (ждём выбор системы агрегации), Prometheus-метрики (ждём endpoint).

---

## 2. Используемые технологии

- **Python 3.14** (`Dockerfile`).
- **aiogram >=3.7,<4.0** — Telegram Bot API.
- **aiosqlite >=0.20** — асинхронный SQLite.
- **python-dotenv** — загрузка `.env`.
- **cryptography (Fernet, HKDF)** — шифрование данных и derivation доменных ключей.
- Хранилище — **SQLite** (`data/markov.db`), WAL-режим.
- Тесты — `unittest` (`IsolatedAsyncioTestCase` для асинхронных).
- Линтер — **ruff** (E/F/I/UP, line-length=100).
- Тайпчекер — **mypy strict для `app/`**, `ignore_errors` для legacy-модулей.
- CI — **GitHub Actions** на 3.12/3.13/3.14 с шагами `ruff check` → `mypy app/` → `unittest discover`.

---

## 3. Структура проекта

```
PepeEdtaBot/
├── main.py                              # 121 строка — compose root + configure_dispatcher()
├── app/                                 # 27 .py-модулей
│   ├── log_masking.py                   # HKDF-helper для маскирования chat_id в логах
│   ├── handlers/
│   │   ├── _helpers.py                  # reply_humanized (toleratesend_chat_action errors)
│   │   ├── common.py                    # /ping, /help, /stats
│   │   ├── admin.py                     # /config, /set, /setprob, /clear + fallback-handlers (denied)
│   │   ├── pivo.py                      # /pivo (quota check), /pivo_on, /pivo_off, /pivo_privacy
│   │   ├── learning.py                  # F.text + extract_context_tokens, маскированные chat-ids в логах
│   │   └── errors.py                    # централизованный error router — TelegramAPIError + generic
│   ├── services/
│   │   ├── learning_service.py          # record_message, is_verbatim_copy (bounded text-cache)
│   │   └── pivo_service.py              # subscribe / unsubscribe / build_call_message / consume_daily_call_quota / refund_daily_call_quota
│   ├── repositories/
│   │   ├── markov_repo.py               # starts/transitions/transitions3/transitions1
│   │   ├── messages_repo.py             # exists / get_recent_normalized
│   │   ├── chat_members_repo.py         # upsert / list_members / remove (таблица chat_members)
│   │   └── pivo_usage_repo.py           # consume_daily_call / refund_daily_call / delete_usage_before
│   ├── filters/
│   │   ├── group_only.py                # ChatType.GROUP/SUPERGROUP
│   │   └── admin_or_owner.py            # OWNER_ID или администратор чата, fail-closed
│   ├── middlewares/
│   │   └── throttling.py                # per-user per-command cooldown + notify_on_throttle
│   ├── infrastructure/
│   │   └── migrator.py                  # NNN_*.sql/.py: атомарный BEGIN/COMMIT + executescript
│   └── migrations/
│       ├── 001_initial.sql              # полная схема для пустых БД
│       ├── 002_normalize_messages_text_column.py
│       ├── 003_anonymize_authors.py
│       ├── 004_chat_member_profiles.sql # legacy — таблица будет дропнута 007
│       ├── 005_drop_messages_text.py
│       ├── 006_pivo_daily_usage.sql     # таблица pivo_daily_usage
│       └── 007_unify_chat_members.sql   # pivo_chat_members + chat_member_profiles → chat_members (без is_bot)
├── db.py                                # фасад: соединение, save_message_and_update_model, clear_chat, get_stats (_fetch_int helper), делегаты; cleanup_pivo_daily_usage при init()
├── markov.py                            # 674 строки — НЕ ТРОГАТЬ
├── pivo.py                              # 124 строки — PivoSecurity, PivoMember (без is_bot)
├── pivo_templates.py                    # 487 строк (контент)
├── bot_messages.py                      # 116 строк — форматирование
├── bot_policy.py                        # 82 строки — bot_is_mentioned, cooldown, should_reply
├── settings.py                          # 120 строк — Settings + load_settings
├── runtime_state.py                     # 43 строки — RuntimeState dataclass
├── runtime_config.py                    # 36 строк — apply_runtime_setting (итерируется по config_registry)
├── config_registry.py                   # 152 строки — FieldSpec × 20, validate_cross_fields
├── text_utils.py                        # 31 строка — sanitize_text (mention RE с lookbehind)
├── seed_db.py                           # one-off seeder для smoke
├── seed_diverse.py                      # альтернативный seeder с разнообразной лексикой
├── tests/                               # 13 файлов
│   ├── test_bot_messages.py             # форматирование
│   ├── test_bot_policy.py               # политика ответа
│   ├── test_db_logic.py                 # save_message_and_update_model, get_stats, daily_usage retention
│   ├── test_filters.py                  # GroupOnly, AdminOrOwner, ThrottlingMiddleware + notify_on_throttle
│   ├── test_handlers.py                 # happy-path по всем 4 роутерам + denied-fallback + reply_humanized resilience
│   ├── test_learning_service.py         # prefix-cache, matches_training_prefix
│   ├── test_log_masking.py              # init_masking, mask_chat_id, secret rotation
│   ├── test_main.py                     # smoke-тест wiring configure_dispatcher()
│   ├── test_markov_and_text.py          # генерация, токенизация, email-safe MENTION_RE
│   ├── test_migrator.py                 # идемпотентность, resume, legacy fixture, атомарность .sql
│   ├── test_pivo.py                     # HMAC, Fernet, mentions, E2E subscribe→quota→unsubscribe
│   ├── test_runtime_config.py           # /set ключи
│   └── test_settings.py                 # load_settings (LOG_LEVEL, BOT_TEXT_ALIASES)
├── Dockerfile                           # python:3.14.0-slim, root-entrypoint + runuser, HEALTHCHECK
├── docker-entrypoint.sh                 # chown bind-mount → runuser -u bot
├── .gitattributes                       # `*.sh text eol=lf`
├── pyproject.toml                       # ruff + mypy конфигурация
├── requirements.txt                     # верхнеуровневые диапазоны (источник для lock)
├── requirements.lock                    # фиксированные версии, используется Dockerfile + CI
├── requirements-dev.txt                 # `-r requirements.lock` + ruff/mypy для CI
├── .github/workflows/ci.yml             # ruff + mypy + tests на 3.12/3.13/3.14
├── docs/ARCHITECTURE.md
├── README.md
└── PROJECT_AUDIT.md (этот файл)
```

**Метрики (актуальное состояние):**

| Метрика | До рефакторинга | Сейчас |
|---|---|---|
| `main.py` | 588 строк | **121 строка** (compose root + `configure_dispatcher`, `init_masking`) |
| `db.py` | ~620 строк (миграции + бизнес) | **423 строки** (фасад) |
| Файлов `.py` в `app/` | 0 | **27** |
| Test-файлов | — | **13** |
| Тестов | 83 | **199** |
| Миграций | 0 (inline `CREATE TABLE`) | **7** (001…007) |
| Слоёв архитектуры | 1 (всё в `main.py`) | 6 |
| Линтер/тайпчекер | нет | ruff + mypy strict для `app/` |
| CI | нет | GitHub Actions × 3 версии Python |
| Lock-файл | нет | `requirements.lock` (используется Docker + CI) |

---

## 4. Что выполнено из старого аудита

### P0 — все три пункта закрыты

| ID | Старая формулировка | Сейчас |
|---|---|---|
| **P0-1** | `messages.text` хранится без opt-in, это PII в открытом виде | ✅ Колонка `text` удалена миграцией [`005_drop_messages_text.py`](app/migrations/005_drop_messages_text.py); хранится только `normalized_text` (используется генератором для exact-match exclude и `LearningService.matches_training_prefix` для построения set'а префиксов обучающих сообщений). |
| **P0-2** | Нет версионирования миграций | ✅ Реализован [`migrator.py`](app/infrastructure/migrator.py) с таблицей `schema_migrations`, 7 версионированных миграций (`.sql`/`.py`). `.sql` выполняются через `executescript` внутри явного `BEGIN; ...; COMMIT;` — атомарны при ошибке. Идемпотентность и атомарность покрыты тестами. |
| **P0-3** | Не спроектирована таблица чувствительных данных участников | ✅ Канонической таблицей участников чата сейчас является [`chat_members`](app/migrations/007_unify_chat_members.sql) (PRIMARY KEY `(chat_hash, user_hash)`, `encrypted_user_id`, `encrypted_username`, `encrypted_display_name`; HMAC-ключи под `PIVO_HMAC_SECRET`, Fernet под `sha256(PIVO_ENCRYPTION_SECRET)`). Изначально (P0-3) была спроектирована параллельная `chat_member_profiles` + `MembersService` с HKDF-доменизацией под будущие домены, но они не материализовались, и в session 21 эта инфраструктура была удалена. |

### P1 — все восемь пунктов закрыты

| ID | Старая формулировка | Сейчас |
|---|---|---|
| **P1-1** | `main.py` — бог-файл с inner-функциями-хендлерами | ✅ `main.py` = 84 строки, чистый compose root. Все хендлеры в `app/handlers/{common,admin,pivo,learning}.py`, каждый — `aiogram.Router`. |
| **P1-2** | `db.py` смешивает миграции, бизнес-запросы, статистику | ✅ Введены репозитории `MarkovRepo`, `MessagesRepo`, `ChatMembersRepo`, `PivoUsageRepo`. `db.py` оставлен как фасад с делегатами + кросс-доменными транзакциями (`save_message_and_update_model`, `clear_chat`, `get_stats`). Объём `db.py` — 423 строки (с ~620). |
| **P1-3** | Нет фильтров авторизации | ✅ [`GroupOnly`](app/filters/group_only.py) и [`AdminOrOwner`](app/filters/admin_or_owner.py) применяются декларативно в декораторе `@router.message(...)`. Старые хелперы `_is_owner` / `_is_chat_admin` / `_can_manage_settings` удалены. |
| **P1-4** | Throttling/rate-limit отсутствует | ✅ [`ThrottlingMiddleware`](app/middlewares/throttling.py) с per-user-per-command cooldown; в production включён `clear=3600` сек (см. `COMMAND_COOLDOWNS_SECONDS` в `main.py`). Quota для `/pivo` реализована отдельно в `PivoService` (daily quota). Поддерживает `notify_on_throttle` — список команд, для которых при throttle возвращается явный ответ вместо silent drop (включает `clear`). |
| **P1-5** | Нет lock-файла | ✅ Закрыто (B2, `db1de05`): `Dockerfile` устанавливает `requirements.lock`, `requirements-dev.txt` — это `-r requirements.lock` + dev-зависимости. CI и Docker реально используют lock. |
| **P1-6** | Нет CI | ✅ [`.github/workflows/ci.yml`](.github/workflows/ci.yml) на матрице `python-version: [3.12, 3.13, 3.14]`, шаги `ruff check` → `mypy app/` → `unittest discover`. |
| **P1-7** | Нет линтеров и mypy | ✅ [`pyproject.toml`](pyproject.toml) с `ruff` (E/F/I/UP, line-length=100) и `mypy strict для app.*`; legacy-модули вынесены в `[[tool.mypy.overrides]] ignore_errors = true`. |
| **P1-8** | Нет тестов хендлеров | ✅ [`tests/test_handlers.py`](tests/test_handlers.py) — happy-path по всем 4 роутерам + denied-fallback для unauthorized admin-команд + `TestReplyHumanizedResilience` для Telegram API errors. Прямые вызовы с `MagicMock`/`AsyncMock`. |

### P2 — все закрыты

| ID | Описание | Статус | Комментарий |
|---|---|---|---|
Все P2-задачи закрыты в течение сессий 17–22. Полный статус — в section 14. Открытые пункты — только заблокированные внешними решениями (JSON-логи, метрики).

### P3 — закрыто

Полный список закрытий — в section 14. Открытого P3-долга нет.

### Известные регрессии относительно старого `main` — **закрыты**

Эти изменения изначально не были отмечены и нашлись при независимой ревизии. Оба к настоящему моменту закрыты.

#### ~~R1. Админ-команды без прав молча игнорировались~~ — закрыто `a0fae76` (B1)

`app/handlers/admin.py` теперь регистрирует fallback-handler с тем же `Command(...)` без фильтра `AdminOrOwner`: при отказе пользователь получает явный «нет прав».

#### ~~R2. `/clear` молча сбрасывался throttling-middleware~~ — закрыто в session 20

`ThrottlingMiddleware` получил параметр `notify_on_throttle`. В production `/clear` входит в этот набор и при попытке повторного вызова под cooldown возвращает «Слишком часто. Подождите ~N сек.». `/pivo` остаётся silent drop по дизайну.

---

## 5. Архитектура — текущая оценка

### 5.1. Слоистость

Три явных слоя в `app/`:

```
handlers (aiogram Router'ы) → services (бизнес-логика) → repositories (SQL)
              ↑                       ↑
           filters                 infrastructure (migrator)
           middlewares             log_masking (HKDF helper)
```

Хендлеры **не знают** про SQL: они работают только с сервисами и моделями `aiogram`. Сервисы **не знают** про aiogram (исключение — `PivoService.subscribe(chat_id, user)`, осознанное компромиссное решение: при подписке нам нужен `aiogram.types.User` целиком для извлечения username/display_name). Это правильное направление и сохранено во всех новых модулях.

### 5.2. Кросс-доменные транзакции

`save_message_and_update_model` ([db.py:64](db.py:64)) трогает 5 таблиц в одной транзакции (`messages`, `starts`, `starts3`, `transitions`, `transitions3`, `transitions1`). Эта операция оставлена в `Database`-фасаде, не размазана по репозиториям. Это разумно: вводить «UnitOfWork» для одной такой операции — overengineering. При появлении второй такой операции стоит рассмотреть выделение отдельного слоя, но сейчас — нет.

### 5.3. DI

Зависимости (`db`, `generator`, `pivo_service`, `learning_service`, `runtime_state`, `settings`, `bot_username`, `bot_id`, `bot_text_aliases`) кладутся в `dp[...]` и автоматически передаются в хендлеры через `workflow_data` aiogram v3. Никаких самодельных DI-контейнеров. Ровно то, что нужно проекту такого размера.

### 5.4. `main.py` как compose root

121 строка, две функции — `configure_dispatcher` (wiring, отдельно для smoke-теста) и `run_bot` (точка входа). Никакой бизнес-логики, никаких хендлеров. Структура `run_bot`:
1. `load_settings()` → `log_masking.init_masking(...)` → `logging.basicConfig` → `Database.init()` → `MarkovGenerator` → сервисы;
2. `Bot` → `me = await bot.get_me()` → `bot.set_my_commands(...)`;
3. `configure_dispatcher(Dispatcher(), ...)` (middleware + `dp[...]` + routers);
4. `dp.start_polling(bot)` в `try/finally`.

Образцовый compose root. Покрыт smoke-тестом `test_main.py`.

### 5.5. Оставшиеся архитектурные мелочи

- ~~**Дублирование настроек** (P2-1)~~ — закрыто в `refactor/audit-p2-batch`: `config_registry.py` стал единственным источником истины, `Settings`/`RuntimeState`/`runtime_config` теперь итерируются по `FieldSpec`.
- **`runtime_config.py`** обращается к `RuntimeState` через `setattr(state, key, value)` без compile-time-гарантий, что `key` действительно типизирован — это работает (validate_cross_fields + типы `FieldSpec` проверяют значения в рантайме), но `mypy strict` модуль не принимает; поэтому он в `[[tool.mypy.overrides]] ignore_errors = true`. Жить с этим можно.

---

## 6. Безопасность — текущее состояние

### 6.1. Хранение сообщений (бывшая P0-1)

Колонка `messages.text` удалена. Хранится только `normalized_text` — `sanitize_text` убирает `@mentions` и URL, нормализует пробелы. Это уже **не сырой текст пользователя**: tokenizable, но не идентичен исходному. Используется только генератором (для exact-match exclude через `db.message_exists`) и `LearningService.matches_training_prefix` (строит set префиксов 3..5 токенов из всех сообщений чата). Это резкое улучшение privacy-постуры.

**Что осознанно не делаем:** команды `/learn_off` / `/privacy` для opt-out на уровне чата нет. Решено снять с backlog'а (session 21): после удаления `messages.text`, анонимизации `author_id` и маскирования `chat_id` в логах чувствительной информации на уровне чата не остаётся; явный opt-out закрывал бы теоретический риск, не отвечающий реальной угрозе.

#### `matches_training_prefix` — вспомогательный novelty-фильтр

Закрыто в session 20: метод раньше назывался `looks_too_close_to_training_sample` и читался как «privacy-guard от копирования обучающих фраз», по реализации же был novelty-эвристикой со случайным префиксом длиной 3..5. Метод переименован в `matches_training_prefix`, docstring явно описывает его как **второстепенный** фильтр поверх основных защит в `MarkovGenerator.generate_text` (`is_low_diversity_reply`, `is_context_heavy_reply`, `trim_repetitive_tail`, exact-match через `db.message_exists`). Случайность префикса оставлена как сознательный novelty-nudge. Мёртвая ветка SQL-fallback для текстов <3 токенов удалена.

### 6.2. Авторизация и throttling

- `OWNER_ID` или admin чата → проверяет [`AdminOrOwner`](app/filters/admin_or_owner.py); при ошибке `bot.get_chat_administrators` (например, бот не в чате) → возвращает `False` (deny by default). Тесты покрывают этот случай.
- `/clear` → ограничен [`ThrottlingMiddleware`](app/middlewares/throttling.py): `clear=3600` сек **на пользователя в данном чате**. `/clear` добавлен в `notify_on_throttle`-набор, поэтому повторный вызов под cooldown получает явный ответ «Слишком часто. Подождите ~N сек.» вместо silent drop (session 20). Quota для `/pivo` реализована отдельно в `PivoService` через дневной счётчик (см. P2-7). Проброшенные сообщения (без `from_user`, не-команды, команды без записи в `limits`) пропускаются без задержек.

### 6.3. HKDF derivation — текущее применение

HKDF используется в одном месте — `app/log_masking.py` для вывода ключа маскирования `chat_id` в логах:

```python
# app/log_masking.py
_key = HKDF(
    algorithm=hashes.SHA256(), length=32, salt=None, info=b"logging:chat_id",
).derive(PIVO_HMAC_SECRET.encode("utf-8"))
mask = hmac.new(_key, str(chat_id).encode(), hashlib.sha256).hexdigest()[:8]
```

Это означает:
- ключ для масок **не равен** raw-секрету `PIVO_HMAC_SECRET` — domain-метка `logging:chat_id` изолирует домен (тест в `tests/test_log_masking.py` фиксирует, что маска при ротации `PIVO_HMAC_SECRET` меняется).
- Маска стабильна между рестартами при том же секрете → можно коррелировать строки логов одного чата.
- 8 hex символов → 32 бита энтропии, достаточно для одного бота с десятками чатов.

Соль (`salt`) у HKDF — `None`. Это допустимо, потому что master-секрет уже высокоэнтропийный (32 байта `secrets.token_urlsafe(32)` из `.env`). Если бы master-ключ выводился из пользовательского пароля — соль была бы обязательна.

**Историческое замечание:** изначально HKDF был введён под `MembersService` с доменами `members:hmac` / `members:encryption`. Эта инфраструктура удалена в session 21 как «заготовка под несуществующие домены». Текущая `chat_members` таблица использует не HKDF, а раздельные секреты `PIVO_HMAC_SECRET` (HMAC напрямую) и `sha256(PIVO_ENCRYPTION_SECRET)` (Fernet) — те же ключи, что и в исходном `/pivo`-flow.

**Замечание про PivoSecurity:** `PivoSecurity` в [pivo.py](pivo.py) не использует HKDF — `sha256(PIVO_ENCRYPTION_SECRET)` напрямую. Переход поломал бы существующие подписи. Это и привело к тому, что HKDF-инфраструктура `MembersService` оказалась изолированной от `/pivo`-flow и не пригодилась. Текущая модель: один секрет = один Fernet-ключ = одна доменная зона.

### 6.4. SQL-инъекции

Все запросы параметризованы. Проверено grep'ом. **Уязвимостей нет.**

### 6.5. Логирование

- `chat_id` **маскируется** через `mask_chat_id(...)` в `app/handlers/learning.py` (8 hex от HKDF-HMAC, session 22). Live smoke 2026-05-10 подтвердил: в логах видим `chat=1f6bfe93`, а не сырой id.
- `user_id` — не пишется.
- `pivo`-операции пишут только `mentions count: N` и `pivo command executed` — без идентификаторов.
- Уровень настраивается через `LOG_LEVEL` в `.env` (закрыто P2-6).

### 6.6. Bot privacy mode (старая 6.7)

Без изменений — README по-прежнему просит выключить privacy mode у бота, иначе `F.text` не получает групповые сообщения. Связанный privacy-trade-off задокументировать в README (см. п.9.1).

---

## 7. Качество кода

### 7.1. Что улучшилось

- Все import-блоки отсортированы (`ruff I`).
- Используются современные конструкции: `X | None` вместо `Optional[X]`, `from collections.abc import Awaitable, Callable` вместо `from typing` (`ruff UP`).
- Все строки ≤100 символов.
- `mypy strict` для `app/`: `disallow_untyped_defs`, `warn_return_any`, `no_implicit_optional` и пр.
- Нет `TODO` / `FIXME` / `XXX` в коде (проверено grep'ом).

### 7.2. Что осталось

Открытого долга по качеству кода нет. Все ранее зафиксированные пункты закрыты:

- ~~`range(4)` / `attempt < 2` в `learning.py`~~ → константы `MAX_GENERATION_ATTEMPTS`, `GENERATION_ATTEMPTS_WITH_CONTEXT` (session 20).
- ~~`len(clean) < 3 or > 500`~~ → константы `MIN_LEARN_MESSAGE_CHARS`, `MAX_LEARN_MESSAGE_CHARS`.
- ~~Дублирование Settings / RuntimeState / runtime_config~~ → `config_registry.py` (P2-1).
- ~~`BOT_TEXT_ALIASES = {"pepe", "пепе"}` хардкод~~ → `Settings.bot_text_aliases` с safe fallback (P2-5 / audit follow-ups).
- ~~`MENTION_RE = r"@\w+"` ломает email~~ → `(?<!\w)@\w+` (session 20).
- ~~`cursor.fetchone()[0]` в `save_message_and_update_model`~~ → `row = ...; assert row is not None` (session 22).

### 7.3. Мёртвый код — проверено явно

- `_ensure_messages_normalized_text_column` и `_anonymize_message_author_ids` — в коде их нет (логика переехала в миграции 002 и 003). ✅
- `MessagesRepo.exists()` — используется в [`MarkovGenerator.generate_text`](markov.py:670) для exact-match фильтра «не отдавать слово в слово сохранённое сообщение». Не мёртвый. ✅
- ~~`MembersService.record_consent` / `get_profile` / `revoke`~~ — удалены в session 21 вместе с `chat_member_profiles` и `app/security/key_derivation.py` как нереализованная инфраструктура.
- ~~`pivo_chat_members.is_bot`~~ — удалено вместе со старой таблицей в миграции 007 (session 21).

---

## 8. БД и хранение

### 8.1. Миграции

- 7 версионированных миграций (`001_initial.sql` … `007_unify_chat_members.sql`), каждая запускается ровно один раз.
- Резюм через таблицу `schema_migrations(name, applied_at)`.
- `.sql`-миграции выполняются через `sqlite3.executescript` внутри явного `BEGIN; ...; COMMIT;` — обёртка добавлена `migrator._apply` (sessions 18 + audit follow-ups). При исключении `run()` делает `conn.rollback()`, in-flight транзакция откатывается, в `schema_migrations` ничего не записывается.
- `.py`-миграции импортируют и вызывают `await mod.apply(conn)`; те же гарантии rollback применяются.
- Тесты на migrator (`tests/test_migrator.py`) покрывают: чистая БД, повторный init, partial resume, legacy-фикстуру (`tests/fixtures/legacy_real_schema.sql` от реальной prod-БД), full-data fixture (все таблицы заполнены, после миграций все row counts сохраняются), атомарность при битом statement, миграцию `pivo_chat_members` → `chat_members`.

### 8.2. Индексы

| Индекс | Использование | Замечание |
|---|---|---|
| `idx_chat_members_chat_hash` | `ChatMembersRepo.list_members(chat_hash)` | Создан миграцией 007 |
| `idx_messages_normalized_lookup(chat_id, normalized_text)` | `MessagesRepo.exists(chat_id, text)` — exact-match exclude в генераторе | OK |
| `idx_messages_chat_id` | `Database.get_stats / clear_chat / get_all_normalized_messages` | OK |

Индексы по `(chat_id, w1, w2)` на `transitions*` и `starts*` существуют через PRIMARY KEY и не требуют отдельных definition'ов.

### 8.3. Целостность

- Нет FK между `messages` и `transitions*` — это агрегаты, FK не нужны. ✅
- `chat_members` хранит только зашифрованные поля и HMAC-индексы. Структура минимальна: одна строка на (chat, user), без unused columns.
- Все cleanup-операции явно ограничены `WHERE chat_id = ?` или `WHERE chat_hash = ?`. Нет «глобальных» DROP/DELETE без области.

---

## 9. Конфигурация и запуск

### 9.1. README — **закрыто** (P2-3, `refactor/audit-p2-batch` + B3)

- Quickstart (`venv`, `pip`, `python main.py`) — есть.
- Команды — актуальные.
- Docker — есть.
- `/pivo` privacy — есть.
- Раздел «Тесты», «Архитектура» (со ссылкой на `docs/ARCHITECTURE.md`), «Privacy» — добавлены.
- Указание `requirements.lock` для воспроизводимости — есть.
- Абсолютные ссылки на `compose.yaml` / `.env.example` починены на относительные (B3).
- Стек обновлён до `Python 3.12+`.
- Инструкция по локальным проверкам (`ruff`, `mypy`, `unittest`) — есть.

### 9.2. `.env.example` — **актуален**

Полный, структурированный, с инструкцией по генерации секретов. Включает `LOG_LEVEL` (P2-6), `BOT_TEXT_ALIASES` (P2-5).

### 9.3. Dockerfile — **hardened** (P2-2 + audit follow-ups)

Текущее состояние:

```dockerfile
FROM python:3.14.0-slim            # pin minor
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd -m -u 1000 bot
COPY --chown=bot:bot requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY --chown=bot:bot . .
HEALTHCHECK CMD python -c "import sys; sys.exit(0)"
ENTRYPOINT ["/app/docker-entrypoint.sh"]    # root → chown bind-mount → runuser -u bot
CMD ["python", "main.py"]
```

`docker-entrypoint.sh` запускается под root, делает best-effort `chown -R bot:bot /app/data` (для bind-mount от root-owned директорий), затем `exec runuser -u bot -- "$@"`. `runuser` есть в `python:*-slim` (`util-linux`). `.gitattributes` фиксирует LF для `*.sh`, чтобы Windows-клон не сделал CRLF.

### 9.4. CI — стабильно

`.github/workflows/ci.yml`:

```yaml
- pip install -r requirements-dev.txt   # = lock + ruff + mypy + bandit + safety
- python -m ruff check app/ tests/
- python -m mypy app/
- python -m unittest discover tests -v
- python -m bandit -r app main.py db.py settings.py pivo.py markov.py -x tests --severity-level medium --confidence-level medium
- safety check -r requirements.lock
```

Матрица — Python 3.12/3.13/3.14. Triggers — push/PR в main. Branch protection активен (PR обязателен). Добавлен отдельный job `docker-build` (smoke build без запуска бота).

### 9.5. `requirements.lock` — **закрыто** (B2, `db1de05`)

`Dockerfile` устанавливает `requirements.lock` (вместо ранее использовавшегося `requirements.txt` с диапазонами), `requirements-dev.txt` ссылается на `-r requirements.lock`, в файл добавлен header с описанием стратегии (`pip freeze` без `pip-tools`) и процедурой регенерации. CI и Docker реально используют lock.

---

## 10. Логирование

- `LOG_LEVEL` валидируется в `Settings`, читается из `.env`, применяется в `main.py` (P2-6, закрыто).
- `aiogram` приглушён до `WARNING`.
- Формат — обычный текст: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`.
- Идентификаторы пользователей не пишутся; `chat_id` маскируется через `mask_chat_id(...)` (P3-4, закрыто session 22). Маска — 8 hex от HKDF-HMAC под `PIVO_HMAC_SECRET` с доменом `logging:chat_id`, стабильна между рестартами при одном секрете, меняется при ротации, не обратима к исходному `chat_id`.
- Корреляция между строками одного чата работает через стабильную маску.
- Нет аудита админ-команд (`/clear`, `/set`) в БД. Опциональная фича, не планируем без запроса.

**Отложено по внешним решениям:** структурированные JSON-логи (ждём выбора системы агрегации); ID-корреляции запросов (ждём метрик).

---

## 11. Тестирование

### 11.1. Состояние

Текущее число тестов: **264** (было 199 до security-audit remediation). 13 test-файлов:

| Файл | Что покрывает |
|---|---|
| `test_bot_messages.py` | форматирование `/help`, `/stats`, `/config` |
| `test_bot_policy.py` | `bot_is_mentioned`, `should_reply`, cooldown |
| `test_db_logic.py` | `save_message_and_update_model`, `get_stats`, `clear_chat`, `chat_members` upsert/get/remove, retention `pivo_daily_usage`, refund |
| `test_filters.py` | `GroupOnly`, `AdminOrOwner` (fail-closed на ошибке Telegram API), `ThrottlingMiddleware` (silent drop + `notify_on_throttle` UX) |
| `test_handlers.py` | happy-path по 4 роутерам, denied-fallback для unauthorized admin-команд, `TestReplyHumanizedResilience` (chat-action 5xx, /pivo_on under failure) |
| `test_learning_service.py` | prefix-cache, `matches_training_prefix` |
| `test_log_masking.py` | `init_masking`/`mask_chat_id`: длина, детерминизм, изменение при ротации секрета, fail-fast без init |
| `test_main.py` | smoke-тест на `configure_dispatcher` (роутеры, middleware, ключи в `dp[]`) |
| `test_markov_and_text.py` | генерация, токенизация, `sanitize_text`, email-safe `MENTION_RE` |
| `test_migrator.py` | пустая БД, повторный init, partial resume, legacy fixture, full-data fixture, real-schema fixture, атомарность `.sql`, миграция `pivo_chat_members` → `chat_members` |
| `test_pivo.py` | HMAC, Fernet, `build_pivo_mention`, E2E subscribe → call → quota → unsubscribe через реальный SQLite |
| `test_runtime_config.py` | `apply_runtime_setting` для всех ключей реестра |
| `test_settings.py` | `load_settings`, валидация ENV (включая `LOG_LEVEL`, `BOT_TEXT_ALIASES` с fallback) |

### 11.2. Покрытие модулей

| Модуль | Тесты | Качество покрытия |
|---|---|---|
| `app/handlers/*` | `test_handlers.py` | ✅ Happy-path, denied-fallback, resilience |
| `app/services/learning_service.py` | `test_learning_service.py` | ✅ Кэш, инвалидация, edge cases |
| `app/services/pivo_service.py` | `test_pivo.py` (`TestPivoServiceFlow`) | ✅ E2E flow через SQLite |
| `app/repositories/*` | `test_db_logic.py` | ✅ |
| `app/filters/*` | `test_filters.py` | ✅ Включая ошибки API `get_chat_administrators` |
| `app/middlewares/throttling.py` | `test_filters.py` | ✅ silent drop, notify_on_throttle, разные команды |
| `app/infrastructure/migrator.py` | `test_migrator.py` | ✅ Атомарность, real-prod fixture |
| `app/log_masking.py` | `test_log_masking.py` | ✅ 6 кейсов |

### 11.3. Дыры в покрытии

Закрыто:
- ~~E2E `/pivo`-flow~~ — `tests/test_pivo.py::TestPivoServiceFlow`.
- ~~Smoke-тест на wiring `main.py`~~ — `tests/test_main.py`.
- ~~Тесты на отказ авторизации с явным ответом~~ — denied-fallback handlers (B1).
- ~~Атомарность .sql-миграций~~ — `TestMigratorAtomicity` (audit follow-ups).
- ~~Поведение хендлеров при ошибках Telegram API~~ — `TestReplyHumanizedResilience` + refund на reply-error + fail-closed на admin-API (session 22).
- ~~`build_call_message` через DB~~ — фактически покрыт `TestPivoServiceFlow.test_subscribe_call_quota_and_unsubscribe_flow` (session 22).

Осталось:
- **Реальная гонка throttling** (одновременные вызовы) — не тестируется, но in-memory dict не имеет race conditions в asyncio (single-threaded), риск низкий. Не планируем.

---

## 12. Что хорошо сделано в проекте

- **Слоистая архитектура** — handlers / services / repositories / filters / middlewares / migrations / infrastructure. Каждая граница реальна (хендлеры не видят SQL, сервисы не видят aiogram кроме одного осознанного исключения).
- **Атомарные миграции** — `.sql` через `executescript` внутри `BEGIN; ...; COMMIT;`. При сбое в середине миграции — `conn.rollback()`, `schema_migrations` не получает запись о половинчатой. Покрыто `TestMigratorAtomicity` с битым statement.
- **Миграции на реальной БД** — проверены на фикстуре `tests/fixtures/legacy_real_schema.sql` (схема реального prod-`markov.db` пользователя), а не только на синтетических случаях. Все full-data fixtures сохраняют row counts after migration.
- **Миграция 007 (unification)** прошла на проде без потери подписок: live smoke 2026-05-10 подтвердил `chat_members` сохранил данные, старые таблицы исчезли.
- **HKDF для log-masking** — стабильная маска чатов в логах с domain-label `logging:chat_id`, отделена от `/pivo`-флоу. Ротация `PIVO_HMAC_SECRET` ломает корреляцию ID между «до/после» — это желаемое свойство.
- **Throttling middleware** правильно использует `TelegramObject` в сигнатуре `__call__` (LSP-compatible), внутри проверяет `isinstance(event, Message)`. `notify_on_throttle` даёт явный ответ для админ-команд, silent drop сохранён для `/pivo` по дизайну.
- **`AdminOrOwner.fail-closed`** — при ошибке Telegram API (бот не в чате, network blip) фильтр возвращает `False`, не `True`. Поведение покрыто тестом.
- **`reply_humanized.resilience`** — `bot.send_chat_action` 5xx не блокирует основной reply; покрыто 3 тестами + 1 для `/pivo_on` под этим сценарием.
- **Параметризованные SQL-запросы** — нет инъекций.
- **`author_id` принудительно анонимизирован** через миграцию 003 при первом обновлении старой БД; новые сообщения пишутся с `author_id=0`.
- **`messages.text` удалён** миграцией 005; хранится только `normalized_text` после `sanitize_text` (URL/mentions удалены, пробелы нормализованы, email-адреса сохранены).
- **Opt-in `/pivo`** поддерживает пользователей **без `@username`** (через `tg://user?id=...`) — образец для упоминаний.
- **Privacy-сообщение `/pivo_privacy`** не врёт: нет хранения сырого текста, нет нерасшифровываемых полей.
- **`pyproject.toml`** настроен с осмысленными overrides: legacy-модули (`markov.py`, `pivo.py`, `bot_messages.py` и т.д.) вынесены в `ignore_errors`. `app/` под `mypy strict`.
- **CI на 3 версиях Python** (3.12/3.13/3.14), branch protection требует PR.
- **Single source of truth для настроек** (`config_registry.py`): `Settings`, `RuntimeState`, `runtime_config.apply_runtime_setting` итерируются по `FieldSpec` × 20.
- **Lock-файл реально используется** Dockerfile и CI (не декорация).
- **Docker non-root** через root-entrypoint, который чинит права на bind-mount и делает `runuser -u bot`.

---

## 13. Готовность к продакшну — историческая запись

Раздел оставлен как историческая запись. Все три P1-блокера были закрыты коммитами `a0fae76` (B1), `db1de05` (B2), `1ef20a2` (B3). С тех пор ветка `refactor/structure` смержена в `main`, проект прошёл через P2-batch, audit follow-ups, session 20 (P3 polish), session 21 (`chat_members` unification) и session 22 (chat_id masking + P3 close-out).

Текущее состояние:
- **Branch protection** для `main` включён (требует PR; force-push заблокирован).
- **CI зелёный** на 3.12/3.13/3.14 на последнем коммите.
- **199 тестов** проходят локально, ruff/mypy без замечаний.
- **Live smoke 2026-05-10** подтвердил миграцию 007, `chat_members`, `/pivo`-flow, маскирование `chat_id` в логах.
- Открытого P0/P1/P2/P3-долга **нет**.

---

## 14. Оставшийся техдолг (актуальный список)

### P0 — пусто

Закрыт фазами 1–6.

### P1 — закрыты

| # | Описание | Закрыто в |
|---|---|---|
| **P1-A** (B1) | Вернуть явный отказ для unauthorized `/set`/`/setprob`/`/clear` | ✅ `a0fae76` |
| **P1-B** (B2) | Подключить `requirements.lock` к Dockerfile и CI | ✅ `db1de05` |
| **P1-C** (B3) | README: починить 2 абсолютные ссылки | ✅ `1ef20a2` |

### P2 — статус после P2-batch и audit follow-ups

| # | Описание | Статус |
|---|---|---|
| ~~**P2-1**~~ | Реестр настроек как один источник истины (Settings ↔ RuntimeState ↔ runtime_config) | ✅ закрыто в `refactor/audit-p2-batch` (`config_registry.py`) |
| ~~**P2-2**~~ | Migrator: атомарные транзакции при ошибке | ✅ закрыто в `refactor/audit-p2-batch` (executescript + rollback) и `fix/audit-followups` (BEGIN/COMMIT-обёртка для true atomicity) |
| ~~**P2-3**~~ | Dockerfile hardening (non-root, HEALTHCHECK, pin минорной версии) | ✅ закрыто в `refactor/audit-p2-batch` + `fix/audit-followups` (entrypoint с `chown` и `runuser` для bind-mount) |
| ~~**P2-4**~~ | README: разделы Tests/Architecture/Privacy, `Python 3.12+` | ✅ закрыто в `refactor/audit-p2-batch`. Структурированные JSON-логи остаются P3 |
| ~~**P2-5**~~ | Решить судьбу `MembersService` | ✅ закрыто в session 21 — `MembersService` / `MembersRepo` / `chat_member_profiles` / `key_derivation.py` удалены; роль канонической таблицы участников перешла к `chat_members` (миграция 007), её использует `/pivo` |
| ~~**P2-6**~~ | `LOG_LEVEL` из `.env` | ✅ закрыто в `refactor/audit-p2-batch` (валидируемый `Settings.log_level`) |
| ~~**P2-7**~~ | E2E-тест `/pivo` flow | ✅ закрыто в `codex/pivo-daily-quota` |
| ~~**P2-8**~~ | Smoke-тест на wiring `main.py` | ✅ закрыто в `codex/pivo-daily-quota` |
| ~~**P2-9**~~ | `/learn_off` opt-out | ❌ снят по продуктовому решению (session 21): после удаления `messages.text`, анонимизации `author_id` и маскирования `chat_id` чувствительной информации на уровне чата не остаётся; явный opt-out закрывал бы теоретический риск, не соответствующий реальной угрозе |
| ~~**P2-10**~~ | Расширить `BOT_TEXT_ALIASES` через `.env` | ✅ закрыто в `fix/audit-followups` с safe fallback на встроенные `{"pepe", "пепе"}` |

### P3 — все закрыты

Закрыто в session 20 (`polish/audit-session-20`):
- ~~Магические числа в `learning.py` (`range(4)`, `attempt < 2`)~~ → константы `MAX_GENERATION_ATTEMPTS`, `GENERATION_ATTEMPTS_WITH_CONTEXT`.
- ~~Концепция `is_duplicate` (privacy vs novelty)~~ → переименовано в `matches_training_prefix`, docstring переписан, мёртвая SQL-ветка удалена.
- ~~Silent drop `/clear` под throttle (R2)~~ → `notify_on_throttle` + явный ответ.
- ~~`MENTION_RE` на email~~ → `(?<!\w)@\w+` (regression-тесты).

Закрыто в session 21 (`feat/unify-chat-members`):
- ~~Неиспользуемое поле `is_bot` в `pivo_chat_members`~~ → миграция 007 не создаёт колонку.

Закрыто в session 22 (`feat/log-masking-and-p3-tests`):
- ~~Маскирование `chat_id` в логах~~ → `app/log_masking.py`, HKDF от `PIVO_HMAC_SECRET` с доменом `logging:chat_id`, 8 hex; live smoke подтвердил `chat=1f6bfe93` в логах.
- ~~Поведение хендлеров при ошибках Telegram API~~ → `TestReplyHumanizedResilience` (+3 теста) поверх уже существующих покрытий refund/fail-closed.
- ~~`build_call_message` через DB E2E~~ → фактически покрыт `test_subscribe_call_quota_and_unsubscribe_flow`.
- ~~`cursor.fetchone()[0]` стиль в `db.py`~~ → унифицирован с репозиториями (`row = ...; assert row is not None`).

Остаётся (заблокировано внешними решениями):
- **Структурированные логи (JSON)** — ждать выбора системы агрегации.
- **Метрики (Prometheus / aiogram-middleware)** — заводить когда будет endpoint.

### Что **не** делать сейчас

Без изменений с прошлого аудита:

- Не вводить тяжёлые фреймворки (FastAPI, Alembic, pydantic-settings) ради «чистоты».
- Не переезжать с SQLite на Postgres — нет потребности.
- Не переписывать `markov.py` — алгоритм работает и покрыт тестами.
- Не делать «универсальный» plugin-механизм — `aiogram.Router` достаточно.
- Не трогать `pivo_templates.py` — это контентный файл.
- Не ротировать `PIVO_*_SECRET` без миграционного скрипта — сломает существующие подписки.
- Не мигрировать на `pytest` — `unittest` работает и не привносит новых зависимостей.

---

## 15. Сводная таблица оставшихся проблем

Только открытые позиции. Закрытое см. в section 14 (P1/P2) и в session updates.

Открытых пунктов P0/P1/P2/P3 в трекере **не осталось**. Заблокированные внешним выбором (JSON-логи, метрики) учтены в section 14 как «отложено».

### Security-backlog (добавлен после аудита 2026-05-12)

| ID | Severity | Описание | Статус |
|---|---|---|---|
| AUD-007 | MEDIUM | Telegram API error middleware | ✅ Закрыто (PR `audit-security-debt-cleanup`) |
| AUD-004 | MEDIUM | DB retention policy для `messages`/transitions | ✅ Задокументировано в `docs/OPERATIONS.md` |
| AUD-011 | LOW | `assert` в runtime DB/service коде | ✅ Закрыто (PR `audit-security-debt-cleanup`) |
| AUD-010 | LOW | Отдельный `LOG_MASKING_SECRET` | Открыто (низкий приоритет) |
| AUD-006 | LOW | Улучшить Docker healthcheck | ✅ Закрыто (PR `audit-security-debt-cleanup`) |
| CA-F8 | LOW | `db_prod_copy/` в workspace | ✅ Obsolete: уже в `.gitignore`/`.dockerignore` |

---

**Дата актуализации:** 2026-05-15, одиннадцатая редакция.
**Статус:** ветка `audit-security-debt-cleanup`, **266 тестов**, ruff/mypy/bandit/pip-audit clean (30 source files).
**История ревизий:**
- 2026-05-08, 1я: первичный аудит, «мёржить после фиксов».
- 2026-05-08, 2я: Codex-ревизия, блокеры P1-A/B/C.
- 2026-05-08, 3я: блокеры закрыты (`a0fae76`, `db1de05`, `1ef20a2`), `refactor/structure` — merge-ready.
- 2026-05-09, 4я: `codex/pivo-daily-quota` — daily quota, hotfixes (DI conflict, clear cooldown, quota refund), 200 тестов, P2-7/P2-8 закрыты.
- 2026-05-10, 5я: `refactor/audit-p2-batch` — P2-1, P2-2, P2-3, P2-6 закрыты, P2-4 частично, debug-лог успешной генерации, 203 теста.
- 2026-05-10, 6я: `fix/audit-followups` — P2-5 закрыто (BOT_TEXT_ALIASES с safe fallback), P1-фиксы атомарности миграций (BEGIN/COMMIT-обёртка) и Docker bind-mount (root-entrypoint + runuser), 208 тестов.
- 2026-05-10, 7я (PR #5, `polish/audit-session-20`): P3-полировка — `matches_training_prefix` (rename + honest docstring + removed dead branch), магические числа → константы, `MENTION_RE` для email, `notify_on_throttle` UX-фикс silent drop `/clear`. 213 тестов.
- 2026-05-10, 8я (`feat/unify-chat-members`): унификация участников — миграция 007 переносит `pivo_chat_members` → `chat_members` (без `is_bot`), `chat_member_profiles` / `MembersService` / `MembersRepo` / `key_derivation.py` удалены, `PivoRepo` → `ChatMembersRepo`. Закрыто P2-5, P3-5, снят P2-9. 190 тестов (-23 за счёт удалённых dead-code тестов).
- 2026-05-10, 9я (`feat/log-masking-and-p3-tests` + live smoke + полный re-sync аудита): закрыт остаточный P3 — `chat_id` маскируется в логах (HKDF от `PIVO_HMAC_SECRET`), добавлены тесты на ошибки Telegram API, стиль `cursor.fetchone()` унифицирован. Live smoke на проде подтвердил миграцию 007 и маскирование. Аудит полностью переписан под актуальное состояние (sections 0–15). 199 тестов.
- 2026-05-15, 10я (security/stability audit + remediation PRs #19–23 + pip-audit clean): проведён отдельный security audit (`PROJECT_SECURITY_STABILITY_AUDIT.md`), закрыты AUD-009, AUD-001×2, AUD-002, AUD-003, AUD-005, AUD-006 (runbook), CI bandit/safety/docker-smoke; стабилизирован prefix-cache window тест. 264 теста. pip-audit: нет уязвимостей. Открытый security-backlog (LOW): AUD-007, AUD-004, AUD-011, AUD-010, AUD-006 (healthcheck).
- 2026-05-15, 11я (security debt cleanup PR `audit-security-debt-cleanup`): закрыты AUD-007 (`app/handlers/errors.py` + тесты), AUD-011 (20 assert → RuntimeError в `db.py`/`markov_repo.py`), AUD-004 (retention runbook в `docs/OPERATIONS.md`), AUD-006 (Dockerfile healthcheck → SQLite SELECT 1). 266 тестов. Открыто только: AUD-010, CA-F11.

> SHA коммитов обновлены после очистки истории `git filter-repo` (2026-05-09): удалены строки атрибуции из 28 коммитов.

---

## 16. Session update — 2026-05-09

### Completed
- Очистка истории git: удалены строки атрибуции из 28 коммитов через `git filter-repo`.
- Force-push перезаписанных веток на GitHub: `main` и `codex/pivo-daily-quota`.
- `PROJECT_AUDIT.md` актуализирован: новые SHA, текущая ветка, обновлённые метрики (200 тестов), закрыты P2-7/P2-8.

### Audit findings updated
- SHA всех коммитов, упомянутых в аудите, обновлены на новые значения после rewrite.
- Статус P2-7 и P2-8 — закрыты.

### Not run / limitations
- Тесты не запускались в этой сессии.
- `docs/ARCHITECTURE.md` по-прежнему устарел — описывает пре-рефакторинговую структуру.
- `refactor/structure` на remote не существует; если нужна — отдельно запушить.

### Remaining work
- Включить обратно branch protection для `main` на GitHub.
- P2-1 (тройное дублирование Settings/RuntimeState/runtime_config) — главный техдолг.
- `docs/ARCHITECTURE.md` — переписать под текущую архитектуру.
- Live-тест `codex/pivo-daily-quota` и merge в `main`.

---

## 24. Session update — 2026-05-15 (security debt cleanup)

### Ветка

`audit-security-debt-cleanup` — один PR с четырьмя изменениями.

### AUD-007 — Централизованный Telegram API error middleware

Создан `app/handlers/errors.py` с `@router.error()` хендлером:
- `TelegramAPIError` → `logger.error("Telegram API error in handler: %s", exc)`
- любое другое исключение → `logger.error("Unhandled exception in handler", exc_info=exc)`

Router подключён в `main.py:configure_dispatcher` последним после всех других роутеров.
Добавлены 2 теста в `tests/test_error_handler.py`. Обновлён assertion в `tests/test_main.py`
(список sub_routers).

### AUD-011 — Замена assert на RuntimeError

`db.py`:
- 2 COALESCE-гарантированных assert (`row3`, `row2` в `save_message_and_update_model`) →
  `if x is None: raise RuntimeError("COALESCE query returned None ...")`
- 8 инициализационных assert `self.markov is not None` → `if self.markov is None: raise RuntimeError(...)`
- 2 assert `self.messages is not None` → аналогично
- 3 assert `self.chat_members is not None` → аналогично
- 3 assert `self.pivo_usage is not None` → аналогично

`app/repositories/markov_repo.py`:
- 2 COALESCE-гарантированных assert в `get_chat_token_volume` → RuntimeError

Итого: 20 assert заменены. Поведение под `-O` теперь корректно: сбой становится явным
`RuntimeError`, а не тихим `AssertionError` или `None`-dereference.

### AUD-004 — Документация retention policy

В `docs/OPERATIONS.md` добавлен раздел `## Database Retention` с:
- командой проверки размера таблиц через `dbstat`
- примером DELETE + VACUUM для ручной очистки сообщений на чат
- рекомендованным графиком обслуживания
- пояснением по transitions/starts (сброс через `/clear`)

### AUD-006 — Docker healthcheck

`Dockerfile` строки 35–36:
```dockerfile
# было:
CMD python -c "import sys; sys.exit(0)"
# стало:
CMD python -c "import sqlite3, os; sqlite3.connect(os.getenv('DB_PATH', 'data/markov.db')).execute('SELECT 1')"
```

Healthcheck теперь проверяет, что файл БД существует и открывается. `DB_PATH` из env.
`start-period=20s` гарантирует отсутствие false-positive до создания БД при первом старте.

### Проверки

| Команда | Результат |
|---|---|
| `python -m ruff check app/ tests/ main.py` | **clean** |
| `python -m mypy app/` | **clean** (30 source files) |
| `python -m unittest discover tests -v` | **266 тестов OK** |
| `python -m bandit -r app main.py db.py ... --severity-level medium` | **0 Medium/High** |

### Changed files

- Новые: `app/handlers/errors.py`, `tests/test_error_handler.py`
- Изменены: `main.py`, `db.py`, `app/repositories/markov_repo.py`, `Dockerfile`,
  `docs/OPERATIONS.md`, `tests/test_main.py`, `PROJECT_AUDIT.md`

### Remaining work

- AUD-010 (LOG_MASKING_SECRET) — открыто, низкий приоритет.
- CA-F11 (синхронизация `.env`) — открыто, не проверяется автоматически.
- JSON-логи, Prometheus-метрики — ждут внешних решений.

---

## 23. Session update — 2026-05-15 (security audit remediation + pip-audit clean)

### Контекст

Полный security/stability audit был проведён 2026-05-12 на ветке `audit-security-stability-review`
и задокументирован в `PROJECT_SECURITY_STABILITY_AUDIT.md`. Затем было выполнено 5 remediation PR
(#19–#23), каждый с полным набором проверок. Детали — в `PROJECT_AUDIT_CODEX.md`.

### Синхронизация с GitHub (2026-05-15)

- Ветка `fix-prefix-cache-window-test` (PR #23) слита в `main` на remote.
- Локальная ветка полностью синхронизирована с `origin/fix-prefix-cache-window-test`.
- Состояние: clean (untracked: только `prompt.md`).
- Remote `main`: commit `5218850` (merge PR #23).

### Закрытые security-находки (remediation PRs #19–#22)

| ID | Ветка / PR | Что сделано |
|---|---|---|
| AUD-009 | audit-security-stability-review | `cryptography` обновлён до `46.0.7`, `requirements.lock` пересобран. Safety перестал находить CVE. |
| AUD-001 (explicit mentions) | audit-tasklist-remediation | Добавлен лимит `PIVO_EXPLICIT_MENTIONS_LIMIT=10` в `settings.py` и `.env.example`; `pivo_parser.py` обрезает список явных упоминаний. |
| AUD-001 (subscriber fanout) | audit-tasklist-remediation | Добавлен лимит `PIVO_SUBSCRIBER_FANOUT_LIMIT=20`; `pivo_service.py` усекает список подписчиков и сообщает об усечении. |
| AUD-002 | audit-runtime-state-bounds (PR #21) | `RuntimeState` и `ThrottlingMiddleware` ограничены TTL (`RUNTIME_STATE_TTL_SEC`, `THROTTLE_STATE_TTL_SEC`) и ёмкостью (`RUNTIME_STATE_MAX_CHATS`, `THROTTLE_STATE_MAX_KEYS`). Добавлены `note_chat_activity`, `prune_inactive`, `forget_chat`. |
| AUD-003 | audit-prefix-cache-bounds (PR #22) | Prefix-cache строится только из последних `PREFIX_CACHE_MAX_MESSAGES=2000` сообщений чата. Полный rebuild заменён bounded-window запросом. |
| AUD-005 / CA-F7 | audit-runbook-ci-followups (PR #19) | `.dockerignore` зеркалирует паттерны `.gitignore`: `db_prod_copy/`, `.test_tmp/`, `Screenshot_*.jpg`. |
| AUD-006 (runbook) | audit-runbook-ci-followups (PR #19) | Добавлен `docs/OPERATIONS.md` с инструкциями по логам, WAL checkpoint, backup/restore. |
| CA-F1 / AUD-005 | audit-tasklist-remediation | `GroupOnly()` добавлен ко всем `/pivo*` хендлерам, regression-тест на отказ в приватном чате. |
| CA-F2 | audit-tasklist-remediation | `README.md` синхронизирован (убран HKDF из описания `/pivo`, исправлен privacy-блок про `chat_id`). |
| CA-F3 / CA-F4 | audit-docs-env-parity (PR #20) | `README.md` quickstart разделён на runtime/development install. `docs/ARCHITECTURE.md` обновлён (CI с bandit/safety/docker-smoke). |
| CI bandit | audit-runbook-ci-followups (PR #19) | `bandit` и `safety` добавлены в `ci.yml`. |
| CI docker-build | audit-runbook-ci-followups (PR #19) | Добавлен отдельный CI job `docker-build` (smoke build без запуска бота). |

### Стабилизация теста (PR #23)

Ветка `fix-prefix-cache-window-test` — единственное изменение: в `tests/test_learning_service.py`
добавлен `unittest.mock.patch("app.services.learning_service.random.randint", return_value=4)`
вокруг теста `test_prefix_cache_window`. Ранее тест мог флапать из-за нестабильного random при
проверке, что «старый» префикс выпал за пределы окна, а «новый» остался. Mock убирает
недетерминизм: `randint` всегда возвращает 4 (длина prefix-window), что гарантирует поиск по
полному диапазону токенов.

### Верификационные проверки (2026-05-15)

| Команда | Результат |
|---|---|
| `python -m ruff check app/ tests/ main.py settings.py runtime_state.py` | **clean** |
| `python -m mypy app/` | **clean** (29 source files) |
| `python -m unittest discover tests -v` | **264 тестов OK** (27.3s) |
| `python -m bandit -r app main.py settings.py runtime_state.py -x tests --severity-level medium` | **0 Medium/High** (13 Low) |
| `pip-audit -r requirements.lock` (с PYTHONUTF8=1) | **No known vulnerabilities** |
| `safety check -r requirements.lock` | Deprecated command, но уязвимостей не найдено |

**Не запущено / ограничения:**
- `safety scan` (новый синтаксис) — требует `safety auth`. Не выполнялся; `pip-audit` дал clean.
- Live smoke на Telegram — не проводился в данной сессии.
- Docker build — не выполнялся локально (CI запускает в ubuntu).

### Текущие установленные версии зависимостей

| Пакет | Версия | Примечания |
|---|---|---|
| `cryptography` | 46.0.7 | AUD-009 закрыт; pip-audit: уязвимостей нет |
| `aiogram` | 3.27.0 | В пределах `>=3.7.0,<4.0.0` |
| `aiosqlite` | 0.22.1 | В пределах `>=0.20.0,<1.0.0` |
| `python-dotenv` | 1.2.2 | В пределах `>=1.0.1,<2.0.0` |

### Открытый security-backlog (LOW-приоритет)

Из `AUDIT_TASKLIST.md` — нерешённые пункты:

| ID | Приоритет | Описание | Статус |
|---|---|---|---|
| AUD-007 | MEDIUM | Централизованный Telegram API error middleware | ✅ Закрыто (`audit-security-debt-cleanup`) |
| AUD-004 | LOW | Политика retention для таблиц `messages`/transitions | ✅ Задокументировано в `docs/OPERATIONS.md` |
| AUD-011 | LOW | Заменить `assert` на runtime-исключения в `db.py` и `markov_repo.py` | ✅ Закрыто (`audit-security-debt-cleanup`) |
| AUD-010 | LOW | Отдельный `LOG_MASKING_SECRET` для стабильной корреляции логов | Открыто (низкий приоритет) |
| AUD-006 | LOW | Улучшить Docker healthcheck | ✅ Закрыто (`audit-security-debt-cleanup`) |
| CA-F8 | LOW | `db_prod_copy/` в workspace — решить судьбу | ✅ Obsolete: уже в `.gitignore`/`.dockerignore` |
| CA-F11 | LOW | Синхронизировать локальный `.env` с `.env.example` | Открыто (не проверяется в аудите) |

**Закрыто / obsolete:**
- CA-F10 (Screenshot_*.jpg): файлы отсутствуют в workspace — **Obsolete / Resolved**.

### Новых архитектурных находок нет

Структура кода не изменилась между 2026-05-10 и 2026-05-15 (все изменения — тесты и конфиги).
Оценка из разделов 5–12 остаётся в силе.

### Changed files (в рамках сессии 23)

- `PROJECT_AUDIT.md` — обновлён (этот файл).

---

## 22. Session update — 2026-05-10 (chat_id masking + P3 close-out)

### Completed
- **P3-4 chat_id masking.** Новый модуль `app/log_masking.py`:
  `init_masking(secret)` выводит HKDF-ключ от `PIVO_HMAC_SECRET`
  с доменом `logging:chat_id` (32 байта), `mask_chat_id(chat_id)`
  возвращает первые 8 hex-символов HMAC-SHA256. Маска стабильна между
  запусками с одним секретом, меняется при ротации, не содержит сырой
  `chat_id`. Вызов `mask_chat_id` без `init_masking` поднимает
  `LogMaskingNotInitialized` — fail-fast вместо silent leak.
  `main.run_bot` вызывает `init_masking` сразу после `load_settings`.
  Все 9 `logger.{info,debug}` вызовов в `app/handlers/learning.py`
  заменены: `chat=%s` теперь принимает маску. 6 unit-тестов на helper.
- **P3-7 поведение хендлеров при ошибках Telegram API.** Добавлены
  3 теста в `tests/test_handlers.py::TestReplyHumanizedResilience`:
  `send_chat_action` 5xx не блокирует основной reply,
  `send_chat_action` вызывается когда `bot` присутствует, `/pivo_on`
  по-прежнему подписывает пользователя даже если chat-action провалился.
  Уже существующие покрытия: refund квоты при провале `message.reply`
  в `/pivo` (`test_pivo_refunds_quota_when_reply_fails`), fail-closed
  `AdminOrOwner` при падении `bot.get_chat_administrators`
  (`test_api_error_falls_back_to_denied`).
- **P3-8 build_call_message E2E.** Подтверждено, что фактически уже
  закрыт `TestPivoServiceFlow.test_subscribe_call_quota_and_unsubscribe_flow`
  — тест ходит через `Database` → `ChatMembersRepo` → `chat_members`
  таблицу с реальным sqlite. Снимаю с backlog'а.
- **7.1 стиль `cursor.fetchone()[0]` в `db.py:165-174`.** Заменено на
  `row = await cursor.fetchone(); assert row is not None; int(row[0])`
  — то же, что в репозиториях.

### Changed files
- Новый: `app/log_masking.py`, `tests/test_log_masking.py`.
- Изменены: `main.py`, `app/handlers/learning.py`, `db.py`,
  `tests/test_handlers.py`, `PROJECT_AUDIT.md`.

### Audit findings updated
- P3-4 (chat_id masking) — закрыто.
- P3-7 (Telegram API errors в хендлерах) — закрыто.
- P3-8 (E2E `build_call_message`) — закрыто (фактическое покрытие).
- 7.1 (стиль `cursor.fetchone()[0]`) — закрыто.

### Tests/checks run
- `python -m unittest discover tests` — **199 тестов OK** (было 190, +9:
  +6 на log_masking, +3 на reply_humanized resilience).
- `python -m ruff check app/ tests/` — clean.
- `python -m mypy app/` — clean (27 source files).

### Not run / limitations
- Live smoke: маскированные `chat_id` в логах не наблюдались на
  реальном чате. Запускать в проде → подтвердить, что в `learning.py`
  пишется маска, а не сырой `chat_id`.

### Remaining work
- Структурированные JSON-логи — ждать выбора системы агрегации.
- Метрики (Prometheus / aiogram-middleware) — заводить когда будет
  endpoint.
- P0/P1 — пусто. P2 — пусто. P3 — пусто (за исключением «отложить до
  внешних решений» сверху).

---

## 21. Session update — 2026-05-10 (chat_members unification)

### Completed
- **P2-5** закрыто. `chat_member_profiles` / `MembersService` / `MembersRepo` /
  `app/security/key_derivation.py` удалены: эта инфраструктура была заделом
  «на будущие HKDF-домены», которые не материализовались, поэтому несла
  только стоимость поддержки. Канонической таблицей участников теперь
  является `chat_members` — единственный потребитель сейчас `/pivo`
  (присутствие в таблице ≡ подписка), будущие фичи с персистентным
  состоянием участника пишут сюда же.
- **P3-5** закрыто. Поле `is_bot` в новой таблице отсутствует:
  `cmd_pivo_on` отсекает ботов до записи, так что колонка читалась только
  defensively в `build_pivo_mention`. Защита упрощена до уровня «всё, что
  лежит в таблице — это пользователь».
- **Миграция 007_unify_chat_members.sql:** CREATE TABLE `chat_members` →
  копирование строк из `pivo_chat_members` → DROP `pivo_chat_members` /
  `chat_member_profiles` и связанных индексов → CREATE INDEX
  `idx_chat_members_chat_hash`. Атомарна (BEGIN/COMMIT-обёртка миграционного
  раннера). Существующие /pivo-подписки сохранены без re-encrypt, потому
  что ключи (HMAC под `PIVO_HMAC_SECRET`, Fernet под
  `sha256(PIVO_ENCRYPTION_SECRET)`) не меняются.
- **`PivoRepo` → `ChatMembersRepo`**. Методы по-прежнему `upsert / remove /
  list_members`, но без `is_bot`. Таблица переименована, репозиторий
  тоже. `db.py` facade методы переименованы:
  `upsert_pivo_member` → `upsert_chat_member`,
  `get_pivo_members` → `get_chat_members`,
  `remove_pivo_member` → `remove_chat_member`.
- `PivoMember` dataclass без `is_bot`. `build_pivo_mention` /
  `collect_pivo_mentions` упрощены. Удалены теперь-нерелевантные
  unit-тесты `test_build_pivo_mention_skips_bots` /
  `test_collect_pivo_mentions_excludes_bots`.
- **Документация:** `docs/ARCHITECTURE.md` и `README.md` обновлены.

### Changed files
- Новый: `app/migrations/007_unify_chat_members.sql`,
  `app/repositories/chat_members_repo.py`.
- Удалены: `app/repositories/pivo_repo.py`,
  `app/repositories/members_repo.py`, `app/services/members_service.py`,
  `app/security/key_derivation.py`, `app/security/__init__.py`,
  `tests/test_members.py`.
- Изменены: `app/repositories/__init__.py`, `app/services/pivo_service.py`,
  `db.py`, `pivo.py`, `tests/test_pivo.py`, `tests/test_db_logic.py`,
  `tests/test_migrator.py`, `docs/ARCHITECTURE.md`, `README.md`,
  `PROJECT_AUDIT.md`.

### Audit findings updated
- P2-5 (MembersService без runtime) — закрыто.
- P3-5 (неиспользуемое `is_bot`) — закрыто.
- P2-9 (`/learn_off` opt-out) — снят как пункт по продуктовому решению:
  после анонимизации `author_id`, удаления `messages.text` и предстоящего
  маскирования `chat_id` чувствительной информации на уровне чата
  фактически не остаётся; явный opt-out закрывает теоретический риск,
  не отвечающий реальной угрозе.

### Tests/checks run
- `python -m unittest discover tests` — **190 тестов OK**.
- `python -m ruff check app/ tests/` — clean.
- `python -m mypy app/` — clean (26 source files, было 30).

### Not run / limitations
- Live smoke в Telegram не проводился. Рекомендуется:
  существующий /pivo-чат → проверить, что `/pivo_on` / `/pivo` / `/pivo_off`
  по-прежнему работают (миграция выполнится при первом запуске).
- Docker build не выполнялся.

### Remaining work
- **P3-4** маскирование `chat_id` в логах — будет в следующей сессии.
- **P3-7** тесты на ошибки Telegram API в хендлерах.
- **P3-8** E2E `PivoService.build_call_message` через DB (фактически уже
  покрыт `TestPivoServiceFlow.test_subscribe_call_quota_and_unsubscribe_flow`
  — можно пересмотреть и закрыть в аудите при подтверждении).
- **7.1** стиль `cursor.fetchone()[0]` в `db.py:165-174`.

---

## 20. Session update — 2026-05-10 (P3 polish + throttle UX)

### Completed
- **P3 (7.1) magic numbers in handler:** `app/handlers/learning.py` — введены
  module-level константы `MAX_GENERATION_ATTEMPTS = 4` и
  `GENERATION_ATTEMPTS_WITH_CONTEXT = 2`; `for attempt in range(4)` и
  `attempt < 2` заменены на ссылки на константы. Поведение идентично.
- **6.1 conceptual ambiguity `is_duplicate`/`looks_too_close_to_training_sample`:**
  метод переименован в `matches_training_prefix`. Docstring переписан:
  фильтр явно описан как второстепенный novelty-хук поверх основных
  защит в `MarkovGenerator.generate_text` (`is_low_diversity_reply`,
  `is_context_heavy_reply`, exact-match `message_exists`). Удалена
  мёртвая ветка `if len(tokens) < 3: return message_exists(...)`:
  exact-match уже отрабатывает в финальной проверке генератора, и
  кандидат с <3 токенами не может пройти туда же по prefix-логике.
  Случайность префикса (`random.randint(3, min(5, ...))`) оставлена
  по решению пользователя — это сознательный novelty-nudge. Обновлены
  все вызовы и тесты (`tests/test_learning_service.py`).
- **P3 (7.6) MENTION_RE на email:** `text_utils.py` — паттерн обновлён до
  `(?<!\w)@\w+`. Теперь `user@example.com` остаётся `user@example.com`
  после `sanitize_text`, а `@bot привет` по-прежнему чистится.
  Добавлены два regression-теста в `tests/test_markov_and_text.py`.
- **R2 (silent drop /clear под throttle):** `ThrottlingMiddleware` получил
  параметр `notify_on_throttle: set[str] | None`. Команды в этом наборе
  при throttle получают короткий ответ «Слишком часто. Подождите ~N сек.»
  вместо silent drop. `/clear` добавлен в notify-set из `main.py`; `/pivo`
  silent-drop сохранён по дизайну (шумная команда). Добавлены 3 теста на
  новое поведение, существующий тест `test_repeated_clear_confirm_is_throttled`
  сохранён (без notify-set он остаётся silent — это backward-compatible
  дефолт). Тест отдельно фиксирует, что `/clear confirm` тоже уведомляет,
  когда `clear` в notify-set.
- **`seed_diverse.py`:** docstring почищен (убрана строка «локальный файл,
  в репозиторий не коммитим»), описано назначение скрипта. Добавлен в git
  как полезный seed-скрипт для smoke-тестирования.

### Changed files
- `app/handlers/learning.py`
- `app/services/learning_service.py`
- `app/middlewares/throttling.py`
- `text_utils.py`
- `main.py`
- `seed_diverse.py`
- `tests/test_learning_service.py`
- `tests/test_markov_and_text.py`
- `tests/test_filters.py`
- `PROJECT_AUDIT.md`

### Audit findings updated
- 7.1 magic numbers in `learning.py` — закрыто (для `range(4)`/`attempt < 2`;
  длины 3..500 уже были константами).
- 7.6 `MENTION_RE` на email — закрыто.
- 6.1 концептуальная неоднозначность `is_duplicate` — закрыто.
- R2 silent drop `/clear` под throttle — закрыто.

### Tests/checks run
- `python -m unittest discover tests` — **213 тестов OK** (было 208, +5).
- `python -m ruff check app/ tests/ text_utils.py main.py seed_diverse.py` — clean.
- `python -m mypy app/` — clean (30 source files).

### Not run / limitations
- Live smoke в Telegram не проводился — поведение mention-фильтра и throttle
  UX желательно проверить вручную на тестовом чате.
- Docker build не выполнялся.

### Remaining work
- P3-долг частично закрыт. Осталось:
  - 7.5 паттерн `cursor.fetchone()[0]` в `db.py:160-162` — стилистическая
    мелочь, можно подтянуть к стилю репозиториев (`assert row is not None`).
  - Маскирование `chat_id` в логах через HMAC — требует решения по схеме.
  - Неиспользуемое поле `is_bot` в `pivo_chat_members` — миграция +
    репозиторий, риск выше пользы.
- P2-9 `/learn_off` opt-out — отдельная фича, не в этой сессии.
- Структурированные JSON-логи — ждать выбора системы агрегации.

---

## 19. Session update — 2026-05-10 (audit followups)

### Completed (post-P2-batch fixes after independent review)
- **Migration atomicity** (P1 от ревью): `migrator._apply` теперь оборачивает
  тело `.sql`-миграции в `BEGIN; ... COMMIT;` перед передачей в
  `executescript`. Это закрывает дыру, описанную в ревью: stdlib
  `sqlite3.executescript` неявно делает `COMMIT` перед запуском скрипта,
  поэтому без явного BEGIN каждый DDL внутри миграции
  авто-коммитился по мере выполнения, и при падении в середине файла
  половина схемы оставалась применённой, а в `schema_migrations`
  записи не было — следующий старт повторял миграцию и снова падал.
  Теперь `run()` ловит исключение, делает `conn.rollback()`,
  in-flight-транзакция откатывается. Добавлен класс тестов
  `TestMigratorAtomicity` (2 теста) с временным `.sql`-файлом, в котором
  второй statement битый. Коммит `af8cac3`.
- **Docker bind-mount writability** (P1 от ревью): `USER bot` снят из
  Dockerfile. Добавлен `docker-entrypoint.sh`, который при старте от
  root делает best-effort `chown -R bot:bot /app/data`, затем
  `exec runuser -u bot -- "$@"`. `runuser` уже есть в `python:*-slim`
  (`util-linux`). Если контейнер запускается с `--user 1000:1000`, то
  entrypoint просто пропускает команду как есть. Добавлен
  `.gitattributes` с `*.sh text eol=lf`, чтобы Windows-клон не сделал
  CRLF в shell-скрипте. README обновлён. Коммит `34d20b9`.
- **P2-5**: `BOT_TEXT_ALIASES` через `.env` с **безопасным fallback'ом**.
  `bot_policy.DEFAULT_BOT_TEXT_ALIASES = frozenset({"pepe", "пепе"})`
  остаётся в коде; `Settings.bot_text_aliases` читает CSV из env и
  при пустом/незаданном значении подставляет defaults. Это критично
  для эксплуатации, когда у оператора нет доступа к
  `.env`/контейнеру — бот всегда отвечает на свои стандартные
  прозвища. Сигнатура `bot_is_mentioned` расширена 4-м аргументом
  с дефолтом, поэтому существующие прямые вызовы из тестов остаются
  совместимыми. +3 теста на settings, +1 на проброс через диспетчер.
  Коммит `b50f580`.

### Changed files
- `app/infrastructure/migrator.py`, `tests/test_migrator.py`
- `Dockerfile`, `docker-entrypoint.sh` (new), `.gitattributes` (new), `README.md`
- `bot_policy.py`, `settings.py`, `main.py`, `app/handlers/learning.py`,
  `.env.example`, `tests/test_settings.py`, `tests/test_main.py`
- `PROJECT_AUDIT.md`

### Tests/checks run
- `python -m unittest discover tests` — **208 тестов OK** (было 205, +3 на atomicity и settings).
- `python -m ruff check app/ tests/ <root modules>` — clean.
- `python -m mypy app/` — clean (30 source files).

### Not run / limitations
- `docker build .` локально не выполнялся (нет docker daemon).
  Ручная проверка на стороне оператора рекомендуется: чистая сборка,
  затем `docker run` с bind-mount от root-owned `./data` — entrypoint
  должен поправить владельца и стартовать без ошибок.
- Удалённый CI после этих коммитов ещё не проверялся.

### Remaining work (без изменений с P2-batch)
- P3-долг (концептуальная неоднозначность `looks_too_close_to_training_sample`,
  silent drop `/clear` под throttle, магические числа в `learning.py`,
  маскирование `chat_id` в логах).
- Структурированные JSON-логи (P3, ждать выбора системы агрегации).

---

## 18. Session update — 2026-05-10 (P2 batch)

### Completed
- **P2-6**: `LOG_LEVEL` вынесен в `Settings` + `.env.example`; убран хардкод
  `logging.INFO` из `main.py`. Поддерживаемые значения:
  `DEBUG/INFO/WARNING/ERROR/CRITICAL`. Тесты на default/lowercase/reject
  добавлены в `tests/test_settings.py`. Коммит `2aba935`.
- **P2-3**: `Dockerfile` — pin `python:3.14.0-slim`, non-root user `bot`
  (UID 1000), HEALTHCHECK, `--chown` на COPY. Коммит `b01e12d`.
- **P2-2**: `app/infrastructure/migrator.py` — наивный
  `str.split(";")` заменён на `conn.executescript`; `_apply` обёрнут
  в `try/except` с `conn.rollback()` на ошибке. Коммит `e8dc8e5`.
- **P2-1**: `config_registry.py` создан как single source of truth для
  20 runtime-mutable полей. `Settings.load_settings`,
  `runtime_state_from_settings` и `runtime_config.apply_runtime_setting`
  итерируются по реестру; cross-field-инварианты вынесены в
  `validate_cross_fields` и проверяются на shallow copy перед мутацией
  state. Публичный API `runtime_config` сохранён, все тесты остаются
  совместимыми. Коммит `4559cd6`.
- **P2-4**: README — добавлены секции Architecture (со ссылкой на
  `docs/ARCHITECTURE.md`), Privacy и Тесты; стек обновлён до
  `Python 3.12+`. Коммит `50eb8c2`.
- **`docs/ARCHITECTURE.md`**: переписан под текущую слоистую структуру
  (диаграмма слоёв, DI через `dp[...]`, config_registry, миграции с
  `executescript`). Коммит `82ba68e`.

### Changed files
- `settings.py`, `runtime_state.py`, `runtime_config.py`, `config_registry.py` (новый),
  `main.py`, `Dockerfile`, `app/infrastructure/migrator.py`,
  `tests/test_settings.py`, `.env.example`, `README.md`,
  `docs/ARCHITECTURE.md`, `PROJECT_AUDIT.md`.

### Audit findings updated
- P2-1, P2-2, P2-3 — закрыты.
- P2-4 — частично (LOG_LEVEL ✅, структурированные логи остаются P3).
- P2-6 — закрыт (был под старым номером, см. секцию 14).

### Tests/checks run
- `python -m unittest discover tests` — 203 теста OK (было 200, +3 на LOG_LEVEL).
- `python -m ruff check app/ tests/ settings.py runtime_state.py runtime_config.py config_registry.py main.py` — clean.
- `python -m mypy app/` — clean (30 source files).
- Линтеры запускались на dev-зависимостях `requirements-dev.txt`
  (`ruff 0.15.12`, `mypy 2.0.0`).

### Not run / limitations
- Docker build не выполнялся (нет docker daemon в среде).
- Удалённый CI на этой ветке ещё не проверялся.
- Live smoke в Telegram — не проводился, поведение runtime не
  затронуто (config_registry сохраняет contract).

### Remaining work
- Открыть PR `refactor/audit-p2-batch → main`, дождаться CI.
- P2-5 (`BOT_TEXT_ALIASES` из `.env`) и P2-9 (`/learn_off` opt-out) —
  следующие кандидаты.
- P3-долг (магические числа, `is_duplicate` концептуальная неоднозначность,
  silent drop `/clear` под throttle, маскирование `chat_id` в логах,
  `is_bot` поле в `pivo_chat_members`, `MENTION_RE` на email) — без изменений.
- Структурированные логи (JSON) откладываются до выбора системы агрегации.

---

## 17. Session update — 2026-05-09 (live smoke investigation)

### Completed
- Проведено расследование live-поведения бота в тестовом чате.
- Подтверждено: все три симптома из тестов — не баги кода.

### Findings

**Симптом 1: "пепе" и обычные сообщения — 0 реакции.**
Причина: privacy mode Telegram кэшируется на уровне конкретного чата. Изменение настройки в BotFather вступает в силу только при повторном добавлении бота в чат. В существующем тестовом чате бот по-прежнему получал только команды и реплаи на свои сообщения. Диагностирован через `outer_middleware` на `dp.update` — три из четырёх сообщений вообще не достигали бота. Исправление: удалить бота из чата и добавить снова. После этого все 4 типа сообщений начали доходить.

**Симптом 2: "Собираю мысли..." вместо генерированного ответа.**
Причина: тестовый датасет (120 предложений, 70-словный словарь) очень однородный. `looks_too_close_to_training_sample` отклоняет все 4 попытки генерации — сгенерированные кандидаты совпадают с префиксами обучающего корпуса. Для non-mentioned сообщений при неудаче генерации бот молчит (штатное поведение — не спамить чат). Для mentioned — отправляет "Собираю мысли...". В продакшн-чате с реальными разнообразными сообщениями генерация проходит успешно.

**Симптом 3: `/pivo` показывает "3 раз(а) в сутки" для "обычного пользователя".**
Причина: тестирование выполнялось с аккаунта, указанного в `OWNER_ID`. В базе данных только одна запись `pivo_daily_usage` с `used_count=3` — то есть оба отказа пришли от одного и того же (admin) аккаунта. Код работает корректно: `is_admin_or_owner=True` → `limit=3`.

### Code changes
- `app/handlers/learning.py` — в процессе расследования добавлялось и убиралось временное INFO-логирование на ключевых точках принятия решений. В финальной версии логирование возвращено к DEBUG-уровню (как было изначально).
- `main.py` — добавлялся и убирался временный `outer_middleware` для логирования входящих Update-объектов. В финальной версии удалён.
- Оба файла возвращены к исходному состоянию, `py_compile` чистый.

### Tests/checks run
- `python -m unittest discover tests -v` — 200 тестов, OK.
- `python -m py_compile main.py app/handlers/learning.py` — OK.

### Not run / limitations
- Полноценный live smoke с реальными данными не проводился (тестовый датасет искусственный).
- `seed_db.py` остаётся неоткоммиченным (вопрос не поднимался).

### Remaining work
- Merge `codex/pivo-daily-quota` в `main` — блокеров нет, code review вручную.
- P2-1 (тройное дублирование Settings/RuntimeState/runtime_config) — главный техдолг.
- Рассмотреть: добавить `.env`-параметр `LOG_LEVEL` (P2-6) — полезно для диагностики без правок кода.
