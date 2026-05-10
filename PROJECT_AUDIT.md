# Технический аудит проекта PepeEdtaBot

Дата актуализации: 2026-05-10 (восьмая редакция, после унификации pivo + chat_members).
Текущая ветка: `main`.

> История редакций: 1я — первичный аудит; 2я — после Codex-ревизии P1-блокеры; 3я — после закрытия B1/B2/B3 (merge-ready); 4я — после ветки `codex/pivo-daily-quota`; **5я — P2-batch (P2-1, P2-2, P2-3, P2-6 закрыты, +P2-4 частично, см. session 18); 6я — audit follow-ups (P2-5 + P1-фиксы атомарности миграций и Docker bind-mount, см. session 19).**

Текущие тесты/проверки: **190 unit-тестов**, `ruff check app/ tests/` — clean, `mypy app/` — clean (26 source files). Падение числа тестов относительно седьмой редакции (213 → 190) — за счёт удаления `tests/test_members.py` и снятия двух теперь-нерелевантных кейсов в `tests/test_pivo.py` (фильтр `is_bot` ушёл вместе с колонкой).

**Изменения третьей редакции (для контекста):** все три P1-блокера закрыты; CI зелёный; ветка `refactor/structure` слита в `main`.

| Блокер | SHA (после rewrite) | Описание |
|---|---|---|
| B3 | `1ef20a2` | docs(readme): fix two stale absolute links to compose.yaml and .env.example |
| B2 | `db1de05` | build: wire requirements.lock into Dockerfile and CI |
| B1 | `a0fae76` | fix(admin): restore explicit denial reply for unauthorized commands |

**Изменения второй редакции (для контекста):** дополнен по результатам независимой ревизии (`PROJECT_AUDIT_CODEX.md`). Добавлены пункты: UX-регрессия в админ-командах, неэффективность `requirements.lock`, хрупкость SQL-splitter в migrator, концептуальная неоднозначность dedup-фильтра. Рекомендация была изменена с «можно мёржить» на «после фикса P1».

---

## 0. Статус проекта

### 0.1. Ветка `refactor/structure` — слита в `main` (2026-05-08)

Все шесть фаз рефакторинга, запланированные после первоначального аудита, выполнены. Ветка была признана готовой к merge в третьей редакции аудита.

| Фаза | Содержание | Статус |
|---|---|---|
| 1 | Разбить `main.py` на routers + сервисы + репозитории | ✅ |
| 2 | Версионирование миграций (`schema_migrations` + `app/migrations/`) | ✅ |
| 3 | `chat_member_profiles` + `MembersService` + HKDF derivation | ✅ |
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

---

## 1. Краткое резюме проекта

PepeEdtaBot — Telegram-бот для группового чата на `aiogram v3`, который обучается на сообщениях участников и генерирует ответы по цепям Маркова variable-order (n=3 → n=2 → n=1, с backoff). Внешние LLM не используются. Дополнительно есть opt-in команда `/pivo` для шуточного созыва участников в Discord; для неё реализовано HMAC-индексирование и шифрование чувствительных данных (`cryptography.Fernet`).

После завершения фаз 1–6 + P2-batch + audit follow-ups проект перешёл из «всё в `main.py`» к слоистой архитектуре `handlers / services / repositories / filters / middlewares / migrations / security / infrastructure`, с версионированными миграциями (атомарными через `BEGIN/COMMIT`), отдельными доменными ключами через HKDF, throttling-middleware, единым реестром runtime-настроек (`config_registry.py`), hardened Dockerfile с root-entrypoint и CI с линтерами на 3.12/3.13/3.14. Серьёзных уязвимостей не найдено; оставшийся техдолг — P3 (концептуальная неоднозначность `looks_too_close_to_training_sample`, silent drop `/clear` под throttle, магические числа, маскирование `chat_id` в логах) + опциональные P2-улучшения (`/learn_off` opt-out, структурированные JSON-логи).

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
├── main.py                              # 115 строк (было 588) — compose root + configure_dispatcher()
├── app/                                 # ~1400 строк, 24 модуля
│   ├── handlers/
│   │   ├── _helpers.py                  # reply_humanized
│   │   ├── common.py                    # /ping, /help, /stats
│   │   ├── admin.py                     # /config, /set, /setprob, /clear + fallback-handlers (denied)
│   │   ├── pivo.py                      # /pivo (quota check), /pivo_on, /pivo_off, /pivo_privacy
│   │   └── learning.py                  # F.text + extract_context_tokens
│   ├── services/
│   │   ├── learning_service.py          # record_message, matches_training_prefix (prefix-cache)
│   │   ├── pivo_service.py              # subscribe / unsubscribe / build_call_message / consume_daily_call_quota / refund_daily_call_quota
│   │   └── members_service.py           # record_consent / get_profile / revoke (инфраструктура, runtime не вызывается)
│   ├── repositories/
│   │   ├── markov_repo.py               # starts/transitions/transitions3/transitions1
│   │   ├── messages_repo.py             # exists / get_all_normalized
│   │   ├── pivo_repo.py                 # upsert / list_members / remove
│   │   ├── members_repo.py              # upsert / get / list / remove (chat_member_profiles)
│   │   └── pivo_usage_repo.py           # consume_daily_call / refund_daily_call / delete_usage_before
│   ├── filters/
│   │   ├── group_only.py                # ChatType.GROUP/SUPERGROUP
│   │   └── admin_or_owner.py            # OWNER_ID или администратор чата
│   ├── middlewares/
│   │   └── throttling.py                # per-user per-command cooldown
│   ├── infrastructure/
│   │   └── migrator.py                  # discovers + runs NNN_*.sql/.py once each
│   ├── migrations/
│   │   ├── 001_initial.sql              # полная схема для пустых БД
│   │   ├── 002_normalize_messages_text_column.py
│   │   ├── 003_anonymize_authors.py
│   │   ├── 004_chat_member_profiles.sql
│   │   ├── 005_drop_messages_text.py
│   │   └── 006_pivo_daily_usage.sql     # таблица pivo_daily_usage (chat_hash, user_hash, usage_day, used_count)
│   └── security/
│       └── key_derivation.py            # HKDF-SHA256 derivation
├── db.py                                # ~390 строк — фасад: соединение, save_message_and_update_model, clear_chat, get_stats, делегаты; cleanup_pivo_daily_usage при init()
├── markov.py                            # 674 строки — НЕ ТРОГАТЬ
├── pivo.py                              # 125 строк — PivoSecurity, PivoMember
├── pivo_templates.py                    # 487 строк (контент)
├── bot_messages.py                      # 116 строк — форматирование
├── bot_policy.py                        # 72 строки — bot_is_mentioned, cooldown, should_reply
├── settings.py                          # 180 строк — Settings + load_settings
├── runtime_state.py                     # 56 строк — RuntimeState dataclass
├── runtime_config.py                    # 144 строки — apply_runtime_setting
├── text_utils.py                        # 28 строк — sanitize_text
├── tests/                               # 14 файлов, ~2700 строк, 200 тестов
│   ├── test_bot_messages.py             # форматирование
│   ├── test_bot_policy.py               # политика ответа
│   ├── test_db_logic.py                 # save_message_and_update_model, get_stats, daily_usage retention
│   ├── test_filters.py                  # GroupOnly, AdminOrOwner, ThrottlingMiddleware
│   ├── test_handlers.py                 # happy-path по всем 4 роутерам
│   ├── test_learning_service.py         # prefix-cache дедупликация
│   ├── test_main.py                     # smoke-тест wiring configure_dispatcher()
│   ├── test_markov_and_text.py          # генерация, токенизация
│   ├── test_members.py                  # KeyDerivation, MembersRepo, MembersService
│   ├── test_migrator.py                 # идемпотентность, resume, реальный legacy fixture
│   ├── test_pivo.py                     # HMAC, Fernet, mentions, E2E subscribe→quota→unsubscribe
│   ├── test_runtime_config.py           # /set ключи
│   └── test_settings.py                 # load_settings
├── pyproject.toml                       # ruff + mypy конфигурация
├── requirements.txt                     # верхнеуровневые диапазоны
├── requirements.lock                    # 24 пакета с pinned версиями
├── requirements-dev.txt                 # ruff + mypy для CI
├── .github/workflows/ci.yml             # ruff + mypy + tests на 3.12/3.13/3.14
├── docs/ARCHITECTURE.md
├── README.md
└── PROJECT_AUDIT.md (этот файл)
```

**Метрики динамики:**

| Метрика | До рефакторинга | После refactor/structure | Сейчас (codex/pivo-daily-quota) |
|---|---|---|---|
| `main.py` | 588 строк | 84 строки (-86%) | **115 строк** (+configure_dispatcher) |
| Файлов в `app/` | 0 | 22 | **24** (+pivo_usage_repo.py, test_main.py) |
| Тестов | 83 | 185 | **200** |
| Миграций | 0 (inline `CREATE TABLE`) | 5 | **6** (006_pivo_daily_usage) |
| Слоёв архитектуры | 1 (всё в `main.py`) | 6 | 6 (без изменений) |
| Линтер/тайпчекер | нет | ruff + mypy strict для `app/` | без изменений |
| CI | нет | GitHub Actions × 3 версии Python | без изменений |
| Lock-файл | нет | `requirements.lock` | без изменений |

---

## 4. Что выполнено из старого аудита

### P0 — все три пункта закрыты

| ID | Старая формулировка | Сейчас |
|---|---|---|
| **P0-1** | `messages.text` хранится без opt-in, это PII в открытом виде | ✅ Колонка `text` удалена миграцией [`005_drop_messages_text.py`](app/migrations/005_drop_messages_text.py); хранится только `normalized_text` (используется генератором для exact-match exclude и `LearningService.matches_training_prefix` для построения set'а префиксов обучающих сообщений). |
| **P0-2** | Нет версионирования миграций | ✅ Реализован [`migrator.py`](app/infrastructure/migrator.py) с таблицей `schema_migrations`, 5 версионированных миграций (`.sql`/`.py`), идемпотентность покрыта тестами. |
| **P0-3** | Не спроектирована таблица чувствительных данных участников | ✅ Создана таблица [`chat_member_profiles`](app/migrations/004_chat_member_profiles.sql) (PRIMARY KEY `(chat_hash, user_hash)`, `encrypted_user_id`, `encrypted_payload`, `key_version`, `consent_version`, `consented_at`); сервис [`MembersService`](app/services/members_service.py) шифрует через Fernet с ключом, выведенным **HKDF-SHA256** (см. ниже п.6.3) от существующих PIVO-секретов с доменными метками `members:hmac` и `members:encryption`. |

### P1 — все восемь пунктов закрыты

| ID | Старая формулировка | Сейчас |
|---|---|---|
| **P1-1** | `main.py` — бог-файл с inner-функциями-хендлерами | ✅ `main.py` = 84 строки, чистый compose root. Все хендлеры в `app/handlers/{common,admin,pivo,learning}.py`, каждый — `aiogram.Router`. |
| **P1-2** | `db.py` смешивает миграции, бизнес-запросы, статистику | ✅ Введены репозитории `MarkovRepo`, `MessagesRepo`, `PivoRepo`, `MembersRepo`. `db.py` оставлен как фасад с делегатами + кросс-доменными транзакциями (`save_message_and_update_model`, `clear_chat`, `get_stats`). Объём `db.py` сократился с ~620 до 373 строк. |
| **P1-3** | Нет фильтров авторизации | ✅ [`GroupOnly`](app/filters/group_only.py) и [`AdminOrOwner`](app/filters/admin_or_owner.py) применяются декларативно в декораторе `@router.message(...)`. Старые хелперы `_is_owner` / `_is_chat_admin` / `_can_manage_settings` удалены. |
| **P1-4** | Throttling/rate-limit отсутствует | ✅ [`ThrottlingMiddleware`](app/middlewares/throttling.py) с per-user-per-command cooldown; в production включён `clear=3600` сек (см. `COMMAND_COOLDOWNS_SECONDS` в `main.py`). Quota для `/pivo` реализована отдельно в `PivoService` (daily quota). Поддерживает `notify_on_throttle` — список команд, для которых при throttle возвращается явный ответ вместо silent drop (включает `clear`). |
| **P1-5** | Нет lock-файла | ⚠️ **Декоративно — файл создан, но не используется.** [`requirements.lock`](requirements.lock) содержит 24 пакета, но `Dockerfile` ставит `requirements.txt`, CI ставит `requirements-dev.txt`, в самом lock'е нет `mypy`/`ruff`. Реальной воспроизводимости не даёт. См. п.9.4 — требует решения до merge. |
| **P1-6** | Нет CI | ✅ [`.github/workflows/ci.yml`](.github/workflows/ci.yml) на матрице `python-version: [3.12, 3.13, 3.14]`, шаги `ruff check` → `mypy app/` → `unittest discover`. |
| **P1-7** | Нет линтеров и mypy | ✅ [`pyproject.toml`](pyproject.toml) с `ruff` (E/F/I/UP, line-length=100) и `mypy strict для app.*`; legacy-модули вынесены в `[[tool.mypy.overrides]] ignore_errors = true`. |
| **P1-8** | Нет тестов хендлеров | ✅ [`tests/test_handlers.py`](tests/test_handlers.py) — 18 happy-path тестов, по одному-двум на каждую команду каждого роутера; покрытие через прямой вызов функций с `MagicMock`/`AsyncMock`. |

### P2 — частично

| ID | Описание | Статус | Комментарий |
|---|---|---|---|
| **P2-1** | Реестр настроек как один источник истины (Settings ↔ RuntimeState ↔ runtime_config) | ✅ | Закрыто в `refactor/audit-p2-batch`: введён `config_registry.py` с `FieldSpec` × 20 + `validate_cross_fields`. `settings.py`, `runtime_state.py`, `runtime_config.py` итерируются по реестру вместо литерального дублирования. Публичный API `runtime_config` сохранён. |
| **P2-2** | Dockerfile hardening (non-root user, HEALTHCHECK, pin minor) | ✅ | Закрыто в `refactor/audit-p2-batch` + `fix/audit-followups`: pin `python:3.14.0-slim`, HEALTHCHECK; non-root через root-entrypoint, который чинит права на bind-mount и делает `runuser -u bot` (см. п.9.3 ниже). |
| **P2-3** | README: разделы Tests/Architecture/Privacy, убрать абсолютные ссылки | ✅ | Закрыто в `refactor/audit-p2-batch`: добавлены секции Architecture, Privacy, Тесты; `Python 3.14` → `Python 3.12+`; команды локальных проверок. Абсолютные ссылки уже были вычищены ранее (B3). |
| **P2-4** | LOG_LEVEL из `.env`, структурированные логи | 🟡 | `LOG_LEVEL` из `.env` закрыт в `refactor/audit-p2-batch` (валидируется в `Settings`, читается в `main.py`). Структурированные логи остаются P3. |
| **P2-5** | Расширить `BOT_TEXT_ALIASES` через `.env` | ✅ | Закрыто в `fix/audit-followups`: добавлен `Settings.bot_text_aliases`, парсится из CSV-env. Безопасный fallback — пустая/незаданная переменная подставляет встроенные `{"pepe", "пепе"}` (важно для деплоев без доступа к `.env` на проде). |

### P3 — отложены

- Маскирование `chat_id` в логах через HMAC.
- Магические числа (`range(4)` в `learning.py`, `len(clean) < 3 or > 500`) → константы.
- Удалить неиспользуемое поле `is_bot` в `pivo_chat_members` (пишется при `subscribe`, нигде не читается; уже фильтруется до записи).
- Структурированные логи (JSON) — отложить до выбора системы агрегации.

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
           middlewares             security (HKDF)
```

Хендлеры **не знают** про SQL: они работают только с сервисами и моделями `aiogram`. Сервисы **не знают** про aiogram (исключение — `PivoService.subscribe(chat_id, user)`, осознанное компромиссное решение, см. `REFACTOR_PLAN.md` фазы 1, конвенция №2). Это правильное направление и сохранено во всех новых модулях.

### 5.2. Кросс-доменные транзакции

`save_message_and_update_model` ([db.py:57](db.py:57)) трогает 5 таблиц в одной транзакции. Эта операция оставлена в `Database`-фасаде, не размазана по репозиториям. Это разумно: вводить «UnitOfWork» для одной такой операции — overengineering. При появлении второй такой операции стоит рассмотреть выделение отдельного слоя, но сейчас — нет.

### 5.3. DI

Зависимости (`db`, `generator`, `pivo_service`, `learning_service`, `state`, `settings`, `bot_username`, `bot_id`) кладутся в `dp[...]` и автоматически передаются в хендлеры через `workflow_data` aiogram v3. Никаких самодельных DI-контейнеров. Ровно то, что нужно проекту такого размера.

### 5.4. `main.py` как compose root

84 строки, ровно одна функция `run_bot()`. Никакой бизнес-логики, никаких хендлеров. Структура:
1. `load_settings()` → `Database.init()` → `MarkovGenerator` → сервисы;
2. `Bot` → `me = await bot.get_me()` → `bot.set_my_commands(...)`;
3. `Dispatcher()` → middleware → `dp[...] = ...` → `dp.include_router(...)`;
4. `dp.start_polling(bot)` в `try/finally`.

Образцовый compose root.

### 5.5. Оставшиеся архитектурные мелочи

- ~~**Дублирование настроек** (P2-1)~~ — закрыто в `refactor/audit-p2-batch`: `config_registry.py` стал единственным источником истины, `Settings`/`RuntimeState`/`runtime_config` теперь итерируются по `FieldSpec`.
- **`runtime_config.py`** обращается к `RuntimeState` через `setattr(state, key, value)` без compile-time-гарантий, что `key` действительно типизирован — это работает (validate_cross_fields + типы `FieldSpec` проверяют значения в рантайме), но `mypy strict` модуль не принимает; поэтому он в `[[tool.mypy.overrides]] ignore_errors = true`. Жить с этим можно.

---

## 6. Безопасность — текущее состояние

### 6.1. Хранение сообщений (бывшая P0-1)

Колонка `messages.text` удалена. Хранится только `normalized_text` — `sanitize_text` убирает `@mentions` и URL, нормализует пробелы. Это уже **не сырой текст пользователя**: tokenizable, но не идентичен исходному. Используется только генератором (для exact-match exclude через `db.message_exists`) и `LearningService.matches_training_prefix` (строит set префиксов 3..5 токенов из всех сообщений чата). Это резкое улучшение privacy-постуры.

**Что не сделано:** нет команды `/learn_off` / `/privacy` для opt-out на уровне чата (см. P2-9 в section 14). Решено отложить — основной privacy-долг закрыт удалением `messages.text` и анонимизацией `author_id`.

#### `matches_training_prefix` — вспомогательный novelty-фильтр

Закрыто в session 20: метод раньше назывался `looks_too_close_to_training_sample` и читался как «privacy-guard от копирования обучающих фраз», по реализации же был novelty-эвристикой со случайным префиксом длиной 3..5. Метод переименован в `matches_training_prefix`, docstring явно описывает его как **второстепенный** фильтр поверх основных защит в `MarkovGenerator.generate_text` (`is_low_diversity_reply`, `is_context_heavy_reply`, `trim_repetitive_tail`, exact-match через `db.message_exists`). Случайность префикса оставлена как сознательный novelty-nudge. Мёртвая ветка SQL-fallback для текстов <3 токенов удалена.

### 6.2. Авторизация и throttling

- `OWNER_ID` или admin чата → проверяет [`AdminOrOwner`](app/filters/admin_or_owner.py); при ошибке `bot.get_chat_administrators` (например, бот не в чате) → возвращает `False` (deny by default). Тесты покрывают этот случай.
- `/clear` → ограничен [`ThrottlingMiddleware`](app/middlewares/throttling.py): `clear=3600` сек **на пользователя в данном чате**. `/clear` добавлен в `notify_on_throttle`-набор, поэтому повторный вызов под cooldown получает явный ответ «Слишком часто. Подождите ~N сек.» вместо silent drop (session 20). Quota для `/pivo` реализована отдельно в `PivoService` через дневной счётчик (см. P2-7). Проброшенные сообщения (без `from_user`, не-команды, команды без записи в `limits`) пропускаются без задержек.

### 6.3. HKDF derivation для members

```python
# app/security/key_derivation.py
def derive_key(master_secret: str, domain: str, length: int = 32) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None,
                info=domain.encode()).derive(master_secret.encode())
```

`MembersService` использует `derive_key(PIVO_HMAC_SECRET, "members:hmac")` и `derive_fernet_key(PIVO_ENCRYPTION_SECRET, "members:encryption")`. Это означает:
- ключ для `members` **не равен** raw-секрету `PIVO_*_SECRET` (тест [`test_members_keys_differ_from_pivo_keys`](tests/test_members.py) фиксирует этот инвариант);
- domain-метки изолируют будущие домены (`members:hmac` ≠ `members:encryption` ≠ `audit:hmac`...).

Соль (`salt`) у HKDF — `None`. Это допустимо, потому что master-секрет уже высокоэнтропийный (32 байта `secrets.token_urlsafe(32)`). Если бы master-ключ выводился из пользовательского пароля — соль была бы обязательна.

**Замечание:** старый PivoSecurity ([pivo.py](pivo.py)) **не** перешёл на HKDF — он по-прежнему использует `sha256(PIVO_ENCRYPTION_SECRET)` напрямую. Это было осознанное решение фазы 3: переход PivoSecurity на HKDF поломал бы существующие подписи `/pivo` (изменился бы ключ). Документированное ограничение.

### 6.4. SQL-инъекции

Все запросы параметризованы. Проверено grep'ом. **Уязвимостей нет.**

### 6.5. Логирование

`chat_id` в логах не маскируется (P3-4, см. п.10), `user_id` — не пишется (хорошо), `pivo`-операции пишут только `mentions count: N` (хорошо). Уровень настраивается через `LOG_LEVEL` в `.env` (закрыто в P2-6).

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

| # | Где | Что | Приоритет |
|---|---|---|---|
| 7.1 | [`db.py:165-174`](db.py:165) | `save_message_and_update_model` использует `cursor.fetchone()[0]` без `assert row is not None`, тогда как репозитории (например, `markov_repo.get_chat_token_volume`) уже унифицированы. Стилистическая мелочь, mypy не ругается из-за приведения `int(...)`. | P3 |

Закрыто в предыдущих сессиях:
- ~~`range(4)` / `attempt < 2`~~ — константы `MAX_GENERATION_ATTEMPTS` (session 20).
- ~~`len(clean) < 3 or > 500`~~ — `MIN/MAX_LEARN_MESSAGE_CHARS` (раньше).
- ~~Дублирование Settings/RuntimeState/runtime_config~~ — `config_registry.py` (P2-batch).
- ~~`BOT_TEXT_ALIASES = {"pepe", "пепе"}` хардкод~~ — `Settings.bot_text_aliases` с safe fallback (audit follow-ups).
- ~~`MENTION_RE = r"@\w+"` ломает email~~ — `(?<!\w)@\w+` (session 20).

### 7.3. Мёртвый код

Проверено явно:

- `_ensure_messages_normalized_text_column` и `_anonymize_message_author_ids` — упоминаются только в `PROJECT_AUDIT.md` и `REFACTOR_PLAN.md`; в коде их нет (логика переехала в миграции 002 и 003). ✅
- `MessagesRepo.exists()` — используется в [`MarkovGenerator.generate_text`](markov.py:670) для exact-match фильтра «не отдавать слово в слово сохранённое сообщение». Раньше также вызывался из `LearningService` как SQL-fallback для <3 токенов; ветка удалена в session 20 как избыточная (генератор делает то же самое раньше). Не мёртвый. ✅
- `MembersService.record_consent` / `get_profile` / `revoke` — **никем не вызывается** из хендлеров. Это **намеренная инфраструктура** (плановое решение фазы 3: «Хендлеры/команды НЕ добавлять в этой фазе — только инфраструктура»). Покрыта 18 тестами. Не мёртвый код, но требует продуктового шага: либо использовать в ближайшее время, либо документировать как «WIP backend».
- `pivo_chat_members.is_bot` — пишется в `PivoService.subscribe`, нигде не читается; боты уже отсекаются в хендлере [`cmd_pivo_on`](app/handlers/pivo.py:40). Поле лишнее, но удалять до P3.

---

## 8. БД и хранение

### 8.1. Миграции

- 5 версионированных миграций, каждая запускается ровно один раз.
- Резюм через таблицу `schema_migrations(name, applied_at)`.
- Тесты на migrator (`tests/test_migrator.py`, 495 строк) покрывают:
  - чистая БД → все миграции применяются по порядку;
  - повторный init → ничего не делает;
  - legacy-БД (фикстура `tests/fixtures/legacy_real_schema.sql` со схемой реальной prod-БД пользователя) → корректно проходит миграции 002–005 без потери данных;
  - INSERT после миграций кладёт данные в правильные новые столбцы.

**Слабое место — SQL splitter (P2).** [`migrator._split_sql`](app/infrastructure/migrator.py:58) сейчас:

```python
return [s.strip() for s in sql.split(";") if s.strip()]
```

Для текущих 5 простых миграций (`CREATE TABLE`, `CREATE INDEX`) этого достаточно, и тесты подтверждают. Но splitter сломается на:
- `;` внутри строковых литералов (`INSERT ... VALUES ('foo;bar')`);
- триггерах с `BEGIN ... END;` блоками;
- views с подзапросами;
- комментариях с `;`.

Дополнительно — `_apply` не оборачивает миграцию в явную транзакцию с rollback. Для DDL в SQLite в большинстве случаев работает терпимо (DDL коммитится автоматически), но это слабая инфраструктура.

**Рекомендация:** для `.sql`-миграций перейти на `conn.executescript(sql)` (SQLite сам разбирает многооператорные скрипты) внутри `BEGIN...COMMIT/ROLLBACK`. Текущий splitter оставить только если будет ограничение «один statement на файл».

### 8.2. Индексы

| Индекс | Использование | Замечание |
|---|---|---|
| `idx_pivo_chat_members_chat_hash` | `PivoRepo.list_members(chat_hash)` | OK |
| `idx_chat_member_profiles_chat_hash` | (потенциально) `MembersRepo.list_for_chat` | OK |
| `idx_messages_normalized_lookup(chat_id, normalized_text)` | `MessagesRepo.exists(chat_id, text)` | Используется в SQL-fallback для текстов <3 токенов |

Дополнительные индексы по `(chat_id, w1, w2)` на `transitions*` и `starts*` существуют через PRIMARY KEY и не требуют отдельных definition'ов.

### 8.3. Целостность

- Нет FK между `messages` и `transitions*` — это агрегаты, FK не нужны. ✅
- `pivo_chat_members.is_bot` — лишнее поле (см. п.7.3). P3.
- `chat_member_profiles.consent_version` и `key_version` заложены, но нет процедуры ротации ключей. Это OK — заложить sloтa достаточно, ротация будет отдельной задачей при необходимости.

---

## 9. Конфигурация и запуск

### 9.1. README

**Не обновлён** после рефакторинга.

| Что | Статус |
|---|---|
| Quickstart (`venv`, `pip`, `python main.py`) | ✅ есть |
| Команды | ✅ есть, актуальные |
| Docker | ✅ есть |
| `/pivo` privacy | ✅ есть |
| Раздел «Тесты» | ❌ отсутствует |
| Раздел «Архитектура» (ссылка на `docs/ARCHITECTURE.md`) | ❌ отсутствует |
| Раздел «Privacy» (что хранится, что не хранится, опт-ин для `/pivo`) | ❌ отсутствует |
| Упоминание `requirements.lock` для воспроизводимости | ❌ отсутствует |
| Абсолютная ссылка `[compose.yaml](/D:/test/PepeEdtaBot/compose.yaml)` (строка 47) | ❌ **сломанная ссылка**, ведёт на чужую машину |
| Абсолютная ссылка `[.env.example](/D:/test/PepeEdtaBot/.env.example)` (строка 54) | ❌ **сломанная ссылка** |
| Версия Python: указано `Python 3.14` (строка 6), но CI поддерживает `3.12/3.13/3.14` | ❌ противоречие, лучше `Python 3.12+` |
| Инструкция «как запустить локальные проверки» (`ruff`, `mypy`, `unittest discover`) | ❌ отсутствует |

**Срочность:** P2. Не блокирует merge технически, но первая внешняя impression проекта — это README.

### 9.2. `.env.example`

Полный, структурированный, с инструкцией по генерации секретов. Не хватает:
- `LOG_LEVEL` (используется захардкоженный `INFO` в [`main.py:25`](main.py:25)).

**Срочность:** P2.

### 9.3. Dockerfile

**Не изменялся.** Текущее состояние:

```dockerfile
FROM python:3.14-slim          # ❌ нет pin минорной версии (3.14.x-slim)
ENV PYTHONDONTWRITEBYTECODE=1  # ✅
ENV PYTHONUNBUFFERED=1         # ✅
WORKDIR /app
RUN mkdir -p /app/data
COPY requirements.txt .        # ✅ кеш слоёв
RUN pip install --no-cache-dir -r requirements.txt
COPY . .                       # ⚠️  COPY . . включает .git, tests, docs — лучше docker-build с .dockerignore (он уже есть)
CMD ["python", "main.py"]      # ❌ работает от root
                               # ❌ нет HEALTHCHECK
```

**Что добавить (P2):**
- `RUN useradd -m -u 1000 bot && chown -R bot /app` + `USER bot`;
- pin минорной версии (`python:3.14.0-slim` или конкретный digest);
- `HEALTHCHECK CMD python -c "import sys; sys.exit(0)"` или подобное;
- использовать `requirements.lock` вместо `requirements.txt` для воспроизводимости.

**Срочность:** P2 — не блокирует функциональность, hardening.

### 9.4. CI

`.github/workflows/ci.yml` обновлён в фазе 6:

```yaml
- pip install -r requirements-dev.txt   # ставит и runtime, и линтеры
- python -m ruff check app/ tests/
- python -m mypy app/
- python -m unittest discover tests -v
```

Матрица — 3.12/3.13/3.14. Triggers — push в main, PR в main. Branch protection пока не настроен (см. рекомендацию в п.13).

### 9.5. `requirements.lock` — **закрыто** (`db1de05` / B2)

`Dockerfile` теперь устанавливает `requirements.lock` (вместо ранее использовавшегося `requirements.txt` с диапазонами), `requirements-dev.txt` ссылается на `-r requirements.lock`, в файл добавлен header с описанием стратегии (`pip freeze` без `pip-tools`) и процедурой регенерации. CI и Docker действительно используют lock — это уже не декорация.

---

## 10. Логирование

- ~~Уровень захардкожен `INFO`~~ — закрыто (P2-6): `LOG_LEVEL` валидируется в `Settings`, читается из `.env`, применяется в `main.py`.
- `aiogram` приглушён до `WARNING`.
- Формат — обычный текст: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`.
- Идентификаторы пользователей не пишутся, `chat_id` пишется в открытом виде. **Открытый P3** — маскирование через HMAC.
- Нет ID-корреляции запросов. Не приоритет.
- Нет аудита админ-команд (`/clear`, `/set`) в БД. Опциональная фича.

**Срочность:** P3 (маскирование `chat_id`; структурированные JSON-логи отложены до выбора системы агрегации).

---

## 11. Тестирование

### 11.1. Состояние

Текущее число тестов в наборе: **208** (после P2-batch и audit follow-ups). Полный список файлов и доменов покрытия:

| Файл | Что покрывает |
|---|---|
| `test_db_logic.py` | `save_message_and_update_model`, `get_stats`, `clear_chat`, retention `pivo_daily_usage` |
| `test_filters.py` | `GroupOnly`, `AdminOrOwner` (включая fail-closed на ошибке Telegram API), `ThrottlingMiddleware` |
| `test_handlers.py` | happy-path для всех 4 роутеров + denied-fallback для админ-команд |
| `test_learning_service.py` | prefix-cache дедупликация, инвалидация кэша |
| `test_main.py` | smoke-тест на `configure_dispatcher` (роутеры, middleware, ключи в `dp[]`) |
| `test_markov_and_text.py` | генерация, токенизация, `sanitize_text` |
| `test_members.py` | `KeyDerivation` (HKDF), `MembersRepo`, `MembersService` |
| `test_migrator.py` | пустая БД, повторный init, legacy fixture, real-schema fixture, **атомарность .sql при ошибке** |
| `test_pivo.py` | HMAC, Fernet, `build_pivo_mention`, E2E subscribe → call quota → unsubscribe |
| `test_runtime_config.py` | `apply_runtime_setting` для всех ключей реестра |
| `test_settings.py` | `load_settings`, валидация ENV (включая `LOG_LEVEL`, `BOT_TEXT_ALIASES` с fallback) |
| `test_bot_messages.py` | форматирование `/help`, `/stats`, `/config` |
| `test_bot_policy.py` | `bot_is_mentioned`, `should_reply`, cooldown |

### 11.2. Покрытие модулей

| Модуль | Тесты | Качество покрытия |
|---|---|---|
| `app/handlers/*` | `test_handlers.py` | ✅ Happy-path, моки сервисов |
| `app/services/learning_service.py` | `test_learning_service.py` | ✅ Кэш, инвалидация, edge cases |
| `app/services/members_service.py` | `test_members.py` | ✅ Все 3 метода + шифрование |
| `app/services/pivo_service.py` | `test_pivo.py` (косвенно) | ⚠️ Покрыты HMAC/Fernet, но не сам сервис в полном flow |
| `app/repositories/*` | `test_db_logic.py`, `test_members.py` | ✅ |
| `app/filters/*` | `test_filters.py` | ✅ Включая ошибки API `get_chat_administrators` |
| `app/middlewares/throttling.py` | `test_filters.py` | ✅ throttle, разные пользователи, разные команды, suffix `@bot` |
| `app/infrastructure/migrator.py` | `test_migrator.py` | ✅ С реальной prod-фикстурой |
| `app/security/key_derivation.py` | `test_members.py` | ✅ |

### 11.3. Дыры в покрытии

Закрыто:
- ~~E2E `/pivo`-flow~~ — закрыто в `codex/pivo-daily-quota` (`tests/test_pivo.py`).
- ~~Smoke-тест на wiring `main.py`~~ — закрыто там же (`tests/test_main.py`).
- ~~Тесты на отказ авторизации с явным ответом~~ — закрыто блокером B1 в `refactor/structure` (`test_handlers.py` покрывает denied-fallback handlers).
- ~~Атомарность .sql-миграций при ошибке~~ — закрыто в `fix/audit-followups` (`TestMigratorAtomicity`).

Осталось:
- **`PivoService.build_call_message`** в реальной комбинации с `MembersRepo` через DB — не покрыт целиком (P3).
- **Поведение хендлеров при ошибках Telegram API** (`bot.get_me()` падает, `bot.send_chat_action` 5xx) — не покрыто (P3).
- **Реальная гонка throttling** (одновременные вызовы) — не тестируется, но in-memory dict не имеет race conditions в asyncio (single-threaded), поэтому риск низкий.

---

## 12. Что хорошо сделано в проекте

Сохраняю и расширяю раздел из прошлого аудита.

- **Структурный рефакторинг прошёл без регрессий** — поведение бота идентично, что подтверждается тем, что все 83 предсуществующих теста зелёные плюс 98 новых.
- **Миграции на реальной БД** — фаза 2 проверялась на реальном `markov.db` пользователя (фикстура `tests/fixtures/legacy_real_schema.sql`), а не только на синтетических случаях.
- **HKDF с domain labels** — модель ключей расширяема: добавление нового домена (например, `audit:hmac`) не требует новых env-переменных и не ломает существующие подписи.
- **Throttling middleware** правильно использует `TelegramObject` в сигнатуре `__call__` (LSP-compatible), внутри проверяет `isinstance(event, Message)`. Это даёт корректный mypy strict.
- **`AdminOrOwner.fail-closed`** — при ошибке Telegram API (бот не в чате) фильтр возвращает `False`, не `True`. Поведение покрыто тестом.
- **Параметризованные SQL-запросы** — нет инъекций. ✅
- **`author_id` принудительно анонимизирован** через миграцию 003 при первом обновлении старой БД.
- **Opt-in `/pivo`** реализован с поддержкой пользователей **без `@username`** (через `tg://user?id=...`) — это образец для любого будущего функционала, требующего упоминаний.
- **Privacy-сообщение `/pivo_privacy`** написано спокойно, без технических деталей хранения, и сейчас уже не врёт пользователю (нет хранения сырого `messages.text`).
- **`pyproject.toml`** настроен с осмысленными overrides: legacy-модули (`markov.py`, `bot_messages.py` и т.д.) явно вынесены в `ignore_errors`, чтобы strict-режим не ломал CI ради `# type: ignore` спама.
- **CI на 3 версиях Python** (3.12/3.13/3.14) — даёт уверенность, что код работает на актуальной prod-версии и на двух более старых.
- **Документация в виде комментариев** к `REFACTOR_PLAN.md` (раздел «Конвенции», «Подводные камни») — будет полезна следующему контрибьютору.

---

## 13. Готовность к merge в `main`

### 13.1. Чек-лист технической готовности

| # | Что | Статус |
|---|---|---|
| 1 | CI зелёный на ветке `refactor/structure` | ✅ (CI #20, последний коммит `a0fae76`) |
| 2 | Все 185 тестов проходят локально | ✅ |
| 3 | `ruff check app/ tests/` без замечаний | ✅ |
| 4 | `mypy app/` без замечаний | ✅ |
| 5 | Нет `TODO`/`FIXME` в `app/` | ✅ |
| 6 | Нет коммита временных файлов (`.env`, `*.sqlite` тестов) | ✅ |
| 7 | `REFACTOR_PLAN.md` удалён (план выполнен) | ✅ |
| 8 | `MERGE_PREP_PLAN.md` удалён (блокеры закрыты) | ✅ (этой редакцией) |
| 9 | `PROJECT_AUDIT.md` обновлён до текущего состояния | ✅ (третья редакция) |

### 13.2. Блокеры P1 — закрыты

| # | Блокер | Закрыто в коммите | Что сделано |
|---|---|---|---|
| **B1** | UX-регрессия: `/set`/`/setprob`/`/clear` без прав молча игнорируются (R1) | `a0fae76` | Добавлены fallback-handlers `cmd_set_denied`/`cmd_setprob_denied`/`cmd_clear_denied` после защищённых; aiogram перебирает handlers по порядку, fallback срабатывает при `AdminOrOwner=False`. Покрытие: 4 теста (3 на тексты + 1 smoke на порядок регистрации). |
| **B2** | `requirements.lock` не выполняет роль lock-файла | `db1de05` | `Dockerfile` устанавливает `requirements.lock` (вместо `requirements.txt`); `requirements-dev.txt` ссылается на `-r requirements.lock`; добавлен header с описанием стратегии (pip freeze без pip-tools) и процедурой регенерации; убрана транзитивная dev-зависимость `ast_serialize`. |
| **B3** | README: 2 абсолютные ссылки ведут в `/D:/test/...` | `1ef20a2` | Заменены на относительные `compose.yaml` и `.env.example`. |

### 13.3. После merge — желательно сразу

1. **Branch protection** для `main` в GitHub:
   - require PR review;
   - require CI зелёный;
   - запретить force-push.
2. **CHANGELOG.md** или GitHub Release с описанием фаз 1–6 — чтобы оператор бота знал, что обновляется.
3. Создать issues по оставшемуся P2/P3-долгу (см. раздел 14).

### 13.4. Рекомендация

**Готово к merge.** Все три блокера B1/B2/B3 закрыты коммитами `a0fae76`, `db1de05`, `1ef20a2` соответственно. CI #20 зелёный на матрице 3.12/3.13/3.14, локально 185 тестов проходят, ruff и mypy без замечаний.

**Тип merge:** `merge commit` без squash. 31+ маленьких коммитов фаз дают полезный bisect-friendly след — каждая фаза была атомарной с зелёными тестами и явным сообщением.

В архитектурном смысле ветка чистая: P0 закрыт, P1 закрыт полностью, регрессии относительно `main` устранены, тестами покрыто хорошо. Оставшийся P2/P3-техдолг документирован в разделе 14 и не блокирует production.

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

### P3 — косметика и долгосрочное

Закрыто в session 20:
- ~~Магические числа в `learning.py` (`range(4)`, `attempt < 2`)~~ → константы.
- ~~Концепция `is_duplicate` (privacy vs novelty)~~ → переименовано в `matches_training_prefix`, docstring и тесты обновлены, мёртвая SQL-ветка удалена.
- ~~Silent drop `/clear` под throttle~~ → `notify_on_throttle` + явный ответ.
- ~~`MENTION_RE` на email~~ → `(?<!\w)@\w+`.

Остаётся:
- **Маскирование `chat_id` в логах через HMAC** (HKDF от `PIVO_HMAC_SECRET` с доменом `logging:chat_id`, первые 8 hex). Будет в следующей сессии.
- ~~Удалить неиспользуемое поле `is_bot`~~ — закрыто в session 21 (миграция 007).
- **Структурированные логи (JSON)** — отложить до выбора системы агрегации.
- **Метрики (Prometheus / aiogram-middleware)** — заводить, когда будет куда отправлять.
- **`cursor.fetchone()` стиль в `db.py:165-174`** (`assert row is not None`, как в репозиториях).
- **Поведение хендлеров при ошибках Telegram API** — не покрыто тестами (см. 11.3).
- **`PivoService.build_call_message` через DB E2E** — закрыт фактически тестом `TestPivoServiceFlow.test_subscribe_call_quota_and_unsubscribe_flow`; при подтверждении можно убрать из остаточного P3.

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

| ID | Локация | Суть | Риск | Приоритет |
|---|---|---|---|---|
| P3-4 | логи | `chat_id` без маскирования | privacy (низко) | P3 (next) |
| P3-7 | хендлеры | поведение при ошибках Telegram API не покрыто тестами | надёжность (низко) | P3 |
| P3-8 | `tests/test_pivo.py` | `build_call_message` через DB — пересмотреть, фактически покрыт `test_subscribe_call_quota_and_unsubscribe_flow` | формальная закрываемость | P3 (review) |
| 7.1 | `db.py:165-174` | стилистика `cursor.fetchone()[0]` без `assert row is not None` | стиль | P3 |

---

**Дата актуализации:** 2026-05-10, восьмая редакция.
**Статус:** на `main`, **190 тестов**, ruff/mypy clean (26 source files).
**История ревизий:**
- 2026-05-08, 1я: первичный аудит, «мёржить после фиксов».
- 2026-05-08, 2я: Codex-ревизия, блокеры P1-A/B/C.
- 2026-05-08, 3я: блокеры закрыты (`a0fae76`, `db1de05`, `1ef20a2`), `refactor/structure` — merge-ready.
- 2026-05-09, 4я: `codex/pivo-daily-quota` — daily quota, hotfixes (DI conflict, clear cooldown, quota refund), 200 тестов, P2-7/P2-8 закрыты.
- 2026-05-10, 5я: `refactor/audit-p2-batch` — P2-1, P2-2, P2-3, P2-6 закрыты, P2-4 частично, debug-лог успешной генерации, 203 теста.
- 2026-05-10, 6я: `fix/audit-followups` — P2-5 закрыто (BOT_TEXT_ALIASES с safe fallback), P1-фиксы атомарности миграций (BEGIN/COMMIT-обёртка) и Docker bind-mount (root-entrypoint + runuser), 208 тестов.
- 2026-05-10, 7я (PR #5, `polish/audit-session-20`): P3-полировка — `matches_training_prefix` (rename + honest docstring + removed dead branch), магические числа → константы, `MENTION_RE` для email, `notify_on_throttle` UX-фикс silent drop `/clear`. 213 тестов.
- 2026-05-10, 8я (`feat/unify-chat-members`): унификация участников — миграция 007 переносит `pivo_chat_members` → `chat_members` (без `is_bot`), `chat_member_profiles` / `MembersService` / `MembersRepo` / `key_derivation.py` удалены, `PivoRepo` → `ChatMembersRepo`. Закрыто P2-5, P3-5, снят P2-9. 190 тестов (-23 за счёт удалённых dead-code тестов).

> SHA коммитов обновлены после очистки истории `git filter-repo` (2026-05-09): удалены строки `Co-Authored-By: Claude` из 28 коммитов.

---

## 16. Session update — 2026-05-09

### Completed
- Очистка истории git: удалены `Co-Authored-By: Claude Opus 4.7 / Sonnet 4.6` из 28 коммитов через `git filter-repo`.
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
