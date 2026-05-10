# Архитектура проекта

Документ описывает текущую слоистую архитектуру `PepeEdtaBot` (после
рефакторинга `refactor/structure`, фазы 1–6). Если вы трогаете код впервые —
читайте этот файл вместе с `PROJECT_AUDIT.md`.

## Слои

```
                ┌─────────────────────────────────────────────┐
                │  main.py (compose root)                     │
                │  load_settings → init DB → wire services →  │
                │  Dispatcher + middleware + routers          │
                └─────────────────────────────────────────────┘
                                  │
                  ┌───────────────┴────────────────┐
                  ▼                                ▼
        ┌───────────────────┐         ┌─────────────────────────┐
        │  filters/         │         │  middlewares/           │
        │  GroupOnly        │         │  ThrottlingMiddleware   │
        │  AdminOrOwner     │         │                         │
        └───────────────────┘         └─────────────────────────┘
                  │                                │
                  ▼                                ▼
        ┌────────────────────────────────────────────────────────┐
        │  handlers/  (aiogram.Router × 4)                       │
        │   common  /ping /help /stats                           │
        │   admin   /config /set /setprob /clear  + denied-fallback │
        │   pivo    /pivo /pivo_on /pivo_off /pivo_privacy       │
        │   learning  F.text — обучение и генерация              │
        └────────────────────────────────────────────────────────┘
                  │
                  ▼
        ┌────────────────────────────────────────────────────────┐
        │  services/  (бизнес-логика, не знают про aiogram*)     │
        │   LearningService   PivoService   MembersService       │
        └────────────────────────────────────────────────────────┘
                  │
                  ▼
        ┌────────────────────────────────────────────────────────┐
        │  repositories/  (per-domain SQL)                       │
        │   markov / messages / pivo / pivo_usage / members      │
        └────────────────────────────────────────────────────────┘
                  │
                  ▼
        ┌────────────────────────────────────────────────────────┐
        │  db.Database  (фасад: соединение + кросс-доменные      │
        │  транзакции типа save_message_and_update_model)        │
        └────────────────────────────────────────────────────────┘
                  │
                  ▼
        ┌────────────────────────────────────────────────────────┐
        │  infrastructure/migrator.py  (NNN_*.sql / .py — once)  │
        └────────────────────────────────────────────────────────┘
```

\* Исключение — `PivoService.subscribe(chat_id, user)` принимает
`aiogram.types.User` напрямую: осознанный компромисс, описан в
`PROJECT_AUDIT.md`, раздел 5.1.

## Модули

### Compose root
- `main.py` — единственная точка инициализации. `run_bot()` собирает граф
  зависимостей, `configure_dispatcher()` регистрирует middleware, ключи в
  `dp[...]` и роутеры. Никакой бизнес-логики.

### `app/`

| Подпакет | Содержимое |
|---|---|
| `handlers/` | `common.py`, `admin.py`, `pivo.py`, `learning.py` — `aiogram.Router` per file. `_helpers.py` — `reply_humanized`. |
| `services/` | `learning_service.py`, `pivo_service.py`, `members_service.py` |
| `repositories/` | `markov_repo.py`, `messages_repo.py`, `pivo_repo.py`, `pivo_usage_repo.py`, `members_repo.py` |
| `filters/` | `group_only.py` (только `GROUP`/`SUPERGROUP`), `admin_or_owner.py` (`OWNER_ID` или админ чата, fail-closed при ошибке Telegram API) |
| `middlewares/` | `throttling.py` — per-user-per-command cooldown, `pivo`=30 сек, `clear`=3600 сек |
| `infrastructure/` | `migrator.py` — пробегает `app/migrations/NNN_*` ровно один раз через `executescript` + rollback на ошибку |
| `migrations/` | `001_initial.sql` … `006_pivo_daily_usage.sql` |
| `security/` | `key_derivation.py` — HKDF-SHA256 derivation |

### Корневые legacy/utility модули

| Файл | Назначение |
|---|---|
| `db.py` | Фасад над репозиториями: соединение `aiosqlite`, кросс-доменные транзакции (`save_message_and_update_model`, `clear_chat`, `get_stats`), retention `pivo_daily_usage`. |
| `markov.py` | Variable-order генератор (3 → 2 → 1). Не трогать. |
| `pivo.py` | `PivoSecurity` (HMAC + Fernet), `PivoMember`. |
| `pivo_templates.py` | Контент сообщений `/pivo`. |
| `bot_messages.py` | Форматирование `/help`, `/stats`, `/config`. |
| `bot_policy.py` | `bot_is_mentioned`, cooldown, `should_reply`. |
| `settings.py` | `Settings` dataclass + `load_settings(env)`. |
| `runtime_state.py` | `RuntimeState` dataclass + `runtime_state_from_settings`. |
| `runtime_config.py` | Тонкая обёртка вокруг `config_registry.try_apply` для `/set`. |
| `config_registry.py` | **Единый реестр** runtime-mutable полей (`FieldSpec` × 20 + `validate_cross_fields`). |
| `text_utils.py` | `sanitize_text` — убирает `@mention`, схлопывает пробелы. |

## DI

Зависимости, которые нужны хендлерам, кладутся в `dp[...]` и автоматически
прокидываются в каждую функцию через `workflow_data` aiogram v3:

```python
dp["db"] = db
dp["generator"] = generator
dp["pivo_service"] = pivo_service
dp["learning_service"] = learning_service
dp["runtime_state"] = runtime_state
dp["settings"] = settings
dp["bot_username"] = bot_username
dp["bot_id"] = bot_id
```

Никаких самодельных DI-контейнеров. Размер проекта это не оправдывает.

> Ключ для `RuntimeState` — `runtime_state` (не `state`), чтобы не
> конфликтовать с aiogram-овским `FSMContext`, который под именем `state`
> подкладывается dispatcher'ом автоматически.

## Конфигурация

Single source of truth для runtime-mutable полей — `config_registry.py`:

```python
RUNTIME_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("reply_probability", "REPLY_PROBABILITY", "0.08",
              _float_in_range(0.0, 1.0)),
    ...
)
```

- `load_settings()` (settings.py) итерируется по реестру и читает
  значения из env через `spec.parse`.
- `runtime_state_from_settings()` копирует runtime-mutable значения
  в `RuntimeState` через тот же реестр.
- `apply_runtime_setting()` (runtime_config.py) ищет `FieldSpec` по имени
  ключа из `/set`, парсит, проверяет на shallow copy через
  `validate_cross_fields`, и только потом мутирует живой state.

Чтобы добавить новый runtime-mutable параметр, нужно:
1. добавить строчку в `RUNTIME_FIELDS`;
2. добавить поле в dataclass `Settings` и `RuntimeState` (нужно для
   статической типизации);
3. (опционально) при cross-field инварианте — расширить
   `validate_cross_fields`.

Поля, которые не должны быть mutable в runtime (`BOT_TOKEN`, `OWNER_ID`,
`DB_PATH`, `PIVO_*_SECRET`, `LOG_LEVEL`), живут только в `Settings` и
парсятся вручную в `load_settings`.

## Модель данных

| Таблица | Назначение |
|---|---|
| `messages` | `chat_id`, `author_id` (анонимизирован), `normalized_text`, `created_at`. Сырой `text` удалён миграцией 005. |
| `starts3` / `transitions3` | Основная триграммная модель (`(w1, w2) → w3`). |
| `starts` / `transitions` | Биграммный fallback (`w1 → w2`). |
| `transitions1` | Униграммный fallback. |
| `pivo_chat_members` | Opt-in подписки `/pivo`. PK `(chat_hash, user_hash)`, payload зашифрован Fernet. |
| `pivo_daily_usage` | Суточная квота `/pivo` (`chat_hash`, `user_hash`, `usage_day`, `used_count`). Retention 7 дней. |
| `chat_member_profiles` | Зарезервированный профиль участника с HKDF-derived ключом, `consent_version`, `key_version`. Сейчас инфраструктура без runtime-вызовов. |
| `schema_migrations` | Учёт применённых миграций. |

## Миграции

`app/infrastructure/migrator.py`:

- сканирует `app/migrations/NNN_<name>.{sql,py}`;
- сравнивает с записями в `schema_migrations`;
- применяет недостающие в порядке имён;
- `.sql` идёт через `conn.executescript(...)` (корректно обрабатывает
  multi-statement файлы, в т.ч. триггеры `BEGIN..END;`);
- `.py` импортирует и вызывает `await mod.apply(conn)`;
- на исключении `await conn.rollback()` и raise — запись в
  `schema_migrations` не появится для не до конца применённой миграции.

## Безопасность

- `OWNER_ID` или admin чата → `AdminOrOwner` filter (fail-closed при
  ошибке Telegram API).
- `/pivo` / `/clear` → `ThrottlingMiddleware`.
- `/pivo`-подписки: HMAC-индексированный `chat_hash`/`user_hash` +
  Fernet-шифрование payload.
- HKDF derivation ключей для `MembersService` с domain-метками
  (`members:hmac`, `members:encryption`) изолирует домены и не использует
  raw `PIVO_*_SECRET` напрямую.
- Все SQL-запросы параметризованы.
- `messages.text` больше не хранится; только `normalized_text` после
  `sanitize_text`.

## Окружения БД

- `data/markov.db` — runtime-база по умолчанию, проброшена в Docker
  volume `./data:/app/data`.
- `markov.db` в корне репозитория — локальная тестовая БД (не боевая).
- Существующие БД мигрируются автоматически при `Database.init()`.

## Тесты и CI

- `tests/` — 13 файлов, ~200 unit-тестов (`unittest`), включая smoke на
  `configure_dispatcher` и фикстуру с реальной legacy-схемой
  (`tests/fixtures/legacy_real_schema.sql`).
- CI: `.github/workflows/ci.yml` — матрица Python 3.12/3.13/3.14, шаги
  `ruff check` → `mypy app/` → `unittest discover`.

Команды для локального прогона см. в `README.md`.
