# Технический аудит проекта PepeEdtaBot

Дата аудита: 2026-05-08 (третья редакция, после закрытия P1-блокеров)
Ветка: `refactor/structure`, последний коммит `037b401 fix(admin): restore explicit denial reply for unauthorized commands`
Предыдущая редакция аудита: 2026-05-08, вторая редакция (после Codex-ревизии).

**Изменения третьей редакции:** все три P1-блокера, выявленные во второй редакции, закрыты; CI зелёный; ветка готова к merge.

| Блокер | SHA | Описание |
|---|---|---|
| B3 | `d342c3e` | docs(readme): fix two stale absolute links to compose.yaml and .env.example |
| B2 | `9a052af` | build: wire requirements.lock into Dockerfile and CI |
| B1 | `037b401` | fix(admin): restore explicit denial reply for unauthorized commands |

**Изменения второй редакции (для контекста):** дополнен по результатам независимой ревизии (`PROJECT_AUDIT_CODEX.md`). Добавлены пункты: UX-регрессия в админ-командах, неэффективность `requirements.lock`, хрупкость SQL-splitter в migrator, концептуальная неоднозначность dedup-фильтра. Рекомендация была изменена с «можно мёржить» на «после фикса P1».

---

## 0. Статус ветки `refactor/structure`

Все шесть фаз рефакторинга, запланированные после первоначального аудита, выполнены. Цель ревизии — независимым взглядом подтвердить, что ветка готова к слиянию в `main`, и зафиксировать оставшийся технический долг с приоритетами.

| Фаза | Содержание | Статус |
|---|---|---|
| 1 | Разбить `main.py` на routers + сервисы + репозитории | ✅ |
| 2 | Версионирование миграций (`schema_migrations` + `app/migrations/`) | ✅ |
| 3 | `chat_member_profiles` + `MembersService` + HKDF derivation | ✅ |
| 4 | Privacy: удаление `messages.text`, дедупликация генерации | ✅ |
| 5 | Фильтры (`GroupOnly`, `AdminOrOwner`) + throttling middleware | ✅ |
| 6 | `ruff`/`mypy strict` для `app/`, `requirements.lock`, CI с линтерами, тесты хендлеров | ✅ |

Все 181 тест зелёные локально и в CI (3.12/3.13/3.14). `ruff check` и `mypy app/` проходят без замечаний.

---

## 1. Краткое резюме проекта

PepeEdtaBot — Telegram-бот для группового чата на `aiogram v3`, который обучается на сообщениях участников и генерирует ответы по цепям Маркова variable-order (n=3 → n=2 → n=1, с backoff). Внешние LLM не используются. Дополнительно есть opt-in команда `/pivo` для шуточного созыва участников в Discord; для неё реализовано HMAC-индексирование и шифрование чувствительных данных (`cryptography.Fernet`).

После завершения фаз 1–6 проект перешёл из «всё в `main.py`» к слоистой архитектуре `handlers / services / repositories / filters / middlewares / migrations`, с версионированными миграциями, отдельными доменными ключами через HKDF, throttling-middleware и CI с линтерами. Серьёзных уязвимостей не найдено; оставшийся техдолг — преимущественно P2/P3 (Dockerfile-hardening, README, реестр настроек).

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
├── main.py                              # 84 строки (было 588) — compose root
├── app/                                 # ~1300 строк, 17 модулей
│   ├── handlers/
│   │   ├── _helpers.py                  # reply_humanized
│   │   ├── common.py                    # /ping, /help, /stats
│   │   ├── admin.py                     # /config, /set, /setprob, /clear
│   │   ├── pivo.py                      # /pivo, /pivo_on, /pivo_off, /pivo_privacy
│   │   └── learning.py                  # F.text + extract_context_tokens
│   ├── services/
│   │   ├── learning_service.py          # record_message, is_duplicate (prefix-cache)
│   │   ├── pivo_service.py              # subscribe / unsubscribe / build_call_message
│   │   └── members_service.py           # record_consent / get_profile / revoke (инфраструктура)
│   ├── repositories/
│   │   ├── markov_repo.py               # starts/transitions/transitions3/transitions1
│   │   ├── messages_repo.py             # exists / get_all_normalized
│   │   ├── pivo_repo.py                 # upsert / list_members / remove
│   │   └── members_repo.py              # upsert / get / list / remove (chat_member_profiles)
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
│   │   └── 005_drop_messages_text.py
│   └── security/
│       └── key_derivation.py            # HKDF-SHA256 derivation
├── db.py                                # 373 строки — фасад: соединение, save_message_and_update_model, clear_chat, get_stats, делегаты
├── markov.py                            # 674 строки — НЕ ТРОГАТЬ
├── pivo.py                              # 125 строк — PivoSecurity, PivoMember
├── pivo_templates.py                    # 487 строк (контент)
├── bot_messages.py                      # 116 строк — форматирование
├── bot_policy.py                        # 72 строки — bot_is_mentioned, cooldown, should_reply
├── settings.py                          # 180 строк — Settings + load_settings
├── runtime_state.py                     # 56 строк — RuntimeState dataclass
├── runtime_config.py                    # 144 строки — apply_runtime_setting
├── text_utils.py                        # 28 строк — sanitize_text
├── tests/                               # 12 файлов, 2518 строк, 181 тест
│   ├── test_bot_messages.py             # форматирование
│   ├── test_bot_policy.py               # политика ответа
│   ├── test_db_logic.py                 # save_message_and_update_model, get_stats
│   ├── test_filters.py                  # GroupOnly, AdminOrOwner, ThrottlingMiddleware
│   ├── test_handlers.py                 # happy-path по всем 4 роутерам
│   ├── test_learning_service.py         # prefix-cache дедупликация
│   ├── test_markov_and_text.py          # генерация, токенизация
│   ├── test_members.py                  # KeyDerivation, MembersRepo, MembersService
│   ├── test_migrator.py                 # идемпотентность, resume, реальный legacy fixture
│   ├── test_pivo.py                     # HMAC, Fernet, mentions
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

| Метрика | До рефакторинга | Сейчас |
|---|---|---|
| `main.py` | 588 строк | **84 строки** (-86%) |
| Файлов в `app/` | 0 | 22 |
| Тестов | 83 | **181** (+118%) |
| Слоёв архитектуры | 1 (всё в `main.py`) | 6 (handlers/services/repos/filters/middlewares/infrastructure) |
| Миграций | 0 (inline `CREATE TABLE`) | 5 версионированных |
| Линтер/тайпчекер | нет | ruff + mypy strict для `app/` |
| CI | нет | GitHub Actions × 3 версии Python |
| Lock-файл | нет | `requirements.lock` |

---

## 4. Что выполнено из старого аудита

### P0 — все три пункта закрыты

| ID | Старая формулировка | Сейчас |
|---|---|---|
| **P0-1** | `messages.text` хранится без opt-in, это PII в открытом виде | ✅ Колонка `text` удалена миграцией [`005_drop_messages_text.py`](app/migrations/005_drop_messages_text.py); хранится только `normalized_text` (нужен для `is_duplicate`/SQL fallback). |
| **P0-2** | Нет версионирования миграций | ✅ Реализован [`migrator.py`](app/infrastructure/migrator.py) с таблицей `schema_migrations`, 5 версионированных миграций (`.sql`/`.py`), идемпотентность покрыта тестами. |
| **P0-3** | Не спроектирована таблица чувствительных данных участников | ✅ Создана таблица [`chat_member_profiles`](app/migrations/004_chat_member_profiles.sql) (PRIMARY KEY `(chat_hash, user_hash)`, `encrypted_user_id`, `encrypted_payload`, `key_version`, `consent_version`, `consented_at`); сервис [`MembersService`](app/services/members_service.py) шифрует через Fernet с ключом, выведенным **HKDF-SHA256** (см. ниже п.6.3) от существующих PIVO-секретов с доменными метками `members:hmac` и `members:encryption`. |

### P1 — все восемь пунктов закрыты

| ID | Старая формулировка | Сейчас |
|---|---|---|
| **P1-1** | `main.py` — бог-файл с inner-функциями-хендлерами | ✅ `main.py` = 84 строки, чистый compose root. Все хендлеры в `app/handlers/{common,admin,pivo,learning}.py`, каждый — `aiogram.Router`. |
| **P1-2** | `db.py` смешивает миграции, бизнес-запросы, статистику | ✅ Введены репозитории `MarkovRepo`, `MessagesRepo`, `PivoRepo`, `MembersRepo`. `db.py` оставлен как фасад с делегатами + кросс-доменными транзакциями (`save_message_and_update_model`, `clear_chat`, `get_stats`). Объём `db.py` сократился с ~620 до 373 строк. |
| **P1-3** | Нет фильтров авторизации | ✅ [`GroupOnly`](app/filters/group_only.py) и [`AdminOrOwner`](app/filters/admin_or_owner.py) применяются декларативно в декораторе `@router.message(...)`. Старые хелперы `_is_owner` / `_is_chat_admin` / `_can_manage_settings` удалены. |
| **P1-4** | Throttling/rate-limit отсутствует | ✅ [`ThrottlingMiddleware`](app/middlewares/throttling.py) с per-user-per-command cooldown; `pivo`=30 сек, `clear`=10 сек. Регистрируется на `dp.message` в `main.py`. |
| **P1-5** | Нет lock-файла | ⚠️ **Декоративно — файл создан, но не используется.** [`requirements.lock`](requirements.lock) содержит 24 пакета, но `Dockerfile` ставит `requirements.txt`, CI ставит `requirements-dev.txt`, в самом lock'е нет `mypy`/`ruff`. Реальной воспроизводимости не даёт. См. п.9.4 — требует решения до merge. |
| **P1-6** | Нет CI | ✅ [`.github/workflows/ci.yml`](.github/workflows/ci.yml) на матрице `python-version: [3.12, 3.13, 3.14]`, шаги `ruff check` → `mypy app/` → `unittest discover`. |
| **P1-7** | Нет линтеров и mypy | ✅ [`pyproject.toml`](pyproject.toml) с `ruff` (E/F/I/UP, line-length=100) и `mypy strict для app.*`; legacy-модули вынесены в `[[tool.mypy.overrides]] ignore_errors = true`. |
| **P1-8** | Нет тестов хендлеров | ✅ [`tests/test_handlers.py`](tests/test_handlers.py) — 18 happy-path тестов, по одному-двум на каждую команду каждого роутера; покрытие через прямой вызов функций с `MagicMock`/`AsyncMock`. |

### P2 — частично

| ID | Описание | Статус | Комментарий |
|---|---|---|---|
| **P2-1** | Реестр настроек как один источник истины (Settings ↔ RuntimeState ↔ runtime_config) | ❌ | Отложено: реализация требует переписать три модуля (`settings.py`, `runtime_state.py`, `runtime_config.py`), не блокирует merge, не критично. |
| **P2-2** | Dockerfile hardening (non-root user, HEALTHCHECK, pin minor) | ❌ | См. п.9.3. |
| **P2-3** | README: разделы Tests/Architecture/Privacy, убрать абсолютные ссылки | ❌ | См. п.9.1. |
| **P2-4** | LOG_LEVEL из `.env`, структурированные логи | ❌ | См. п.10. |
| **P2-5** | Расширить `BOT_TEXT_ALIASES` через `.env` | ❌ | Не блокирующее. |

### P3 — отложены

- Маскирование `chat_id` в логах через HMAC.
- Магические числа (`range(4)` в `learning.py`, `len(clean) < 3 or > 500`) → константы.
- Удалить неиспользуемое поле `is_bot` в `pivo_chat_members` (пишется при `subscribe`, нигде не читается; уже фильтруется до записи).
- Структурированные логи (JSON) — отложить до выбора системы агрегации.

### Известные регрессии относительно `main`

Эти изменения **не были отмечены в первой редакции аудита**, найдены при независимой ревизии. Это поведенческие изменения, которые в коде проходят как нормальные, но для пользователя выглядят как регресс.

#### R1. Админ-команды без прав теперь молча игнорируются (P1)

**Где:** [`app/handlers/admin.py:43,100,140`](app/handlers/admin.py:43), [`app/filters/admin_or_owner.py:14-26`](app/filters/admin_or_owner.py:14).
**Было в `main`:** `/set` и `/setprob` отвечали «Команда доступна OWNER_ID и администраторам чата.»; `/clear` — «Недостаточно прав. Нужен OWNER_ID или права админа чата.»
**Стало:** `AdminOrOwner()` filter возвращает `False` → handler не вызывается → пользователь не получает ничего. Для `/clear` это особенно неприятно: пользователь думает, что бот сломан или не видит команду.
**Как исправить:**
- вариант A — fallback handler с тем же `Command(...)` без `AdminOrOwner()` после защищённого, отвечающий «нет прав»;
- вариант B — вернуть проверку в handler с явным `reply` при отказе, оставив `GroupOnly()` как фильтр.

В фазе 5 этот случай был осознанно проигнорирован под формулировкой «behavior change is acceptable» — это была ошибка суждения, перед merge должно быть исправлено.

#### R2. `/clear` молча сбрасывается throttling-middleware (P3)

**Где:** [`app/middlewares/throttling.py:35-37`](app/middlewares/throttling.py:35).
**Было в `main`:** rate-limit отсутствовал (`/clear` всегда выполнялся).
**Стало:** при попытке повторить `/clear` в течение 10 секунд middleware возвращает `None`, ответа нет. Для `/pivo` (шумная команда) это OK, но для админской `/clear` — пользователь не понимает, почему бот молчит.
**Как исправить:** для админ-команд возвращать короткий ответ «слишком часто, подождите N секунд» вместо silent drop. Для `/pivo` оставить silent drop.

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

### 5.5. Что НЕ улучшилось архитектурно

- **Дублирование настроек** (старая P2-1) сохранилось: `Settings`, `RuntimeState`, `apply_runtime_setting` описывают одни и те же 20+ полей трижды. Это **самый ощутимый оставшийся техдолг**, и он будет царапать каждый раз, когда добавляется новая настройка. Не блокирует merge, но первая постмёрж-задача.
- **`runtime_config.py`** обращается к `RuntimeState` через `setattr(state, key, value)` без проверок, что `key` действительно типизирован — это работает, но бьёт по `mypy strict` (поэтому модуль и в `ignore_errors`).

---

## 6. Безопасность — текущее состояние

### 6.1. Хранение сообщений (бывшая P0-1)

Колонка `messages.text` удалена. Хранится только `normalized_text` (нужен для `is_duplicate` SQL-fallback при <3 токенах). Это уже **не сырой текст пользователя** — `sanitize_text` убирает `@mentions` и приводит пробелы к одиночным. Tokenizable-но не идентично исходному. Это резкое улучшение privacy-постуры.

**Что не сделано:** нет команды `/learn_off` / `/privacy` для opt-out на уровне чата. Это не было в скоупе фаз 1–6 (фаза 4 была про устранение хранения сырого текста, а не про opt-out flow). Можно вынести в P2.

#### Уточнение: `is_duplicate` — эвристика разнообразия, а не privacy-guard

В фазе 4 фильтр был сделан **намеренно вероятностным**: `random.randint(3, min(5, len(tokens)))` выбирает длину префикса для проверки. Одно и то же сгенерированное сообщение может пройти один раз и быть отклонено в другой. Цель — снизить «занудность» бота, а не гарантировать защиту от копирования обучающих фраз.

Это создаёт концептуальную неоднозначность:
- если читать имя метода **`is_duplicate`** и комментарий **«дедупликация»**, это звучит как защита от утечки обучающих сообщений (privacy-guard);
- по реализации — это **эвристика разнообразия генерации**: гарантий нет, near-copy с длинным префиксом могут пройти.

**Что нужно сделать:** либо явно зафиксировать в комментариях/docstring, что это эвристика разнообразия (и переименовать, например, в `looks_like_known_text`), либо сделать детерминированной — проверять **все** длины префикса от 3 до `min(5, len(tokens))` и отклонять при любом совпадении. Текущая полу-формулировка вводит в заблуждение читателя кода.

**Срочность:** P3 (концептуальная чистка).

### 6.2. Авторизация и throttling

- `OWNER_ID` или admin чата → проверяет [`AdminOrOwner`](app/filters/admin_or_owner.py); при ошибке `bot.get_chat_administrators` (например, бот не в чате) → возвращает `False` (deny by default). Тесты покрывают этот случай.
- `/pivo` / `/clear` → ограничены [`ThrottlingMiddleware`](app/middlewares/throttling.py): `pivo`=30 сек, `clear`=10 сек **на пользователя в данном чате**. Проброшенные команды (без `from_user`, не команды, не из throttle-списка) пропускаются без задержек.

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

Без изменений с прошлого аудита: `chat_id` в логах не маскируется, `user_id` — не пишется (хорошо), `pivo`-операции пишут только `mentions count: N` (хорошо). Уровень захардкожен `INFO`. См. п.10.

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
| 7.1 | [`app/handlers/learning.py:162`](app/handlers/learning.py:162) | `for attempt in range(4)` — магическое число попыток генерации | P3 |
| 7.2 | [`app/handlers/learning.py:75`](app/handlers/learning.py:75) | `if len(clean) < 3 or len(clean) > 500` — магические длины | P3 |
| 7.3 | [`runtime_state.py`](runtime_state.py) + [`runtime_config.py`](runtime_config.py) + [`settings.py`](settings.py) | Дублирование 20+ полей в трёх местах | P2 |
| 7.4 | [`bot_policy.py:6`](bot_policy.py:6) | `BOT_TEXT_ALIASES = {"pepe", "пепе"}` — хардкод | P2 |
| 7.5 | [`db.py:160-162`](db.py:160) | `save_message_and_update_model` имеет тот же `cursor.fetchone()[0]` шаблон, что и `markov_repo.get_chat_token_volume` (репозиторий уже исправил это `assert row is not None`) | P3 |
| 7.6 | [`text_utils.py:6`](text_utils.py:6) | `MENTION_RE = r"@\w+"` — на email-адресах удалит только `@host` | P3 |

Все эти пункты не блокирующие, документировать как backlog.

### 7.3. Мёртвый код

Проверено явно:

- `_ensure_messages_normalized_text_column` и `_anonymize_message_author_ids` — упоминаются только в `PROJECT_AUDIT.md` и `REFACTOR_PLAN.md`; в коде их нет (логика переехала в миграции 002 и 003). ✅
- `MessagesRepo.exists()` — используется в [`LearningService.is_duplicate`](app/services/learning_service.py:49) для текстов из <3 токенов (SQL-fallback). Не мёртвый. ✅
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

### 9.5. `requirements.lock` — декоративное наличие (P1)

Файл [`requirements.lock`](requirements.lock) был создан в фазе 6 как закрытие пункта P1-5 из старого аудита. **Реально он не используется ни одним консумером** — это формальная галочка, не дающая воспроизводимости:

| Где должен использоваться | Что используется фактически |
|---|---|
| `Dockerfile` (production-сборка) | `requirements.txt` (диапазоны) |
| CI (`.github/workflows/ci.yml`) | `requirements-dev.txt` → `-r requirements.txt` (диапазоны) |
| Документация по локальной установке | `requirements.txt` (диапазоны) |

Дополнительно: в `requirements.lock` нет `mypy` и `ruff`, хотя CI без них упадёт. Нет процедуры обновления (когда и кто перегенерирует lock). Нет указания, какой Python-version он отражает.

**Это не безобидная мелочь.** Если оставить как есть, документация ниже (этот файл, README в будущем) будет утверждать, что у проекта есть lock-файл — но воспроизводимости не будет, и при апгрейде какой-нибудь зависимости это всплывёт на проде.

**Что нужно решить до merge:**
- **Вариант A** (предпочтительный) — `Dockerfile` ставит `requirements.lock`, CI ставит `requirements.lock` + `mypy/ruff` отдельно (или включить их в lock). Документировать процедуру `pip-compile`/`pip freeze` для обновления.
- **Вариант B** — переименовать в `requirements.constraints` и явно описать как «справочный snapshot, не lock». Удалить упоминания «lock» в коммитах/документации.
- **Вариант C** — удалить файл и вернуться к диапазонам. Если воспроизводимость не цель — это честнее декорации.

**Срочность:** P1, должно быть решено перед merge.

---

## 10. Логирование

Без изменений с прошлого аудита:
- Уровень захардкожен `INFO` в [`main.py:25`](main.py:25).
- `aiogram` приглушён до `WARNING`.
- Формат — обычный текст: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`.
- Идентификаторы пользователей не пишутся, `chat_id` пишется в открытом виде.
- Нет ID-корреляции запросов.
- Нет аудита админ-команд (`/clear`, `/set`) в БД.

**Срочность:** P2 (LOG_LEVEL из `.env`, маскирование `chat_id`).

---

## 11. Тестирование

### 11.1. Состояние

| Файл | Строк | Тестов | Что покрывает |
|---|---:|---:|---|
| `test_db_logic.py` | 247 | ~14 | save_message_and_update_model, get_stats, clear_chat |
| `test_filters.py` | 180 | 18 | GroupOnly (4), AdminOrOwner (6), ThrottlingMiddleware (8) |
| `test_handlers.py` | 238 | 18 | happy-path для всех 4 роутеров |
| `test_learning_service.py` | 141 | 13 | prefix-cache дедупликация |
| `test_markov_and_text.py` | 427 | ~30 | генерация, токенизация, sanitize_text |
| `test_members.py` | 225 | 18 | KeyDerivation, MembersRepo, MembersService (включая шифрование payload) |
| `test_migrator.py` | 495 | ~12 | пустая БД, повторный init, legacy fixture, real-schema fixture |
| `test_pivo.py` | 152 | ~18 | HMAC, Fernet, build_pivo_mention |
| `test_runtime_config.py` | 137 | ~20 | apply_runtime_setting для всех ключей |
| `test_settings.py` | 86 | ~10 | load_settings, валидация |
| `test_bot_messages.py` | 104 | ~10 | format_*_message |
| `test_bot_policy.py` | 86 | ~10 | bot_is_mentioned, should_reply |
| **Итого** | **2518** | **181** | |

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

- **E2E `/pivo`-flow** (subscribe → call → mention render → unsubscribe) — нет интеграционного теста, есть только разрозненные unit-тесты компонентов.
- **Smoke-тест на wiring `main.py`** — нет теста, который проверял бы, что `dp.include_router(...)` действительно регистрирует все 4 роутера, что middleware подключён, что нужные ключи положены в `dp[...]`. Если кто-то случайно удалит строку `dp.include_router(pivo_handlers.router)` — тесты пройдут, бот сломается.
- **`PivoService.build_call_message`** в реальной комбинации с `MembersRepo` через DB — не покрыт целиком.
- **Поведение хендлеров при ошибках Telegram API** (`bot.get_me()` падает, `bot.send_chat_action` 5xx) — не покрыто.
- **Тесты на отказ авторизации с явным ответом** — отсутствуют (см. R1 в разделе «Известные регрессии»). Если регрессия будет исправлена, тест на «неавторизованный `/clear` получает понятное сообщение» должен быть добавлен сразу.
- **Реальная гонка throttling** (одновременные вызовы) — не тестируется, но in-memory dict не имеет race conditions в asyncio (single-threaded).

E2E `/pivo` и smoke `main.py` — P2. Тест на denied auth — приходит вместе с R1-фиксом.

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
| 1 | CI зелёный на ветке `refactor/structure` | ✅ (CI #20, последний коммит `037b401`) |
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
| **B1** | UX-регрессия: `/set`/`/setprob`/`/clear` без прав молча игнорируются (R1) | `037b401` | Добавлены fallback-handlers `cmd_set_denied`/`cmd_setprob_denied`/`cmd_clear_denied` после защищённых; aiogram перебирает handlers по порядку, fallback срабатывает при `AdminOrOwner=False`. Покрытие: 4 теста (3 на тексты + 1 smoke на порядок регистрации). |
| **B2** | `requirements.lock` не выполняет роль lock-файла | `9a052af` | `Dockerfile` устанавливает `requirements.lock` (вместо `requirements.txt`); `requirements-dev.txt` ссылается на `-r requirements.lock`; добавлен header с описанием стратегии (pip freeze без pip-tools) и процедурой регенерации; убрана транзитивная dev-зависимость `ast_serialize`. |
| **B3** | README: 2 абсолютные ссылки ведут в `/D:/test/...` | `d342c3e` | Заменены на относительные `compose.yaml` и `.env.example`. |

### 13.3. После merge — желательно сразу

1. **Branch protection** для `main` в GitHub:
   - require PR review;
   - require CI зелёный;
   - запретить force-push.
2. **CHANGELOG.md** или GitHub Release с описанием фаз 1–6 — чтобы оператор бота знал, что обновляется.
3. Создать issues по оставшемуся P2/P3-долгу (см. раздел 14).

### 13.4. Рекомендация

**Готово к merge.** Все три блокера B1/B2/B3 закрыты коммитами `037b401`, `9a052af`, `d342c3e` соответственно. CI #20 зелёный на матрице 3.12/3.13/3.14, локально 185 тестов проходят, ruff и mypy без замечаний.

**Тип merge:** `merge commit` без squash. 31+ маленьких коммитов фаз дают полезный bisect-friendly след — каждая фаза была атомарной с зелёными тестами и явным сообщением.

В архитектурном смысле ветка чистая: P0 закрыт, P1 закрыт полностью, регрессии относительно `main` устранены, тестами покрыто хорошо. Оставшийся P2/P3-техдолг документирован в разделе 14 и не блокирует production.

---

## 14. Оставшийся техдолг (актуальный список)

### P0 — пусто

Закрыт фазами 1–6.

### P1 — закрыты

| # | Описание | Закрыто в |
|---|---|---|
| **P1-A** (B1) | Вернуть явный отказ для unauthorized `/set`/`/setprob`/`/clear` | ✅ `037b401` |
| **P1-B** (B2) | Подключить `requirements.lock` к Dockerfile и CI | ✅ `9a052af` |
| **P1-C** (B3) | README: починить 2 абсолютные ссылки | ✅ `d342c3e` |

### P2 — желательно в ближайшие 1–2 итерации

| # | Описание | Где | Эффект |
|---|---|---|---|
| **P2-1** | Реестр настроек как один источник истины (Settings ↔ RuntimeState ↔ runtime_config) | [`settings.py`](settings.py), [`runtime_state.py`](runtime_state.py), [`runtime_config.py`](runtime_config.py) | Самый ощутимый архитектурный техдолг |
| **P2-2** | Migrator: перейти на `conn.executescript()` + явные транзакции с rollback (см. п.8.1) | [`app/infrastructure/migrator.py`](app/infrastructure/migrator.py) | Снимает хрупкость для будущих SQL-миграций |
| **P2-3** | Dockerfile hardening (non-root, HEALTHCHECK, pin минорной версии) | [`Dockerfile`](Dockerfile) | Best practices, безопасность контейнера |
| **P2-4** | README: разделы Tests/Architecture/Privacy, `Python 3.12+` вместо `Python 3.14`, инструкция «локальные проверки» | [`README.md`](README.md) | UX для нового пользователя/контрибьютора |
| **P2-5** | Решить судьбу `MembersService`: либо подключить через продуктовый сценарий, либо вынести в feature-branch до реального использования | [`app/services/members_service.py`](app/services/members_service.py), `main.py` | Сейчас инфраструктура без runtime-вызовов |
| **P2-6** | `LOG_LEVEL` из `.env` | [`main.py:25`](main.py:25), [`.env.example`](.env.example) | Эксплуатация |
| **P2-7** | E2E-тест `/pivo` flow (subscribe → call → unsubscribe) | `tests/test_pivo_e2e.py` (новый) | Регрессии в основном opt-in flow |
| **P2-8** | Smoke-тест на wiring `main.py` (все routers зарегистрированы, middleware подключён, ключи в `dp[]`) | `tests/test_main_wiring.py` (новый) | Защита от случайного удаления строки в compose root |
| **P2-9** | Команда `/learn_off` / `/learn_on` или `/privacy` для opt-out обучения на уровне чата | новый handler + repo | Завершает privacy-историю, начатую удалением `messages.text` |
| **P2-10** | Расширить `BOT_TEXT_ALIASES` через `.env` | [`bot_policy.py:6`](bot_policy.py:6) | Конфигурируемость |

### P3 — косметика и долгосрочное

- Магические числа → константы ([`learning.py:75`](app/handlers/learning.py:75), [`learning.py:162`](app/handlers/learning.py:162)).
- **Зафиксировать концепцию `is_duplicate`**: либо переименовать/задокументировать как эвристику разнообразия (см. п.6.1), либо сделать детерминированной (`for length in range(3, min(5, len(tokens))+1): if tuple(tokens[:length]) in cache: return True`). Текущая полу-формулировка вводит в заблуждение.
- Throttling: для админских команд (`/clear`) возвращать «слишком часто, подождите» вместо silent drop (см. R2).
- Маскирование `chat_id` в логах через HMAC.
- Удалить неиспользуемое поле `is_bot` в `pivo_chat_members` (миграция `006_drop_is_bot.py`).
- Структурированные логи (JSON) — отложить до выбора системы агрегации.
- `MENTION_RE` в [`text_utils.py:6`](text_utils.py:6) — переписать так, чтобы не ломать email-адреса.
- Метрики (Prometheus / aiogram-middleware) — заводить, когда будет куда отправлять.
- Унифицировать стиль `cursor.fetchone()` в [`db.py:160-162`](db.py:160) (`assert row is not None`, как уже сделано в `markov_repo`).

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

| ID | Локация | Суть | Риск | Приоритет |
|---|---|---|---|---|
| ~~P1-A~~ | `app/handlers/admin.py` | UX-регрессия в админ-командах | UX | ✅ закрыто `037b401` |
| ~~P1-B~~ | `requirements.lock`, `Dockerfile` | lock-файл декоративен | воспроизводимость | ✅ закрыто `9a052af` |
| ~~P1-C~~ | `README.md` | абсолютные ссылки `/D:/test/...` | документация | ✅ закрыто `d342c3e` |
| P2-1 | `settings.py` / `runtime_state.py` / `runtime_config.py` | Тройное дублирование настроек | DX / ошибки при добавлении | P2 |
| P2-2 | `app/infrastructure/migrator.py` | хрупкий `sql.split(";")` без транзакций | будущие миграции | P2 |
| P2-3 | `Dockerfile` | non-root, HEALTHCHECK, pin минорной | hardening | P2 |
| P2-4 | `README.md` | нет разделов Tests/Architecture/Privacy, `Python 3.14` vs `3.12+` | UX | P2 |
| P2-5 | `app/services/members_service.py` | инфраструктура без runtime-использования | поверхность поддержки | P2 |
| P2-6 | `main.py:25`, `.env.example` | LOG_LEVEL захардкожен | эксплуатация | P2 |
| P2-7 | `tests/` | нет E2E-теста `/pivo` flow | регрессии | P2 |
| P2-8 | `tests/` | нет smoke-теста на wiring `main.py` | потеря роутера незаметна | P2 |
| P2-9 | (новый handler) | нет opt-out на обучение чата | privacy-завершение | P2 |
| P2-10 | `bot_policy.py:6` | `BOT_TEXT_ALIASES` хардкод | конфигурируемость | P2 |
| P3-1 | `learning.py:75,162` | магические числа | качество | P3 |
| P3-2 | `app/services/learning_service.py:36-56` | `is_duplicate` — концептуальная неоднозначность (privacy-guard vs эвристика) | путаница в коде | P3 |
| P3-3 | `app/middlewares/throttling.py` | silent drop для админ `/clear` | UX | P3 |
| P3-4 | логи | `chat_id` без маскирования | privacy (низко) | P3 |
| P3-5 | `pivo_chat_members.is_bot` | неиспользуемое поле | чистка | P3 |
| P3-6 | `text_utils.py:6` | MENTION_RE на emails | corner case | P3 |

---

**Дата актуализации:** 2026-05-08, третья редакция.
**Статус:** ветка `refactor/structure` **готова к merge в `main`**. Все P0/P1 закрыты, CI #20 зелёный на матрице 3.12/3.13/3.14, 185 тестов проходят, ruff/mypy без замечаний.
**История ревизии:**
- первая редакция (2026-05-08, до Codex-ревизии) рекомендовала «можно мёржить»;
- вторая редакция (2026-05-08, после Codex-ревизии) уточнила список блокеров P1-A/B/C;
- третья редакция (2026-05-08, после закрытия блокеров `037b401`, `9a052af`, `d342c3e`) — готово к merge.
