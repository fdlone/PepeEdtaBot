# План рефакторинга PepeEdtaBot

Документ для возобновления работы. Читать **перед** любой следующей сессией рефакторинга.

Связанные документы:
- [PROJECT_AUDIT.md](PROJECT_AUDIT.md) — полный технический аудит с приоритетами P0–P3.
- Этот файл — оперативный план: где мы сейчас, что делать следующим, как.

---

## Статус: фаза 1 (структурный рефакторинг) завершена

- **Ветка:** `refactor/structure` (запушена в origin, под защитой CI).
- **Не сливать в main**, пока не закончим хотя бы фазу 2 (миграции).
- **Последний коммит:** `b58e773 refactor: extract LearningService and learning Router`.
- **Тесты:** 83/83 зелёные на каждом коммите фазы 1.
- **Поведение бота:** идентично исходному (ни одна команда, ни один сценарий обучения/ответа не изменён).

### Что сделано в фазе 1

| Коммит | Что |
|---|---|
| `4612644` | `db.py` разбит на репозитории `app/repositories/{markov,messages,pivo}_repo.py`. `Database` — фасад с делегатами. |
| `6d4116d` | CI: `.github/workflows/ci.yml`, тесты на 3.12/3.13/3.14. |
| `526dc4a` | `PROJECT_AUDIT.md`. |
| `4da9b2c` | `PivoService` — бизнес-логика opt-in /pivo. |
| `32e0f88` | `app/handlers/pivo.py` — Router с командами /pivo*. |
| `cd9e85e` | `app/handlers/common.py` — /ping, /help, /stats. |
| `e1765c2` | `app/handlers/admin.py` — /clear, /set, /setprob, /config + проверки прав. |
| `b58e773` | `LearningService` + `app/handlers/learning.py` — обработчик F.text. |

### Метрики

- `main.py`: **588 → 82 строки** (-86%). Теперь это compose root и больше ничего.
- В `app/`: 11 новых файлов.
- `markov.py`, `pivo_templates.py`, `bot_messages.py`, `bot_policy.py`, `pivo.py`, `runtime_*.py`, `settings.py`, `text_utils.py` — **не трогали**.

---

## Текущая структура

```
PepeEdtaBot/
├── main.py                          # 82 строки, только wiring
├── app/
│   ├── handlers/
│   │   ├── _helpers.py              # reply_humanized, is_group_message
│   │   ├── common.py                # /ping, /help, /stats
│   │   ├── admin.py                 # /clear, /set, /setprob, /config + auth helpers
│   │   ├── pivo.py                  # /pivo, /pivo_on, /pivo_off, /pivo_privacy
│   │   └── learning.py              # F.text + extract_context_tokens
│   ├── services/
│   │   ├── pivo_service.py          # subscribe / unsubscribe / build_call_message
│   │   └── learning_service.py      # record_message
│   └── repositories/
│       ├── markov_repo.py           # все read-only по starts/transitions
│       ├── messages_repo.py         # exists()
│       └── pivo_repo.py             # upsert / remove / list_members
├── db.py                            # Database — фасад: соединение + миграции +
│                                    # save_message_and_update_model (атомарный)
│                                    # + clear_chat + get_stats + делегаты
├── markov.py                        # НЕ ТРОГАТЬ
├── pivo_templates.py                # НЕ ТРОГАТЬ (контент)
├── pivo.py                          # PivoSecurity, PivoMember, чистые функции
├── bot_messages.py                  # форматирование текстов
├── bot_policy.py                    # bot_is_mentioned, cooldown, should_reply
├── settings.py                      # Settings dataclass + load_settings
├── runtime_state.py                 # RuntimeState dataclass
├── runtime_config.py                # apply_runtime_setting + ALLOWED_RUNTIME_KEYS
├── text_utils.py                    # sanitize_text
└── tests/                           # unittest, 83 теста
```

---

## Установленные конвенции — следовать им во всех новых модулях

### 1. Репозитории (`app/repositories/`)

Каждый репозиторий получает в конструктор:
```python
def __init__(self, conn_provider: ConnProvider, lock: asyncio.Lock) -> None:
    self._conn_provider = conn_provider
    self._lock = lock
```

`ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]` — это `Database._get_conn`. Каждый метод репозитория сам берёт `async with self._lock` и сам вызывает `commit()`. Кросс-доменные транзакции — **не в репозитории**, а в `Database` (см. `save_message_and_update_model`, `clear_chat`).

В `Database.init()` репозитории создаются после миграций и **до** `commit()` на схеме (порядок важен): сейчас они инстанцируются последними тремя строками в `init()`.

### 2. Сервисы (`app/services/`)

- Принимают `db: Database` или нужный репозиторий + другие сервисы/секрет-объекты.
- **Не зависят от aiogram.** Получают и возвращают чистые типы или dataclass'ы.
- Исключение: `PivoService.subscribe(chat_id, user)` принимает `aiogram.types.User`. Это допустимо: для Telegram-бота `User` — естественный входной тип. Если когда-то понадобится тестировать без aiogram — можно ввести Protocol с минимальными атрибутами (id/username/full_name/is_bot). Пока не делаем.

### 3. Хендлеры (`app/handlers/`)

Используем `aiogram.Router`:
```python
router = Router(name="<domain>")

@router.message(Command("foo"))
async def cmd_foo(message: Message, foo_service: FooService, state: RuntimeState) -> None:
    ...
```

Зависимости приходят через **встроенный DI aiogram v3**: что положено в `dp["key"]`, попадает в параметры хендлера по имени. **Не наша самоделка — это штатный механизм workflow_data.**

### 4. main.py — только compose root

Создаёт зависимости, кладёт в `dp[]`, подключает `Router`'ы через `dp.include_router()`, запускает polling. Никакой бизнес-логики, никаких хендлеров.

### 5. Имена

- Репозитории: `XxxRepo` с методами `get/list/upsert/remove`.
- Сервисы: `XxxService` с глаголами действий.
- Хендлеры: `cmd_<command>` или `on_<event>` (например, `on_text_message`).
- Файлы хендлеров: домен (common/admin/pivo/learning), без префикса `handlers_`.

### 6. Тесты

**Импортируют из канонических мест.** Не из `main.py` — там больше нет re-export'ов. Если новый тест использует `extract_context_tokens` — берёт из `app.handlers.learning`, не из main.

---

## Что **НЕ** трогать

- **`markov.py`** — алгоритм генерации, покрыт тестами, работает. Менять только при изменении самого алгоритма.
- **`pivo_templates.py`** — контентный файл, не код.
- **Схема БД** в `db.py` — никаких ALTER, никаких новых таблиц до фазы 2 (миграции). Если нужна новая таблица — сначала вводим миграционный механизм.
- **Контракт `.env`** — никаких новых обязательных переменных без явного согласования.
- **`PIVO_HMAC_SECRET` / `PIVO_ENCRYPTION_SECRET`** — менять или ротировать ломает существующие подписки. Если нужны новые секреты для других доменов — добавлять отдельные переменные, не переиспользовать эти.
- **Поведение хендлеров** — все ответы, тексты, тайминги, retry-логика должны остаться идентичными. Любое изменение поведения — отдельный коммит с явным обсуждением, не в общем потоке рефакторинга.

---

## Подводные камни (узнал на собственном опыте)

1. **`aiogram` автоматически инъектирует `Bot`** в параметры хендлеров — `dp["bot"] = bot` не нужен. Но `Database`/`Settings`/любые наши объекты — **нужен** `dp["..."] = ...` явно.
2. **`Bot` ↔ `bot_username`/`bot_id`** — последние нужно класть в `dp[]` отдельно (`dp["bot_username"] = ...; dp["bot_id"] = me.id`), потому что `me = await bot.get_me()` делается один раз при старте, а в хендлерах нужно знать username и id без повторного запроса.
3. **`save_message_and_update_model` атомарен** — он трогает 5 таблиц в одной транзакции. **Нельзя дробить** на вызовы разных репозиториев. Оставить на `Database` или ввести явный «UnitOfWork» паттерн (пока избыточно).
4. **`db._get_conn` возвращает текущее соединение или бросает RuntimeError** — это контракт для репозиториев. Не пытаться открывать новое соединение в репозитории.
5. **CI matrix включает 3.12/3.13/3.14** — `setup-python@v5` поддерживает 3.14. Если в CI вылезет ошибка по 3.14 — убрать его из матрицы (`Dockerfile` оставить как есть, бот в проде на 3.14).
6. **Локальный `git config` уже настроен** для этого репо: `fdlone <hvedchenyas@gmail.com>`. Не менять.
7. **`.claude/settings.local.json`** игнорируется через `.gitignore` (запись `.claude/`). Не коммитить.

---

## План фаз 2–6

Идём строго по приоритетам из аудита. Каждая фаза — отдельная серия маленьких коммитов на ветке `refactor/structure`. Тесты должны быть зелёными после каждого коммита.

### Фаза 2 — Версионирование миграций (P0)

**Цель:** обеспечить воспроизводимое создание/обновление схемы перед добавлением любых новых таблиц.

**Шаги:**

1. Завести таблицу `schema_migrations(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, applied_at TEXT NOT NULL DEFAULT (datetime('now')))`.
2. Создать каталог `app/migrations/` с файлами в формате `NNN_<short_name>.sql` (или `.py` для миграций с логикой). Первый файл — `001_initial.sql` — содержит весь текущий `CREATE TABLE` блок из `Database.init()`. Второй — `002_normalize_messages_text_column.py` — ALTER + бэкфил из `_ensure_messages_normalized_text_column`. Третий — `003_anonymize_authors.py` — UPDATE из `_anonymize_message_author_ids`.
3. Создать `app/repositories/migrations_repo.py` (или `app/infrastructure/migrator.py`):
   - `list_pending(applied: set[str]) -> list[Migration]`
   - `apply(migration: Migration, conn) -> None`
4. `Database.init()`: после connect выполнять `migrator.run()`. Старый код inline-CREATE удалить.
5. **Тесты на migrator:** новый файл `tests/test_migrator.py` — пустая БД → миграции применяются по порядку → второй запуск ничего не делает → новая миграция применяется только она.
6. **Тест на совместимость:** существующая БД (с уже созданными таблицами) после init по-прежнему работает — добавить fixture с pre-baked схемой.

**Критерий готовности:** новая БД создаётся через миграции; существующие БД (`data/markov.db` пользователя) не ломаются; добавление новой таблицы — это новый файл `004_*.sql`, и больше ничего.

**Ожидаемый объём:** 4–6 коммитов.

**Подвох:** SQLite не поддерживает большинство ALTER операций. Все «изменения схемы» — через CREATE TABLE _new + INSERT SELECT + DROP + RENAME. Учесть в `002_*`.

---

### Фаза 3 — Таблица чувствительных данных участников (P0)

**Цель:** заложить инфраструктуру для будущего хранения профилей пользователей с шифрованием и opt-in.

**Шаги:**

1. **Миграция `004_chat_member_profiles.sql`:**
   ```sql
   CREATE TABLE chat_member_profiles (
       chat_hash TEXT NOT NULL,
       user_hash TEXT NOT NULL,
       encrypted_user_id TEXT NOT NULL,
       encrypted_payload TEXT NOT NULL,         -- JSON, зашифрован Fernet'ом
       key_version INTEGER NOT NULL DEFAULT 1,  -- для будущей ротации
       consent_version INTEGER NOT NULL,        -- версия согласия
       consented_at TEXT NOT NULL DEFAULT (datetime('now')),
       updated_at TEXT NOT NULL DEFAULT (datetime('now')),
       PRIMARY KEY(chat_hash, user_hash)
   );
   CREATE INDEX idx_chat_member_profiles_chat_hash ON chat_member_profiles(chat_hash);
   ```
2. **HKDF derivation** в `pivo.py` (или новом `app/security/key_derivation.py`):
   - Один master-secret из `.env` → HKDF → доменные ключи `pivo` и `members`.
   - `MEMBERS_HMAC_SECRET` и `MEMBERS_ENCRYPTION_SECRET` либо отдельные env-переменные, либо derived через HKDF из мастера. Решить **в момент начала фазы**, согласовав с пользователем.
3. **`MembersRepo`** в `app/repositories/members_repo.py` — те же паттерны (`conn_provider, lock`).
4. **`MembersService`** в `app/services/members_service.py`:
   - `record_consent(chat_id, user, payload: dict, consent_version: int) -> None`
   - `get_profile(chat_id, user_id) -> Optional[dict]`
   - `revoke(chat_id, user_id) -> None`
   - Внутри: HMAC, Fernet шифрование JSON payload.
5. **Тесты** на сервис и репозиторий.

**Хендлеры/команды НЕ добавлять** в этой фазе — только инфраструктура. Команды — отдельной продуктовой задачей.

**Идентификация без `@username`:**
- `MembersService` принимает `aiogram.User`, использует `user.id` для HMAC и шифрования.
- Username хранится только если есть, и только как часть payload, не как ключ поиска.
- В будущих упоминаниях — `tg://user?id=...` (как уже сделано в `pivo.build_pivo_mention`).

**Критерий готовности:** новая таблица существует, репо+сервис покрыты тестами, ни одна существующая команда не задета.

**Ожидаемый объём:** 4–5 коммитов.

---

### Фаза 4 — Privacy: opt-out на обучение и решение по `messages.text` (P0)

**Цель:** закрыть главный privacy-риск из аудита (п.6.1 — сырые сообщения хранятся без opt-in).

**Шаги:**

1. **Согласовать с пользователем:** убираем поле `messages.text` совсем (для Markov достаточно агрегатов в `transitions*`) или оставляем, но добавляем opt-out на уровне чата. Рекомендация: **убрать `messages.text`**, оставить только `normalized_text`. `message_exists` нигде не вызывается из активного кода — её можно удалить вместе с полем.
2. Если убираем — миграция `005_drop_messages_text.sql` (через CREATE+INSERT+DROP+RENAME, см. подвох в фазе 2).
3. Если оставляем — добавить таблицу `chat_learning_settings(chat_id, learning_enabled, updated_at)` + команды `/learn_off`, `/learn_on` в `admin.py` (или отдельный `learning_admin.py`).
4. Обновить `README.md` и `pivo_privacy`-стилизованное сообщение `/help` или новую команду `/privacy`.
5. **Тесты** на новый flow.

**Критерий готовности:** понятная privacy-политика, нет сырого текста сообщений (либо есть opt-out), README обновлён.

**Ожидаемый объём:** 3–5 коммитов.

---

### Фаза 5 — Фильтры авторизации и throttling middleware (P1)

**Цель:** убрать дублирование `_can_manage_settings`/`_is_owner`/`_is_chat_admin` и закрыть риск abuse команд.

**Шаги:**

1. `app/filters/group_only.py`:
   ```python
   class GroupOnly(BaseFilter):
       async def __call__(self, message: Message) -> bool:
           return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
   ```
2. `app/filters/admin_or_owner.py`:
   ```python
   class AdminOrOwner(BaseFilter):
       async def __call__(self, message: Message, bot: Bot, settings: Settings) -> bool:
           # перенести логику _can_manage_settings из admin.py
   ```
3. Применить декоративно:
   ```python
   @router.message(Command("clear"), GroupOnly(), AdminOrOwner())
   ```
4. Удалить `_is_owner`, `_is_chat_admin`, `_can_manage_settings` из `admin.py`.
5. **Throttling middleware** в `app/middlewares/throttling.py`:
   - `dp.message.middleware(ThrottlingMiddleware(per_user_per_command={"pivo": 30, "clear": 10}))`
   - Хранилище: in-memory dict с TTL (для одного процесса достаточно).
6. **Тесты** на фильтры (можно на чистых функциях) и middleware (с FakeMessage).

**Критерий готовности:** `admin.py` короче на ~40 строк, `/pivo` нельзя спамить чаще раза в 30 секунд.

**Ожидаемый объём:** 3–4 коммита.

---

### Фаза 6 — Качество и DX (P1)

**Цель:** ruff/mypy/lock-файл/тесты хендлеров.

**Шаги:**

1. `pyproject.toml` с разделами `[tool.ruff]` и `[tool.mypy]`. Для mypy — `strict = true` **только для `app/`**, остальное `ignore_errors = false` но без strict.
2. `requirements.lock` через `python -m pip freeze` в чистом venv. `Dockerfile` использует `requirements.lock`, README — `requirements.txt`.
3. CI: добавить шаги `ruff check` и `mypy app`. Не добавлять `ruff format --check` сразу — сначала прогнать форматирование одним коммитом, чтобы не получить миллион diff'ов.
4. **Тесты хендлеров:** для каждого роутера — минимум один happy-path тест через aiogram's TestDispatcher или прямой вызов `await cmd_xxx(message=fake_message, ...)` с моком сервиса.

**Критерий готовности:** CI зелёный с ruff+mypy, lock-файл существует, каждый router имеет хотя бы один интеграционный тест.

**Ожидаемый объём:** 5–7 коммитов.

---

## Что **отложить**

- Миграция с `unittest` на `pytest` — необязательно, текущие тесты работают.
- Миграция с SQLite на Postgres — нет реальной потребности.
- Метрики (Prometheus) — заведём, когда будет куда отправлять.
- `pydantic-settings` вместо ручного парсера — добавляет зависимость без явной выгоды.
- DI-фреймворки (`dependency-injector`, `punq`) — `dp[]` хватает.
- Структурированные логи (JSON) — отдельная фаза, когда определимся с системой агрегации.

---

## Как продолжить работу (для будущей сессии)

1. Прочитать **этот файл** и **PROJECT_AUDIT.md**.
2. Убедиться, что мы на ветке `refactor/structure`:
   ```
   git status
   git log --oneline -8
   ```
   Должен быть последний коммит `b58e773` или новее (если фаза уже начата).
3. Прогнать тесты: `python -m unittest discover tests` — должно быть **OK** (сейчас 83 теста).
4. Спросить у пользователя, какую фазу начинаем (рекомендация: **фаза 2 — миграции**, она разблокирует фазы 3 и 4).
5. После каждого коммита — пушить в `origin/refactor/structure`, ждать зелёный CI на GitHub. Не делать длинных серий локальных коммитов без push.
6. **Не сливать ветку в main** до явного согласования.
7. Перед новыми изменениями повторно сверяться с разделом «Что НЕ трогать» в этом документе.

### Команды для быстрой ориентации

```bash
git status
git log --oneline refactor/structure ^origin/main      # коммиты, не слитые в main
python -m unittest discover tests                       # все тесты
python -m unittest tests.test_db_logic                  # один файл
python -c "import main; print('ok')"                    # импорт-смоук
```

---

## Контакты и идентичность

- Локальный git: `fdlone <hvedchenyas@gmail.com>` (только этот репозиторий, не глобально).
- Origin: `https://github.com/fdlone/PepeEdtaBot`.
- CI: GitHub Actions, файл `.github/workflows/ci.yml`.
- Просмотреть прогон CI: https://github.com/fdlone/PepeEdtaBot/actions

---

**Статус документа:** актуален на коммит `b58e773` (фаза 1 завершена). Обновлять после каждой завершённой фазы — добавлять в раздел «Статус» и помечать выполненные пункты в плане.
