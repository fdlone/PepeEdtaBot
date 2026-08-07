# Архитектура проекта

Документ описывает текущую слоистую архитектуру `PepeEdtaBot`. Если вы трогаете
код впервые — читайте его вместе с [GENERATION_PIPELINE.md](GENERATION_PIPELINE.md)
(конвейер ответа) и [OPEN.md](OPEN.md) (что не закрыто).

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
        │  handlers/  (aiogram.Router × 5)                       │
        │   common  /ping /help /stats                           │
        │   admin   /config /set /setprob /clear  + denied-fallback │
        │   pivo    /pivo /pivo_on /pivo_off /pivo_privacy       │
        │   learning  F.text — обучение и генерация              │
        └────────────────────────────────────────────────────────┘
                  │
                  ▼
        ┌────────────────────────────────────────────────────────┐
        │  services/  (бизнес-логика, не знают про aiogram*)     │
        │   ReplyPipeline   LearningService   PivoService         │
        └────────────────────────────────────────────────────────┘
                  │
                  ▼
        ┌────────────────────────────────────────────────────────┐
        │  repositories/  (per-domain SQL)                       │
        │   markov / messages / chat_members / pivo_usage        │
        └────────────────────────────────────────────────────────┘
                  │
                  ▼
        ┌────────────────────────────────────────────────────────┐
        │  infrastructure/database.py (соединение, миграции,     │
        │  доступ к репозиториям и кросс-доменные транзакции      │
        │  типа save_message_and_update_model)                    │
        └────────────────────────────────────────────────────────┘
                  │
                  ▼
        ┌────────────────────────────────────────────────────────┐
        │  infrastructure/migrator.py  (NNN_*.sql / .py — once)  │
        └────────────────────────────────────────────────────────┘
```

**Направление зависимостей.** `app/core/` не знает, откуда берутся данные: он
объявляет порт `app/core/markov_port.py` (семь методов чтения цепи, ровно те,
что вызываются), а `MarkovRepo` ему удовлетворяет — реализацию подставляет
`main.py` (`MarkovGenerator(db=db.markov)`). До этого ядро импортировало конкретный класс `Database`, и цена
была не в чистоте: из-за цикла импортов инфраструктура держала собственные
копии набора пунктуации, размера контентной n-граммы и функции построения
окон 4-грамм, а их расхождение молча развело бы накопительный индекс цитат с
тем, что видит скорер. Определение окна теперь одно (`candidate_scorer.
verbatim_ngram_windows`); копия в миграции 016 оставлена намеренно — миграции
застывшая история и не должны меняться вместе с кодом. Правило проверяется
`tests/test_layering.py`.

**Слой доступа к данным.** SQL живёт в репозиториях, и сервисы обращаются к
ним напрямую: `db.markov.get_starts(...)`, `db.chat_emoji_stats.bump(...)`.
Однострочных методов-делегатов в `Database` нет — раньше их было около сорока,
и каждый новый метод репозитория приходилось дублировать в фасаде, иначе им
нельзя было пользоваться. `Database` остаётся владельцем того, что репозиторию
не принадлежит: соединения и его блокировки, миграций, настроек SQLite, точки
доступа к репозиториям и кросс-доменных операций. Репозитории отдаются
свойствами, поэтому обращение до `await db.init()` даёт понятную ошибку, а не
`None`.

\* Исключение — `PivoService.subscribe(chat_id, user)` принимает
`aiogram.types.User` напрямую: осознанный компромисс — сервису нужны сразу
несколько полей пользователя (`id`, `username`, `full_name`/`first_name`/
`last_name`), и промежуточный DTO дублировал бы форму `aiogram.types.User`
без практической выгоды при текущем размере проекта.

## Модули

### Compose root
- `main.py` — единственная точка инициализации. `run_bot()` собирает граф
  зависимостей, `configure_dispatcher()` регистрирует middleware, ключи в
  `dp[...]` и роутеры. Никакой бизнес-логики.

### `app/`

| Подпакет | Содержимое |
|---|---|
| `config/` | `registry.py`, `settings.py`, `runtime_config.py`, `runtime_state.py`, `defaults.py` |
| `core/` | `markov.py`, `response_generator.py`, `candidate_scorer.py`, `context_state_matcher.py`, `morphology.py`, `gen_trace_log.py`, `reply_flavor.py`, `emoji.py`, `hot_ngrams.py`, `mood.py`, `lexicon.py`, `privacy_filter.py`, `reply_policy.py`, `text.py` |
| `domain/` | `pivo.py`, `pivo_templates.py` |
| `presentation/` | `bot_messages.py`, `fallback_phrases.py` |
| `handlers/` | `common.py`, `admin.py`, `pivo.py`, `learning.py`, `errors.py` — `aiogram.Router` per file. `_helpers.py` — `reply_humanized`, `reply_humanized_sequence`. |
| `services/` | `reply_pipeline.py`, `learning_service.py`, `pivo_service.py`, `pivo_message_builder.py`, `pivo_parser.py` |
| `repositories/` | `markov_repo.py`, `messages_repo.py`, `chat_members_repo.py`, `pivo_usage_repo.py`, `pivo_pool_usage_repo.py`, `chat_emoji_stats_repo.py`, `chat_hot_ngrams_repo.py`, `chat_user_interactions_repo.py` |
| `filters/` | `group_only.py` (только `GROUP`/`SUPERGROUP`), `admin_or_owner.py` (`OWNER_ID` или админ чата, fail-closed при ошибке Telegram API) |
| `middlewares/` | `throttling.py` — per-user-per-command cooldown, `clear`=3600 сек; команды из `notify_on_throttle` получают явный ответ при throttle вместо silent drop |
| `infrastructure/` | `database.py` — соединение, миграции и кросс-доменные операции; `migrator.py` — пробегает `app/migrations/NNN_*` ровно один раз. `.sql`-файлы оборачиваются в `BEGIN; ... COMMIT;` и проходят через `executescript`; на исключении вызывается `conn.rollback()` и схема не остаётся в полу-применённом состоянии |
| `migrations/` | `001_initial.sql` … `015_lowercase_model.py` |

### Внутренние модули пакета

| Файл | Назначение |
|---|---|
| `app/infrastructure/database.py` | Соединение `aiosqlite` и его блокировка, миграции, настройки SQLite, доступ к репозиториям (`db.markov`, `db.chat_members`, …) и кросс-доменные операции: `save_message_and_update_model`, `clear_chat`, `clear_pivo_chat_data`, `get_stats`, суточные затухания и retention `pivo_daily_usage`. Однострочных делегатов в репозитории не держит — потребители ходят в репозиторий напрямую. |
| `app/core/markov.py` | Variable-order генератор (3 → 2; порядка 1 нет — его блуждания были словесным салатом). Контекстно-аффинные старты, topic-drift jumps. |
| `app/core/morphology.py` | Приближённый русский стеммер (`stem_token`) — единый fold-ключ для IDF-релевантности и аффинности стартов. Контекст-матчинг стемы не использует: этот тир удалён 2026-07-14 (см. [CLOSED.md](CLOSED.md)). |
| `app/core/gen_trace_log.py` | Пошаговый лайв-трейс отбора кандидатов (логгер `chat_markov.gen`), включается env-флагом `GEN_TRACE_LOG`; поведения не меняет. |
| `app/core/response_generator.py` | Конвейер best-of-N: генерация кандидатов, фильтры (verbatim, echo, анти-повтор), softmax-отбор по скорингу, reply flavor. |
| `app/core/candidate_scorer.py` | Скоринг кандидатов: качество завершения, длина по режимам short/medium/long, IDF-релевантность контексту (по стемам, с echo-гардом), штрафы повторов (включая бывший diversity-компонент) и дословного цитирования (рамп от 60% корпусных 4-грамм). |
| `app/core/mood.py` | Пер-чатовое настроение (sleepy/calm/lively/heated) из EWMA-сигналов; модулирует поведение генерации (M1). |
| `app/core/reply_flavor.py` | Вариации финальной пунктуации ответа (QW5); редкие события и фальстарты (L3): ролл и трансформация ответа в последовательность сообщений (вердикт/КАПС/двойное сообщение/филлер). |
| `app/core/emoji.py` | Эмодзи-канал (M3): извлечение эмодзи из текста (без tone-модификаторов, флаги собираются из пары региональных индикаторов) и частотный сэмплинг для добавления в конец ответа; `strip_trailing_emojis` снимает добавленное эмодзи перед анти-повторным сравнением. |
| `app/core/hot_ngrams.py` | «Локальные мемы» (L1): извлечение контентных би/триграмм выученного сообщения для окна горячих n-грамм; горячие n-граммы изредка сидируют самостоятельные ответы через seed API генератора. |
| `app/services/reply_pipeline.py` | Конвейер ответа на обычное сообщение: ритм и настроение чата, эмодзи-канал, гейты обучаемости (`observe`), политика ответа с генерацией, причудами и редкими событиями (`respond`), запись в модель (`learn`). Типов Telegram не знает — обработчик передаёт факты (`IncomingMessage`) и функцию отправки, поэтому ветки политики проверяются без моков Telegram. |
| `app/presentation/fallback_phrases.py` | Пулы fallback-фраз с анти-повтором, ночными и «heated»-вариантами. |
| `app/domain/pivo.py` | `PivoSecurity` (HMAC + Fernet), `PivoMember`. |
| `app/domain/pivo_templates.py` | Контент сообщений `/pivo`. |
| `app/presentation/bot_messages.py` | Форматирование `/help`, `/stats`, `/config`. |
| `app/core/reply_policy.py` | `bot_is_mentioned`, cooldown, `should_reply` + `DEFAULT_BOT_TEXT_ALIASES` (встроенные «прозвища», на которые бот отзывается). |
| `app/config/settings.py` | `Settings` dataclass + `load_settings(env)`. |
| `app/config/runtime_state.py` | `RuntimeState` dataclass + `runtime_state_from_settings`. |
| `app/config/runtime_config.py` | Тонкая обёртка вокруг `app.config.registry.try_apply` для `/set`. |
| `app/config/registry.py` | **Единый реестр** runtime-mutable полей (`FieldSpec` на каждый параметр + `validate_cross_fields`). |
| `app/core/text.py` | `sanitize_text` — убирает `@mention`, схлопывает пробелы. |

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
dp["bot_text_aliases"] = settings.bot_text_aliases
```

Никаких самодельных DI-контейнеров. Размер проекта это не оправдывает.

> Ключ для `RuntimeState` — `runtime_state` (не `state`), чтобы не
> конфликтовать с aiogram-овским `FSMContext`, который под именем `state`
> подкладывается dispatcher'ом автоматически.

## Конфигурация

Single source of truth для runtime-mutable полей — `app/config/registry.py`:

```python
RUNTIME_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("reply_probability", "REPLY_PROBABILITY", "0.08",
              _float_in_range(0.0, 1.0)),
    ...
)
```

- `load_settings()` (`app/config/settings.py`) итерируется по реестру и читает
  значения из env через `spec.parse`.
- `runtime_state_from_settings()` копирует runtime-mutable значения
  в `RuntimeState` через тот же реестр.
- `apply_runtime_setting()` (`app/config/runtime_config.py`) ищет `FieldSpec` по имени
  ключа из `/set`, парсит, проверяет на shallow copy через
  `validate_cross_fields`, и только потом мутирует живой state.

Чтобы добавить новый runtime-mutable параметр, нужно:
1. добавить строчку в `RUNTIME_FIELDS`;
2. добавить поле в dataclass `Settings` и `RuntimeState` (нужно для
   статической типизации);
3. (опционально) при cross-field инварианте — расширить
   `validate_cross_fields`.

Поля, которые не должны быть mutable в runtime (`BOT_TOKEN`, `OWNER_ID`,
`DB_PATH`, `PIVO_*_SECRET`, `LOG_LEVEL`, `BOT_TEXT_ALIASES`), живут
только в `Settings` и парсятся вручную в `load_settings`.

`BOT_TEXT_ALIASES` имеет специальное поведение «безопасного fallback'а»:
если переменная не задана или содержит только разделители — берутся
встроенные `app.core.reply_policy.DEFAULT_BOT_TEXT_ALIASES = {"pepe", "пепе"}`, чтобы
бот отвечал на свои стандартные прозвища даже без редактируемого
`.env` (например, на проде с ограниченным доступом к контейнеру).

## Модель данных

| Таблица | Назначение |
|---|---|
| `messages` | `chat_id`, `author_id` (анонимизирован), `normalized_text`, `created_at`. Сырой `text` удалён миграцией 005. |
| `starts3` / `transitions3` | Основная цепь порядка 3: состояние `(w1, w2, w3)` → `w4`. Токены в нижнем регистре (`NORMALIZE_LOWER=true`); накопленную модель разово привела к нему миграция 015. |
| `starts` / `transitions` | Бэкофф порядка 2: состояние `(w1, w2)` → `w3`. Цепь порядка 1 (`transitions1`) удалена миграцией 013. |
| `chat_members` | Каноническая таблица участников чата. PK `(chat_hash, user_hash)`, payload зашифрован Fernet. Сейчас единственный потребитель — `/pivo` (присутствие в таблице ≡ подписка); будущие фичи, которым нужно персистентное состояние участника, ходят сюда же. Профиль (`@username`, имя) освежается по сообщениям подписчика, но не чаще раза в сутки (`PivoService.refresh_member`) — иначе устаревший ник в упоминании никого не тегает. |
| `pivo_daily_usage` | Суточная квота `/pivo` (`chat_hash`, `user_hash`, `usage_day`, `used_count`). Retention 7 дней. |
| `pivo_pool_usage` | Анти-повтор шаблонов `/pivo`: последние использованные индексы top/body/bottom per chat per pool (`chat_hash`, `pool_name`, `recent_indices`). Миграция 010. |
| `chat_emoji_stats` | Частоты эмодзи per chat для эмодзи-канала (`chat_id`, `emoji`, `cnt`, `updated_at`); ключ — сырой `chat_id`, как у таблиц модели; чистится в `clear_chat`, стареющие строки затухают. Миграция 011 (M3). |
| `chat_hot_ngrams` | Скользящее окно контентных n-грамм per chat (`chat_id`, `w1`, `w2`, `w3` (`''` для биграмм), `cnt`, `updated_at`); «горячесть» = доля оконного счётчика от всевременного в `transitions` (для биграмм — SUM по `w3`). Ключ — сырой `chat_id`; чистится в `clear_chat`, затухает при старте. Миграция 012 (L1). |
| `chat_user_interactions` | Счётчик отвеченных обращений per user per chat для L2-причуд (`chat_id`, `user_hash`, `cnt`, `updated_at`); `user_hash` — HMAC-SHA256 под `PIVO_HMAC_SECRET` (как у `/pivo`), никаких имён/username. Ключ — сырой `chat_id`; чистится в `clear_chat`, затухает за ~30 дней тишины (медленнее мемных таблиц). Миграция 014 (L2). |
| `schema_migrations` | Учёт применённых миграций. |

### Кэши в памяти

`LearningService` держит по чату производные структуры, чтобы путь ответа не
перечитывал базу: окно последних `TEXT_CACHE_MAX_MESSAGES` нормализованных
текстов (гейт дословного повтора), кумулятивный индекс контентных 4-грамм
(штраф за цитирование), IDF контент-токенов, интонационный профиль и частотный
словарь чата (слот-мутации).

Правила, по которым они живут:

- **Пополнение, а не сброс.** Выученное сообщение добавляется в окно текстов,
  в индекс 4-грамм и в частоты. Раньше эти кэши сбрасывались целиком на каждом
  сообщении, из-за чего в активном чате почти каждый ответ шёл по холодному
  пути — на копии прода это ~95 мс полной пересборки против ~0.04 мс тёплого
  чтения.
- **Отсрочка для тяжёлой статистики.** IDF и интонационный профиль считаются
  по выборке в сотни сообщений и пересобираются не чаще, чем раз в
  `STATS_REBUILD_EVERY_MESSAGES` выученных сообщений: одно сообщение сдвигает
  доли на доли процента.
- **Вытеснение.** Кэши ключуются `chat_id` и подчиняются той же политике, что
  остальные состояния в памяти: срок хранения и предел по числу чатов
  (переиспользуются `RUNTIME_STATE_TTL_SEC` / `RUNTIME_STATE_MAX_CHATS`).
  Вытеснение не меняет решений — данные перечитываются из базы.
- **`/clear` забывает кэши чата** (`LearningService.forget_chat`): иначе гейт
  дословного повтора продолжал бы считать цитатой сообщения, которых в базе
  уже нет.

Тем же правилом ограничен суточный дроссель обновления профилей `/pivo`
(`PivoService`): при смене суток словарь сбрасывается целиком, потому что
записи прошлых дней бесполезны.

## Миграции

`app/infrastructure/migrator.py`:

- сканирует `app/migrations/NNN_<name>.{sql,py}`;
- сравнивает с записями в `schema_migrations`;
- применяет недостающие в порядке имён;
- `.sql` оборачивается в `BEGIN; <тело>; COMMIT;` и пропускается через
  `conn.executescript(...)` — это даёт **атомарность**: stdlib
  `sqlite3.executescript` неявно делает `COMMIT` перед запуском, и без
  явного `BEGIN` каждый DDL внутри файла авто-коммитился бы по мере
  выполнения. Теперь при падении в середине файла `run()` ловит
  исключение, делает `conn.rollback()`, in-flight-транзакция
  откатывается, и `schema_migrations` не получает запись о
  половинчатой миграции;
- `.py` импортирует и вызывает `await mod.apply(conn)`;
- на исключении `await conn.rollback()` и raise — запись в
  `schema_migrations` не появится для не до конца применённой миграции.

**Конвенция:** файлы `app/migrations/*.sql` НЕ должны содержать собственных
`BEGIN`/`COMMIT` — runner добавляет их сам.

## Безопасность

- `OWNER_ID` или admin чата → `AdminOrOwner` filter (fail-closed при
  ошибке Telegram API).
- `/pivo` / `/clear` → `ThrottlingMiddleware`.
- `chat_members` (и зависящий от неё `/pivo`-flow): HMAC-индексированный
  `chat_hash`/`user_hash` под `PIVO_HMAC_SECRET`, Fernet-шифрование
  payload под `PIVO_ENCRYPTION_SECRET`. Прежняя HKDF-инфраструктура с
  доменными метками (`members:hmac` / `members:encryption`) и
  `chat_member_profiles` была удалена в миграции 007 как
  «заготовка под несуществующие домены». Если в будущем появится
  независимый домен, HKDF можно подключить отдельным сервисом.
- Все SQL-запросы параметризованы.
- `messages.text` больше не хранится; только `normalized_text` после
  `sanitize_text`.

## Docker

Контейнер собран из `python:3.14.0-slim` (pin минорной версии, осознанный
апгрейд при изменении). Внутри:

- При сборке создаётся пользователь `bot` (UID 1000), `/app` принадлежит ему.
- `requirements.lock` ставится первым слоем (cache-friendly).
- В образ копируется белый список — `main.py` и `app/` (вместе с
  `app/migrations/*.sql`), — а не весь контекст. Данные чата, тесты, `tools/`,
  `docs/`, `openspec/` физически не попадают в слои, даже если кто-то забыл
  дописать строку в `.dockerignore`. Сам `.dockerignore` остаётся вторым
  рубежом и сокращает контекст, уходящий демону.
- `HEALTHCHECK` каждые 60 секунд проверяет, что Python-интерпретатор жив
  (бот polling-only, HTTP endpoint'а нет — это формальная заглушка).
- **`docker-entrypoint.sh`** запускается от root: best-effort
  `chown -R bot:bot /app/data`, затем `exec runuser -u bot -- "$@"`. Это
  защищает от «host bind-mount принадлежит другому UID/GID, бот не может
  писать в `/app/data`». `runuser` входит в `util-linux` в slim-образе,
  ничего доустанавливать не нужно.
- Если оператор гарантирует владельца хост-директории, можно запустить
  контейнер с `--user 1000:1000` — тогда entrypoint пропустит команду без
  изменений.
- `.gitattributes` фиксирует LF-окончания для `*.sh`, чтобы Windows-клон
  не сделал CRLF и не сломал shebang в Linux-контейнере.

## Окружения БД

- `data/markov.db` — runtime-база по умолчанию, проброшена в Docker
  volume `./data:/app/data`.
- `markov.db` в корне репозитория — локальная тестовая БД (не боевая).
- Существующие БД мигрируются автоматически при `Database.init()`.

## Тесты и CI

- `tests/` — широкий набор unit-тестов (`unittest`), включая smoke на
  `configure_dispatcher`, атомарность миграций, фикстуру с реальной
  legacy-схемой (`tests/fixtures/legacy_real_schema.sql`).
- CI: `.github/workflows/ci.yml` — матрица Python 3.12/3.13/3.14, шаги
  `ruff check` → `mypy app/` → `unittest discover` (с coverage) → `bandit` →
  `pip-audit`; отдельный job собирает Docker-образ и прогоняет по нему smoke:
  `import main` внутри контейнера плюс сверка числа миграций в образе с
  репозиторием (белый список `COPY` иначе молча теряет файлы — сборка
  останется зелёной, а бот упадёт при первом подключении к БД).

Команды для локального прогона см. в `README.md`.
