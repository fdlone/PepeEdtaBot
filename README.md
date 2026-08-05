# PepeEdtaBot

[![CI](https://github.com/fdlone/PepeEdtaBot/actions/workflows/ci.yml/badge.svg)](https://github.com/fdlone/PepeEdtaBot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **EN:** A Telegram group-chat bot that learns from the chat's own messages
> and replies via a per-chat Markov chain — fully algorithmic, no LLM. Chat
> mood tracking, context-aware candidate scoring with Russian stemming, a
> "personality" layer (chat-local memes, emoji, rare form breaks, regulars'
> quirks) and privacy-first storage (HMAC-hashed ids, no raw texts kept).
> Python 3.12+, aiogram 3, SQLite. Docs are in Russian — start with
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Telegram-бот для группового чата. Учится на сообщениях чата и генерирует
ответы на цепях Маркова — полностью алгоритмически, без внешней LLM.

## Что умеет

- **Генерация ответов** — пер-чатовая цепь Маркова 3-го порядка с бэкоффом до
  2-го; best-of-N конвейер: до 5 кандидатов, скоринг (завершённость, длина по
  режимам short/medium/long, IDF-релевантность контексту со стеммингом русской
  морфологии, штрафы за повторы и дословное цитирование), softmax-отбор.
- **Ответы «в тему»** — сообщение, на которое отвечают, задаёт контекст:
  якорение стартовых состояний (exact/casefold-матчинг), биас по стемам,
  дословные цитаты корпуса достраиваются «отсебятиной» вместо отбраковки;
  позиция контекстного якоря варьируется — часть ответов подхватывает тему
  не с первого слова, а серединой или концом («сегментация якоря»).
- **Настроение чата** — sleepy/calm/lively/heated из темпа сообщений и
  эмфатики; модулирует вероятность ответа, вариативность, длину и flavor.
  Серия прямых обращений «заводит» бота — эскалация в heated от трёх
  обращений подряд.
- **AI-директор ответов** — вероятность самостоятельного ответа следует за
  моментумом беседы (бёрсты после ответа, суточные капы, анти-флуд обращений).
- **Слой «личности»** — эмодзи из словаря конкретного чата, «локальные мемы»
  (горячие n-граммы последних дней сидируют ответы), редкие сломы формы
  (вердикт/КАПС/двойное сообщение/фальстарт), причуды для завсегдатаев
  («опять ты» — отдельным сообщением перед ответом).
- **`/pivo`** — шуточный созыв в Discord по opt-in подписке с анти-повтором
  шаблонов и временными вариациями (ночь/пятница/понедельник).
- **Privacy-first** — сырые тексты не хранятся дольше retention-окна, авторы
  анонимизированы, идентификаторы — только HMAC-хэши, `chat_id` в логах
  маскируется; всё стирается `/clear confirm`.

## Стек
- Python 3.12+ (CI прогоняет матрицу 3.12 / 3.13 / 3.14)
- aiogram v3
- SQLite + aiosqlite
- `cryptography` (Fernet для `/pivo`, HKDF для маскирования `chat_id` в логах)
- конфигурация через `.env`

## Быстрый старт

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

1. Создайте `.env` на основе [.env.example](.env.example).
2. Заполните `BOT_TOKEN`.
3. Опционально задайте `OWNER_ID`.
4. Запустите:

```bash
python main.py
```

Отключите privacy mode у бота в BotFather, иначе бот не увидит сообщения группы:
`Bot Settings -> Group Privacy -> Turn off`.

## Docker
Контейнер стартует как `root`, затем `docker-entrypoint.sh` приводит
владельца `/app/data` к пользователю `bot` (UID 1000) и сбрасывает
привилегии через `runuser` — это нужно для bind-mount-сценария, когда
хостовая `./data` принадлежит другому UID/GID. Сам процесс бота всегда
работает от непривилегированного пользователя.

Если хост-директория уже принадлежит UID 1000 (или используется named
volume), entrypoint можно обойти, запустив контейнер с
`--user 1000:1000` — тогда команда сразу стартует от `bot`.

Разовый запуск через `docker run`:

```bash
docker build -t pepe-bot:latest .
docker run -d --name pepe-bot --env-file .env -v ${PWD}/data:/app/data pepe-bot:latest
```

Пересборка и перезапуск через Docker Compose ([compose.yaml](compose.yaml)):

```bash
docker compose up -d --build   # запуск/пересборка
docker compose down            # остановка
```

## Конфигурация
Все параметры с описаниями — в [.env.example](.env.example). Чаще всего меняются:

- `REPLY_PROBABILITY` — вероятность случайного ответа без прямого обращения.
- `MIN_COOLDOWN_SEC` — минимальная пауза между самостоятельными ответами.
- `MIN_TOKENS_FOR_MODEL` — минимальный объём модели для генерации.
- `MAX_REPLY_TOKENS` / `MAX_REPLY_CHARS` — основной (в токенах) и аварийный
  (в символах) лимиты длины ответа.
- `RANDOMNESS_STRENGTH` — сила вариативности генерации.
- `REPETITION_PENALTY_STRENGTH` — подавление повторов токенов и n-грамм в ответе.
- `RECENT_REPLY_PENALTY_STRENGTH` — анти-повтор между ответами: штраф за
  пересечение кандидата с последними 20 отправленными ответами чата (точные
  совпадения отбрасываются всегда).
- `LENGTH_MODE_WEIGHTS` — веса режимов целевой длины ответа
  (short/medium/long), выбираемых на каждый ответ.
- `LENGTH_CONTEXT_ADAPTATION` — насколько длина сообщения, на которое отвечают,
  наклоняет выбор режима: короткое тянет к short, длинное — к long; `0`
  отключает наклон.
- `MOOD_ENABLED` / `MOOD_MODULATION_STRENGTH` — скрытое настроение чата
  (sleepy/calm/lively/heated) из темпа сообщений, эмфатики и частоты обращений;
  плавно меняет вероятность ответа, вариативность, длину и flavor. Пороги и
  сглаживание задаются `MOOD_*` ключами; `MOOD_MODULATION_STRENGTH=0` оставляет
  трекинг настроения, но отключает влияние на поведение.
- `USE_REPLY_CONTEXT` — учитывать текст сообщения, на которое отвечают, как
  контекст генерации (настраивается группой ключей `REPLY_CONTEXT_*`).
- `HOT_NGRAM_SEED_CHANCE` — «локальные мемы» (L1): вероятность начать
  самостоятельный ответ с горячей n-граммы — фразы, которую чат подхватил за
  последние ~7 дней (пороги: `HOT_NGRAM_MIN_COUNT`, `HOT_NGRAM_RECENCY_SHARE`;
  0 полностью выключает канал; ответы на обращения не сидируются).
- `RARE_EVENT_CHANCE` / `FALSE_START_CHANCE` — редкие «сломы формы» (L3):
  односложный вердикт, КАПС или двойное сообщение (`RARE_EVENT_CHANCE`) и
  фальстарты «филлер → печатает… → ответ» (`FALSE_START_CHANCE`); общий
  суточный бюджет на чат — `RARE_EVENT_DAILY_CAP`; 0 отключает
  соответствующий канал.
- `USER_QUIRK_CHANCE` — причуды для завсегдатаев (L2): шанс предварить ответ
  на обращение «постоянного» пользователя коротким вокативом («опять ты»)
  отдельным сообщением; порог «постоянности» — `USER_QUIRK_MIN_INTERACTIONS`
  отвеченных обращений (с ~30-дневным затуханием); не чаще раза в сутки (UTC)
  на пользователя; 0 полностью выключает канал (включая учёт счётчиков).
- `AUTO_CAPITALIZE_REPLIES` — капитализация начала предложений в ответе
  (output-side постобработка, по умолчанию выключено).
- `MESSAGES_RETENTION_PER_CHAT` — число последних нормализованных сообщений,
  хранимых отдельно для каждого чата; должно быть не меньше
  `TEXT_CACHE_MAX_MESSAGES`.
- `SQLITE_BUSY_TIMEOUT_MS` / `SQLITE_WAL_AUTOCHECKPOINT_PAGES` — ожидание при
  конкурирующей записи и порог автоматического WAL checkpoint.

Многие из этих ключей доступны через `/set` и меняются на лету (см. ниже).

## Команды
Команды предназначены для групповых чатов. Личный чат с ботом не является
поддерживаемым рабочим сценарием.

- `/help` — список команд.
- `/ping` — проверка, что бот онлайн.
- `/pivo [время] [повод] [@кого]` — позвать в Discord шуточным сообщением.
- `/pivo_on` / `/pivo_off` — включить/выключить себя в списке упоминаний `/pivo`.
- `/pivo_privacy` — посмотреть, как используются данные для `/pivo`.
- `/pivo_check` — диагностика упоминаний `/pivo`: каким путём построено
  упоминание каждого подписчика (ссылка по `user_id` / строкой `@ник` /
  пропущен) и почему. Доступно только `OWNER_ID`. Ответ намеренно не содержит
  разметки упоминаний — диагностика никого не пингует. Заведено под O2, см.
  `docs/OPEN.md`.
- `/stats` — статистика модели по текущему чату.
- `/config` — текущие runtime-настройки процесса (`/config full` — полный список).
- `/set <key> <value>` — изменить runtime-настройку (`/set help` — подсказка по
  ключам); доступно только `OWNER_ID` (настройка процесс-глобальная — влияет
  сразу на все чаты, поэтому админы чата больше не допускаются, см. O5 в
  `docs/OPEN.md`).
- `/setprob 0.2` — быстрый setter вероятности ответа; доступно только `OWNER_ID`
  (та же причина, что у `/set`).
- `/clear confirm` — очистить данные текущего чата (модель, сообщения,
  эмодзи-статистику, горячие n-граммы, счётчики взаимодействий, а также
  подписки `/pivo` и их квоты); доступно `OWNER_ID` или админам чата — эта
  команда чат-скоуплена, процесс-глобального эффекта нет.

Изменения через `/set` действуют только до перезапуска процесса.

## /pivo
`/pivo` работает строго по opt-in: без явных упоминаний бот зовёт только тех
пользователей, которые сами включили себя командой `/pivo_on`. Все команды
`/pivo*` доступны только в группах и супергруппах.

Можно добавить время, повод и явные упоминания:

```text
/pivo
/pivo 20:00
/pivo watch movie
/pivo 20:00 watch movie @friend
```

Если в команде есть явные `@mentions`, бот уведомит только их; иначе
используется текущий список подписчиков `/pivo_on`. Число явных упоминаний и
размер списка ограничиваются `PIVO_EXPLICIT_MENTIONS_LIMIT` и
`PIVO_SUBSCRIBER_FANOUT_LIMIT`.

Чтобы шаблоны приглашения не приедались, бот применяет анти-повтор (S2):
последние `PIVO_RECENT_POOL_WINDOW` использованных вариантов top/body/bottom
запоминаются отдельно для каждого чата (в таблице `pivo_pool_usage`) и
исключаются при следующем выборе; `0` отключает механизм. Дополнительно
`PIVO_TEMPORAL_FLAVOR_CHANCE` задаёт вероятность подмены завершающей строки на
тематическую в зависимости от времени (ночь / пятница / понедельник); нейтральный
пул всегда остаётся запасным вариантом. Оба ключа доступны через `/set`.

Аргументы влияют на тело приглашения: время встраивается в фразы про сбор, а
при указанном поводе бот собирает нейтральное тело без игровых activity-фраз
(`СИГейм`, `Codenames`, `рисовалка` и т. п.). Время распознаётся только в
начале аргументов; поддерживаются `20:00`, `today/tomorrow 21:00`,
`сегодня/завтра 21:00`, `evening`, `вечером`, `сегодня/завтра вечером` и их
комбинации. Всё остальное считается свободным описанием повода.

Для хранения подписок нужны секреты в `.env` — `PIVO_HMAC_SECRET` и
`PIVO_ENCRYPTION_SECRET`. Используйте длинные случайные значения и не меняйте
их без необходимости: при смене секретов старые подписки станут недоступны.

## База данных
- Runtime/Docker база по умолчанию: `data/markov.db` (volume `./data:/app/data`).
- Корневой `markov.db`, если он есть, — локальная тестовая база, не боевое хранилище.
- Существующие базы мигрируются автоматически при запуске.

Полностью чистый старт: остановите бота, удалите `data/markov.db`,
`data/markov.db-wal`, `data/markov.db-shm`, запустите снова.

Синтетические данные для локального smoke-теста:

```bash
python -m tools.seed_db --db markov.db
python -m tools.seed_diverse --db markov.db
```

Операционные процедуры (логи, WAL checkpoint, backup, restore) — в
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

## Privacy
- Колонка `messages.text` удалена миграцией `005_drop_messages_text.py`;
  хранится только `normalized_text` (без `@mention`, со схлопнутыми пробелами)
  для дедупликации генерации.
- `author_id` принудительно анонимизируется миграцией `003_anonymize_authors.py`.
- `/pivo` работает строго по opt-in: подписки хранятся как HMAC-индексированные
  хэши `chat_hash`/`user_hash` (HMAC-SHA256 под `PIVO_HMAC_SECRET`), payload
  зашифрован Fernet'ом (ключ — SHA-256 от `PIVO_ENCRYPTION_SECRET`).
  `/pivo_privacy` показывает пользователю, что именно хранится.
- Для L2-причуд («опять ты» завсегдатаям) бот хранит только анонимный счётчик
  взаимодействий per chat: `user_hash` (HMAC-SHA256 под `PIVO_HMAC_SECRET`,
  та же схема, что `/pivo`) и число отвеченных обращений — без имён,
  username и обратимых идентификаторов. Счётчик затухает после ~30 дней
  тишины и стирается `/clear confirm`; `USER_QUIRK_CHANCE=0` отключает и
  запись, и чтение.
- В learning-логах `chat_id` маскируется HKDF-SHA256-derived ключом и коротким
  HMAC-mask; не добавляйте raw `chat_id` в новые log-сообщения.

## Архитектура
Подробное описание слоёв и DI — в [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Краткий обзор:

- `main.py` — compose root: загрузка `Settings`, инициализация БД, сервисов,
  middleware и роутеров.
- `app/handlers/` — пять `aiogram.Router`'ов (`common`, `admin`, `pivo`,
  `learning`, `errors`); зависят только от сервисов.
- `app/services/` — бизнес-логика (`LearningService`, `PivoService`).
- `app/config/` — настройки, runtime-state и реестр изменяемых параметров.
- `app/core/` — генерация ответов: цепь Маркова, конвейер best-of-N (отбор и
  скоринг кандидатов, reply-контекст), текстовая нормализация, privacy-фильтр
  и reply policy.
- `app/domain/` — доменная логика и шаблоны `/pivo`.
- `app/presentation/` — пользовательские тексты и форматирование ответов.
- `app/repositories/` — SQL по доменам (`markov`, `messages`, `chat_members`,
  `pivo_usage`, `pivo_pool_usage`, `chat_emoji_stats`, `chat_hot_ngrams`,
  `chat_user_interactions`).
- `app/filters/` и `app/middlewares/` — `GroupOnly`, `AdminOrOwner`,
  `ThrottlingMiddleware`.
- `app/infrastructure/` — фасад БД и migrator, который однократно прогоняет
  `app/migrations/NNN_*.sql|.py` и пишет в `schema_migrations`.

Реестр runtime-настроек — [`app/config/registry.py`](app/config/registry.py):
любое поле, доступное через `/set`, описано там одной строкой и автоматически
попадает в `Settings`, `RuntimeState` и `apply_runtime_setting`.

## Разработка и тесты
Проект использует стандартный `unittest` (без pytest), `ruff` и `mypy strict`
для `app/`. Источник истины по зависимостям — `requirements*.txt`/`.lock`
(pip); `uv` можно использовать как локальный раннер (`uv run ...`), но lock он
не ведёт (`[tool.uv] managed = false`). Dev-зависимости ставят закреплённый
`requirements.lock` плюс инструменты, совпадающие с CI:

```bash
pip install -r requirements-dev.txt
```

Локальные проверки — те же команды, что и в CI:

```bash
python -m ruff check app/ tests/ tools/ main.py
python -m mypy app/
python -m unittest discover tests -v
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) прогоняет эти
проверки на матрице Python 3.12 / 3.13 / 3.14, добавляет security-сканы
(`bandit`, `pip-audit`) и отдельным job собирает Docker-образ без запуска бота.

Для отладки генерации есть подробный лайв-трейс отбора кандидатов:
`GEN_TRACE_LOG=true` включает его независимо от `LOG_LEVEL` (см.
[`docs/GENERATION_PIPELINE.md`](docs/GENERATION_PIPELINE.md), §10), а
`tools/eval_generation.py` / `tools/eval_prod.py` дают синтетический и
продовый eval конвейера.

## Документация

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — слои, DI, схема БД, миграции.
- [`docs/GENERATION_PIPELINE.md`](docs/GENERATION_PIPELINE.md) — полный путь
  сообщения от хендлера до ответа: скоринг, гейты, ручки.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — операционный runbook: логи,
  WAL checkpoint, backup/restore, retention.
- [`docs/OPEN.md`](docs/OPEN.md) — открытые вопросы и бэклог.
- [`docs/CLOSED.md`](docs/CLOSED.md) — журнал закрытого: аудиты, ревью, фичи.

## Безопасность
- не коммитьте `.env` и не храните реальные токены в репозитории;
- при утечке токена перевыпустите его в BotFather.

## Лицензия
[MIT](LICENSE).
