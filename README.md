# PepeEdtaBot

Telegram-бот для группового чата. Учится на сообщениях чата и генерирует ответы на цепях Маркова без внешней LLM.

## Стек
- Python 3.12+ (CI прогоняет матрицу 3.12 / 3.13 / 3.14)
- aiogram v3
- SQLite + aiosqlite
- `cryptography` (Fernet для `/pivo`, HKDF для маскирования `chat_id` в логах)
- конфигурация через `.env`

## Быстрый старт
Минимальный локальный запуск:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

1. Создайте `.env` на основе `.env.example`.
2. Заполните `BOT_TOKEN`.
3. Опционально задайте `OWNER_ID`.
4. Запустите:

```bash
python main.py
```

Для разработки и локальных проверок используйте dev-зависимости — этот путь
ставит `requirements.lock` и инструменты, совпадающие с CI:

```bash
pip install -r requirements-dev.txt
```

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

Удобная пересборка и перезапуск через Docker Compose:

```bash
docker compose up -d --build
```

Остановка:

```bash
docker compose down
```

Файл конфигурации: [compose.yaml](compose.yaml)

В Docker по умолчанию используется база `data/markov.db`, подключенная через volume
`./data:/app/data`. Файл `markov.db` в корне репозитория является локальной
тестовой базой и не считается боевым хранилищем.

## Основные настройки
Все параметры перечислены в [.env.example](.env.example).

Чаще всего меняются:
- `REPLY_PROBABILITY` — вероятность случайного ответа без прямого обращения.
- `MIN_COOLDOWN_SEC` — минимальная пауза между самостоятельными ответами.
- `MIN_TOKENS_FOR_MODEL` — минимальный объём модели для генерации.
- `MAX_REPLY_TOKENS` — основной лимит длины ответа в токенах.
- `MAX_REPLY_CHARS` — аварийный лимит длины ответа в символах.
- `RANDOMNESS_STRENGTH` — сила вариативности генерации.
- `REPETITION_PENALTY_STRENGTH` — насколько сильно подавлять повторы токенов и n-грамм в одном ответе.

## Команды
Команды предназначены для групповых чатов. Личный чат с ботом не является
поддерживаемым рабочим сценарием.

- `/help` — список команд.
- `/ping` — проверка, что бот онлайн.
- `/pivo [время] [повод] [@кого]` — позвать в Discord шуточным сообщением.
- `/pivo_on` — включить себя в список упоминаний для `/pivo`.
- `/pivo_off` — выключить себя из списка упоминаний для `/pivo`.
- `/pivo_privacy` — посмотреть, как используются данные для `/pivo`.
- `/stats` — статистика модели по текущему чату.
- `/config` — текущие runtime-настройки процесса.
- `/config full` — полный список runtime-настроек.
- `/set help` — подсказка по ключам для `/set`.
- `/set <key> <value>` — изменить runtime-настройку, доступно `OWNER_ID` или админам чата.
- `/setprob 0.2` — быстрый setter вероятности ответа.
- `/clear confirm` — очистить данные текущего чата.

Через `/set` можно менять runtime-настройки из `/config`. Изменения действуют только до перезапуска.

## /pivo
`/pivo` работает только по opt-in: без явных упоминаний бот зовёт только тех
пользователей, которые сами включили себя командой `/pivo_on`.
Команды `/pivo*` доступны только в группах и супергруппах.

Можно добавить время, повод и явные упоминания:

```text
/pivo
/pivo 20:00
/pivo watch movie
/pivo 20:00 watch movie @friend
```

Если в команде есть явные `@mentions`, бот уведомит только этих пользователей.
Если упоминаний нет, используется текущий список подписчиков `/pivo_on`.
Время распознаётся только в начале аргументов. Поддерживаются форматы:
`20:00`, `today 21:00`, `tomorrow 21:00`, `сегодня 21:00`,
`завтра 21:00`, `evening`, `today evening`, `tomorrow evening`, `вечером`,
`сегодня вечером`, `завтра вечером`. Всё остальное считается свободным
описанием повода.

Для хранения списка нужны секреты в `.env`:
- `PIVO_HMAC_SECRET`
- `PIVO_ENCRYPTION_SECRET`

Используйте длинные случайные значения и не меняйте их без необходимости: при смене секретов старые подписки `/pivo` станут недоступны.

## База данных
- Docker/runtime база по умолчанию: `data/markov.db`.
- Корневой `markov.db`, если он есть, используется как локальная тестовая база.
- Существующие базы мигрируются автоматически при запуске.

Если нужен полностью чистый старт:
1. остановите бота;
2. удалите `data/markov.db`, `data/markov.db-wal`, `data/markov.db-shm`;
3. запустите бота снова.

## Важно для групп
Отключите privacy mode у бота в BotFather:
`Bot Settings -> Group Privacy -> Turn off`

## Архитектура
Подробное описание слоёв (`handlers / services / repositories / filters /
middlewares / migrations / security / infrastructure`), DI через
`Dispatcher` и compose root — в [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Краткий обзор:
- `main.py` — compose root: загрузка `Settings`, инициализация БД, сервисов,
  middleware и роутеров.
- `app/handlers/` — четыре `aiogram.Router`'а: `common`, `admin`, `pivo`,
  `learning`. Хендлеры зависят только от сервисов.
- `app/services/` — бизнес-логика (`LearningService`, `PivoService`).
- `app/repositories/` — SQL по доменам (`markov`, `messages`,
  `chat_members`, `pivo_usage`).
- `app/filters/` и `app/middlewares/` — `GroupOnly`, `AdminOrOwner`,
  `ThrottlingMiddleware`.
- `app/infrastructure/migrator.py` — пробегает по `app/migrations/NNN_*.sql|.py`
  один раз и записывает в `schema_migrations`.

Реестр runtime-настроек живёт в [`config_registry.py`](config_registry.py):
любое поле, доступное через `/set`, описано там одной строкой и
автоматически попадает в `Settings`, `RuntimeState` и
`apply_runtime_setting`.

## Privacy
- Колонка `messages.text` удалена миграцией `005_drop_messages_text.py`.
  Хранится только `normalized_text` — с убранными `@mention` и
  схлопнутыми пробелами; используется для дедупликации генерации.
- `author_id` принудительно анонимизируется миграцией `003_anonymize_authors.py`.
- `/pivo` работает строго по opt-in: подписки хранятся как
  HMAC-индексированные хэши `chat_hash` и `user_hash`, payload зашифрован
  Fernet'ом. Хэши считаются через HMAC-SHA256 под `PIVO_HMAC_SECRET`,
  Fernet-ключ строится как SHA-256 от `PIVO_ENCRYPTION_SECRET`.
- `/pivo_privacy` показывает пользователю, что хранится для `/pivo`.
- Для learning-логов `chat_id` маскируется через HKDF-SHA256-derived key
  и короткий HMAC-mask; не добавляйте raw `chat_id` в новые log-сообщения.

## Тесты
Проект использует стандартный `unittest` (без pytest) и линтеры `ruff` +
`mypy strict` для `app/` (legacy-модули вынесены в `ignore_errors`).

Для локальной проверки среды используйте тот же набор зависимостей и команд,
что и CI. Установка dev-зависимостей:

```bash
pip install -r requirements-dev.txt
```

Локальные проверки:

```bash
python -m ruff check app/ tests/
python -m mypy app/
python -m unittest discover tests -v
```

CI-конфиг: [`.github/workflows/ci.yml`](.github/workflows/ci.yml). Тесты
прогоняются на матрице Python 3.12 / 3.13 / 3.14; отдельный job проверяет
Docker build без запуска бота.

## Безопасность
- не коммитьте `.env`;
- не храните реальные токены в репозитории;
- при утечке токена перевыпустите его в BotFather.
