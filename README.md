# PepeEdtaBot

Telegram-бот для группового чата. Учится на сообщениях чата и генерирует ответы на цепях Маркова без внешней LLM.

## Стек
- Python 3.12+ (CI прогоняет матрицу 3.12 / 3.13 / 3.14)
- aiogram v3
- SQLite + aiosqlite
- `cryptography` (Fernet, HKDF) для шифрования и доменных ключей `/pivo`
- конфигурация через `.env`

## Быстрый старт
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

## Docker
Разовый запуск через `docker run`:

```bash
docker build -t pepe-edta-bot .
docker run -d --name pepe-edta-bot --env-file .env -v ${PWD}/data:/app/data pepe-edta-bot
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
- `/help` — список команд.
- `/ping` — проверка, что бот онлайн.
- `/pivo` — позвать подписанных участников в Discord шуточным сообщением.
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
`/pivo` работает только по opt-in: бот зовёт только тех пользователей, которые сами включили себя командой `/pivo_on`.

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
- `app/services/` — бизнес-логика (`LearningService`, `PivoService`,
  `MembersService`).
- `app/repositories/` — SQL по доменам (`markov`, `messages`, `pivo`,
  `members`, `pivo_usage`).
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
  Fernet'ом. Ключи выводятся через HKDF-SHA256 от `PIVO_*_SECRET` с
  доменными метками (`members:hmac`, `members:encryption`).
- `/pivo_privacy` показывает пользователю, что хранится для `/pivo`.
- Раскрытые уровни логирования (`LOG_LEVEL=DEBUG` и т.п.) могут
  печатать `chat_id` — на проде используйте `INFO` или выше.

## Тесты
Проект использует стандартный `unittest` (без pytest) и линтеры `ruff` +
`mypy strict` для `app/` (legacy-модули вынесены в `ignore_errors`).

Установка dev-зависимостей:

```bash
pip install -r requirements-dev.txt
```

Локальные проверки (соответствуют шагам CI):

```bash
python -m ruff check app/ tests/
python -m mypy app/
python -m unittest discover tests -v
```

CI-конфиг: [`.github/workflows/ci.yml`](.github/workflows/ci.yml). Тесты
прогоняются на матрице Python 3.12 / 3.13 / 3.14.

## Безопасность
- не коммитьте `.env`;
- не храните реальные токены в репозитории;
- при утечке токена перевыпустите его в BotFather.
