# PepeEdtaBot

Telegram-бот для группового чата. Учится на сообщениях чата и генерирует ответы на цепях Маркова без внешней LLM.

## Стек
- Python 3.14
- aiogram v3
- SQLite + aiosqlite
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

Файл конфигурации: [compose.yaml](/D:/test/PepeEdtaBot/compose.yaml)

В Docker по умолчанию используется база `data/markov.db`, подключенная через volume
`./data:/app/data`. Файл `markov.db` в корне репозитория является локальной
тестовой базой и не считается боевым хранилищем.

## Основные настройки
Все параметры перечислены в [.env.example](/D:/test/PepeEdtaBot/.env.example).

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
- `/stats` — статистика модели по текущему чату.
- `/config` — текущие runtime-настройки процесса.
- `/config full` — полный список runtime-настроек.
- `/set help` — подсказка по ключам для `/set`.
- `/set <key> <value>` — изменить runtime-настройку, доступно `OWNER_ID` или админам чата.
- `/setprob 0.2` — быстрый setter вероятности ответа.
- `/clear confirm` — очистить данные текущего чата.

Через `/set` можно менять runtime-настройки из `/config`. Изменения действуют только до перезапуска.

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

## Безопасность
- не коммитьте `.env`;
- не храните реальные токены в репозитории;
- при утечке токена перевыпустите его в BotFather.
