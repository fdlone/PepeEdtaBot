# PepeEdtaBot

Telegram-бот для групповых чатов с генерацией реплик на цепях Маркова без ML и без внешней LLM.

Текущая версия использует модель переменного порядка:
- основной порядок `n=3`;
- fallback `n=2`;
- fallback `n=1`, если включен `ENABLE_BACKOFF`.

Дополнительно бот умеет учитывать `reply_to_message` как локальный контекст и смещать генерацию в сторону сообщения, на которое отвечает.

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

## Основные настройки
Все параметры документированы в [.env.example](/D:/test/PepeEdtaBot/.env.example).

Ключевые runtime-настройки:
- `REPLY_PROBABILITY` — вероятность случайного ответа без прямого обращения.
- `MIN_COOLDOWN_SEC` — минимальная пауза между самостоятельными ответами.
- `MIN_TOKENS_FOR_MODEL` — минимальный объём модели для генерации.
- `MAX_REPLY_CHARS` — ограничение длины ответа.
- `RANDOMNESS_STRENGTH` — сила вариативности генерации.
- `REPETITION_PENALTY_STRENGTH` — насколько сильно подавлять повторы токенов и n-грамм в одном ответе.
- `MARKOV_ORDER` — основной порядок модели: `2` или `3`.
- `ENABLE_BACKOFF` — разрешить откат на более низкий порядок.
- `BACKOFF_MIN_ORDER` — минимальный порядок такого отката.

Ключевые настройки reply-контекста:
- `USE_REPLY_CONTEXT` — включить reply-aware генерацию.
- `REPLY_CONTEXT_MAX_TOKENS` — максимум токенов из reply-контекста.
- `REPLY_CONTEXT_LAST_TOKENS` — сколько последних токенов использовать как приоритетный seed.
- `REPLY_CONTEXT_BIAS` — насколько усиливать токены из контекста во время генерации.
- `REPLY_CONTEXT_START_BIAS` — насколько усиливать стартовые цепочки, найденные в контексте.
- `REPLY_CONTEXT_ONLY_FOR_REPLIES` — применять контекст только для Telegram reply.
- `REPLY_CONTEXT_INCLUDE_CURRENT_MESSAGE` — добавлять токены текущего сообщения к reply-контексту.

## Как бот работает
Обучение:
1. Сохраняет сырое сообщение в `messages`.
2. Очищает текст: удаляет ссылки, `@mentions`, нормализует повторы и пробелы.
3. Токенизирует текст в слова и знаки пунктуации.
4. Обновляет статистику переходов в `starts3`, `transitions3`, `starts`, `transitions`, `transitions1`.

Генерация:
1. Выбирает стартовую цепочку из модели или из заданного seed.
2. Если включён `USE_REPLY_CONTEXT`, пытается использовать токены `reply_to_message` как контекстный старт и bias.
3. На каждом шаге сначала пробует переходы высокого порядка, затем fallback на низшие.
4. Отбрасывает слишком короткие и дословно повторяющие историю ответы.

Когда бот отвечает:
- при mention или reply на бота пытается ответить обязательно;
- без прямого обращения отвечает по вероятности `REPLY_PROBABILITY`;
- если данных мало, сообщает, что модели нужно больше материала.

Подробности внутренней архитектуры вынесены в [docs/ARCHITECTURE.md](/D:/test/PepeEdtaBot/docs/ARCHITECTURE.md).

## Команды
- `/help` — список команд.
- `/ping` — проверка, что бот онлайн.
- `/stats` — статистика модели по текущему чату.
- `/config` — текущие runtime-настройки процесса.
- `/set <key> <value>` — изменить runtime-настройку, доступно `OWNER_ID` или админам чата.
- `/setprob 0.2` — быстрый setter вероятности ответа.
- `/seed "текст"` — одноразовый старт генерации.
- `/clear` — очистить данные текущего чата.

Через `/set` можно менять и reply-context параметры:
- `use_reply_context`
- `reply_context_max_tokens`
- `reply_context_last_tokens`
- `reply_context_bias`
- `reply_context_start_bias`
- `reply_context_only_for_replies`
- `reply_context_include_current_message`
- `repetition_penalty_strength`

## Совместимость БД
Текущая версия не требует новой схемы БД для reply-контекста.

Важно:
- существующие `markov.db` остаются совместимыми;
- по умолчанию база хранится в `data/markov.db`;
- новые настройки живут в `.env` и runtime state;
- reply-context влияет только на логику генерации, а не на структуру SQLite.

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
