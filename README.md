# PepeEdtaBot

Telegram-бот для группового чата с генерацией сообщений на базе простой Markov chain (биграммы, n=2) без ML.

## Stack
- Python 3.11+
- aiogram v3
- SQLite (aiosqlite)
- Конфиг через `.env`

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Создайте `.env` на основе `.env.example` и заполните `BOT_TOKEN`.

## Run
```bash
python main.py
```

## Команды
- `/stats` - статистика модели по текущему чату
- `/clear` - очистка данных текущего чата (только OWNER_ID)
- `/setprob 0.2` - изменить вероятность ответа (только OWNER_ID)
- `/seed "текст"` - одноразовый seed для следующей генерации (только OWNER_ID)

## Важно для групп
Отключите privacy mode у бота в BotFather: `Bot Settings` -> `Group Privacy` -> `Turn off`.
