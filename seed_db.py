"""
Seed script: inserts random Markov training data via save_message_and_update_model.
Generates sentences from a fixed vocabulary so transitions accumulate realistic weights.

Usage:
    python seed_db.py [--chat-id CHAT_ID] [--messages N] [--db PATH]

Defaults:
    --chat-id  -1001147461458  (existing chat in markov.db)
    --messages 120             (~1000 tokens total)
    --db       markov.db
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiosqlite

from app.infrastructure import migrator
from db import Database

VOCAB: list[str] = [
    "привет", "пока", "да", "нет", "ну", "вот", "тут", "там", "так", "ещё",
    "уже", "всё", "это", "он", "она", "они", "мы", "я", "ты", "вы",
    "хорошо", "плохо", "можно", "нельзя", "надо", "просто", "очень", "немного", "всегда", "иногда",
    "сегодня", "завтра", "вчера", "утром", "вечером", "ночью", "быстро", "тихо", "громко", "точно",
    "думаю", "знаю", "хочу", "могу", "буду", "пойду", "сказал", "видел", "слышал", "понял",
    "работа", "дом", "время", "день", "ночь", "место", "люди", "человек", "слово", "вопрос",
    "хватит", "стоп", "давай", "окей", "понял", "ладно", "сделал", "получилось", "нормально", "отлично",
    "пиво", "чай", "кофе", "еда", "вода", "стол", "стул", "экран", "код", "баг",
]

# Weighted starts to create skewed cnt distribution on common openers
SENTENCE_STARTERS: list[str] = [
    "ну вот так", "да нет же", "я думаю что", "ну и ладно", "хорошо окей",
    "да я знаю", "просто так", "ну давай", "я хочу", "вот именно",
]


def random_sentence(min_len: int = 4, max_len: int = 14) -> list[str]:
    length = random.randint(min_len, max_len)
    if random.random() < 0.35:
        starter = random.choice(SENTENCE_STARTERS).split()
        tail_len = max(0, length - len(starter))
        return starter + random.choices(VOCAB, k=tail_len)
    return random.choices(VOCAB, k=length)


async def run(db_path: str, chat_id: int, num_messages: int) -> None:
    database = Database(db_path)
    await database.init()

    token_total = 0
    for i in range(num_messages):
        tokens = random_sentence()
        raw_text = " ".join(tokens)
        volume = await database.save_message_and_update_model(chat_id, raw_text, tokens)
        token_total += len(tokens)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{num_messages} messages — model volume: {volume}")

    print(f"\nDone. Inserted {num_messages} messages (~{token_total} tokens) into chat {chat_id}.")
    print(f"DB: {os.path.abspath(db_path)}")

    await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Markov DB with random tokens")
    parser.add_argument("--chat-id", type=int, default=-1001147461458)
    parser.add_argument("--messages", type=int, default=120)
    parser.add_argument("--db", type=str, default="markov.db")
    args = parser.parse_args()

    print(f"Seeding chat_id={args.chat_id}, messages={args.messages}, db={args.db}")
    asyncio.run(run(args.db, args.chat_id, args.messages))


if __name__ == "__main__":
    main()
