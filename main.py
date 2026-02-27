from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.types import Message

from db import Database
from markov import MarkovGenerator, tokenize
from settings import Settings, load_settings
from text_utils import sanitize_text


@dataclass(slots=True)
class RuntimeState:
    reply_probability: float
    last_reply_ts: dict[int, float] = field(default_factory=dict)
    pending_seed: dict[int, list[str]] = field(default_factory=dict)


def is_group_message(message: Message) -> bool:
    return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}


def is_owner(message: Message, owner_id: Optional[int]) -> bool:
    return owner_id is not None and message.from_user is not None and message.from_user.id == owner_id


def extract_command_arg(text: str) -> str:
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def bot_is_mentioned(message: Message, bot_username: str) -> bool:
    if not message.text:
        return False

    username_mention = f"@{bot_username}".lower()
    if username_mention in message.text.lower():
        return True

    if message.entities:
        for ent in message.entities:
            if ent.type == "mention":
                mention = message.text[ent.offset : ent.offset + ent.length]
                if mention.lower() == username_mention:
                    return True
    return False


async def run_bot() -> None:
    settings: Settings = load_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("chat_markov")

    db = Database(settings.db_path)
    await db.init()
    generator = MarkovGenerator(db=db)
    state = RuntimeState(reply_probability=settings.reply_probability)

    bot = Bot(token=settings.bot_token)
    me = await bot.get_me()
    bot_username = (me.username or "").lower()

    dp = Dispatcher()

    @dp.message(Command("stats"))
    async def cmd_stats(message: Message) -> None:
        if not is_group_message(message):
            return
        stats = await db.get_stats(message.chat.id)
        text = (
            "Статистика модели:\n"
            f"messages: {stats['messages']}\n"
            f"start pairs: {stats['starts']}\n"
            f"transition edges: {stats['transitions']}\n"
            f"volume(sum cnt): {stats['volume']}"
        )
        await message.reply(text)

    @dp.message(Command("clear"))
    async def cmd_clear(message: Message) -> None:
        if not is_group_message(message):
            return
        if not is_owner(message, settings.owner_id):
            await message.reply("Недостаточно прав.")
            return
        await db.clear_chat(message.chat.id)
        generator.invalidate_chat_cache(message.chat.id)
        await message.reply("Данные чата очищены.")

    @dp.message(Command("setprob"))
    async def cmd_setprob(message: Message) -> None:
        if not is_owner(message, settings.owner_id):
            await message.reply("Команда только для OWNER_ID.")
            return

        raw = extract_command_arg(message.text or "")
        if not raw:
            await message.reply("Использование: /setprob 0.2")
            return
        try:
            value = float(raw)
        except ValueError:
            await message.reply("Нужно число в диапазоне 0..1")
            return

        if not 0.0 <= value <= 1.0:
            await message.reply("Значение должно быть в диапазоне 0..1")
            return

        state.reply_probability = value
        await message.reply(f"REPLY_PROBABILITY теперь: {value}")

    @dp.message(Command("seed"))
    async def cmd_seed(message: Message) -> None:
        if not is_owner(message, settings.owner_id):
            await message.reply("Команда только для OWNER_ID.")
            return
        raw = extract_command_arg(message.text or "")
        if not raw:
            await message.reply('Использование: /seed "ваш текст"')
            return

        clean = sanitize_text(raw)
        tokens = tokenize(clean, normalize_lower=settings.normalize_lower)
        if not tokens:
            await message.reply("Не удалось извлечь токены из seed.")
            return

        state.pending_seed[message.chat.id] = tokens[:2]
        await message.reply("Seed сохранен для следующей генерации (одноразово).")

    @dp.message(F.text)
    async def on_text_message(message: Message) -> None:
        if not is_group_message(message):
            return
        if message.from_user is None:
            return
        if message.from_user.is_bot:
            return

        raw_text = message.text or ""
        if raw_text.startswith("/"):
            return

        clean = sanitize_text(raw_text)
        if len(clean) < 3 or len(clean) > 500:
            return

        tokens = tokenize(clean, normalize_lower=settings.normalize_lower)

        await db.save_message(message.chat.id, message.from_user.id, clean)
        await db.update_model(message.chat.id, tokens)
        generator.invalidate_chat_cache(message.chat.id)

        now = time.time()
        mentioned = bot_is_mentioned(message, bot_username)

        enough_data = (
            await db.get_chat_token_volume(message.chat.id)
        ) >= settings.min_tokens_for_model

        if mentioned and not enough_data:
            await message.reply("Пока мало материала, поболтайте ещё 🙂")
            return

        if not enough_data:
            return

        last_ts = state.last_reply_ts.get(message.chat.id, 0.0)
        cooldown_ok = now - last_ts >= settings.min_cooldown_sec

        should_reply = False
        if mentioned and cooldown_ok:
            should_reply = True
        elif cooldown_ok and random.random() < state.reply_probability:
            should_reply = True

        if not should_reply:
            return

        seed = state.pending_seed.pop(message.chat.id, None)
        reply_text = await generator.generate_text(
            chat_id=message.chat.id,
            max_chars=settings.max_reply_chars,
            seed_tokens=seed,
        )
        if not reply_text:
            return

        state.last_reply_ts[message.chat.id] = now
        await message.reply(reply_text)

    logger.info("Bot %s started in polling mode", me.username)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except (KeyboardInterrupt, SystemExit):
        pass
