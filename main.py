from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatType
from aiogram.enums import MessageEntityType
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ChatAction

from db import Database
from markov import MarkovGenerator, tokenize
from settings import Settings, load_settings
from text_utils import sanitize_text


@dataclass(slots=True)
class RuntimeState:
    reply_probability: float
    min_cooldown_sec: int
    min_tokens_for_model: int
    max_reply_chars: int
    normalize_lower: bool
    typing_min_ms: int
    typing_max_ms: int
    randomness_strength: float
    markov_order: int
    enable_backoff: bool
    backoff_min_order: int
    last_reply_ts: dict[int, float] = field(default_factory=dict)
    pending_seed: dict[int, list[str]] = field(default_factory=dict)


def is_group_message(message: Message) -> bool:
    return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}


def is_owner(message: Message, owner_id: Optional[int]) -> bool:
    return owner_id is not None and message.from_user is not None and message.from_user.id == owner_id


async def is_chat_admin(bot: Bot, chat_id: int, user_id: int) -> bool:
    admins = await bot.get_chat_administrators(chat_id)
    for admin in admins:
        if admin.user.id == user_id:
            return True
    return False


async def can_manage_settings(
    message: Message, bot: Bot, owner_id: Optional[int], logger: logging.Logger
) -> bool:
    if is_owner(message, owner_id):
        return True
    if message.from_user is None:
        return False
    try:
        return await is_chat_admin(bot, message.chat.id, message.from_user.id)
    except Exception as exc:
        logger.warning("Cannot verify chat admins for settings command: %s", exc)
        return False


def extract_command_arg(text: str) -> str:
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def parse_bool(value: str) -> Optional[bool]:
    val = value.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return None


def bot_is_mentioned(message: Message, bot_username: str, bot_id: int) -> bool:
    if not message.text:
        return (
            message.reply_to_message is not None
            and message.reply_to_message.from_user is not None
            and message.reply_to_message.from_user.id == bot_id
        )

    username_mention = f"@{bot_username}".lower()
    if username_mention in message.text.lower():
        return True

    if message.entities:
        for ent in message.entities:
            if ent.type == MessageEntityType.MENTION:
                mention = message.text[ent.offset : ent.offset + ent.length]
                if mention.lower() == username_mention:
                    return True
    if message.reply_to_message and message.reply_to_message.from_user:
        if message.reply_to_message.from_user.id == bot_id:
            return True
    return False


async def reply_humanized(
    message: Message, text: str, typing_min_ms: int, typing_max_ms: int
) -> None:
    try:
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
        delay_ms = random.randint(typing_min_ms, typing_max_ms)
        await asyncio.sleep(delay_ms / 1000)
    except Exception:
        # Ошибка chat action не должна блокировать отправку обычного ответа.
        pass
    await message.reply(text)


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
    state = RuntimeState(
        reply_probability=settings.reply_probability,
        min_cooldown_sec=settings.min_cooldown_sec,
        min_tokens_for_model=settings.min_tokens_for_model,
        max_reply_chars=settings.max_reply_chars,
        normalize_lower=settings.normalize_lower,
        typing_min_ms=settings.typing_min_ms,
        typing_max_ms=settings.typing_max_ms,
        randomness_strength=settings.randomness_strength,
        markov_order=settings.markov_order,
        enable_backoff=settings.enable_backoff,
        backoff_min_order=settings.backoff_min_order,
    )

    bot = Bot(token=settings.bot_token)
    me = await bot.get_me()
    bot_username = (me.username or "").lower()
    await bot.delete_webhook(drop_pending_updates=False)

    dp = Dispatcher()

    @dp.message(Command("stats"))
    async def cmd_stats(message: Message) -> None:
        if not is_group_message(message):
            return
        stats = await db.get_stats(message.chat.id)
        text = (
            "Статистика модели:\n"
            f"messages: {stats['messages']}\n"
            f"starts2: {stats['starts2']} | starts3: {stats['starts3']}\n"
            f"edges2: {stats['transitions2']} | edges3: {stats['transitions3']} | edges1: {stats['transitions1']}\n"
            f"volume2: {stats['volume2']} | volume3: {stats['volume3']} | volume1: {stats['volume1']}\n"
            f"effective_volume: {stats['volume']}"
        )
        await reply_humanized(message, text, state.typing_min_ms, state.typing_max_ms)

    @dp.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        text = (
            "Доступные команды:\n"
            "/help - показать эту справку\n"
            "/ping - проверка, что бот онлайн и видит чат в реальном времени\n"
            "/config - показать текущие runtime-настройки\n"
            "/set <key> <value> - изменить runtime-настройку (OWNER_ID или админ)\n"
            "/stats - статистика модели по текущему чату\n"
            "/clear - очистить данные чата (OWNER_ID или админ чата)\n"
            "/setprob 0.2 - изменить вероятность ответов (OWNER_ID или админ)\n"
            '/seed "текст" - одноразово задать старт генерации (OWNER_ID или админ)'
        )
        await reply_humanized(message, text, state.typing_min_ms, state.typing_max_ms)

    @dp.message(Command("ping"))
    async def cmd_ping(message: Message) -> None:
        await message.reply("pong")

    @dp.message(Command("config"))
    async def cmd_config(message: Message) -> None:
        text = (
            "Текущие runtime-настройки:\n"
            f"reply_probability={state.reply_probability}\n"
            f"min_cooldown_sec={state.min_cooldown_sec}\n"
            f"min_tokens_for_model={state.min_tokens_for_model}\n"
            f"max_reply_chars={state.max_reply_chars}\n"
            f"normalize_lower={state.normalize_lower}\n"
            f"typing_min_ms={state.typing_min_ms}\n"
            f"typing_max_ms={state.typing_max_ms}\n"
            f"randomness_strength={state.randomness_strength}\n"
            f"markov_order={state.markov_order}\n"
            f"enable_backoff={state.enable_backoff}\n"
            f"backoff_min_order={state.backoff_min_order}\n"
            "Изменения через /set действуют до перезапуска."
        )
        await reply_humanized(message, text, state.typing_min_ms, state.typing_max_ms)

    @dp.message(Command("set"))
    async def cmd_set(message: Message) -> None:
        if not await can_manage_settings(message, bot, settings.owner_id, logger):
            await reply_humanized(
                message,
                "Команда доступна OWNER_ID и администраторам чата.",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return
        raw = extract_command_arg(message.text or "")
        if not raw:
            await reply_humanized(
                message,
                "Использование: /set <key> <value>",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            await reply_humanized(
                message,
                "Использование: /set <key> <value>",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return

        key, value = parts[0].strip().lower(), parts[1].strip()
        try:
            if key == "reply_probability":
                v = float(value)
                if not 0.0 <= v <= 1.0:
                    raise ValueError
                state.reply_probability = v
            elif key == "min_cooldown_sec":
                v = int(value)
                if v < 0:
                    raise ValueError
                state.min_cooldown_sec = v
            elif key == "min_tokens_for_model":
                v = int(value)
                if v < 0:
                    raise ValueError
                state.min_tokens_for_model = v
            elif key == "max_reply_chars":
                v = int(value)
                if v < 20 or v > 4000:
                    raise ValueError
                state.max_reply_chars = v
            elif key == "normalize_lower":
                v_bool = parse_bool(value)
                if v_bool is None:
                    raise ValueError
                state.normalize_lower = v_bool
            elif key == "typing_min_ms":
                v = int(value)
                if v < 0 or v > state.typing_max_ms:
                    raise ValueError
                state.typing_min_ms = v
            elif key == "typing_max_ms":
                v = int(value)
                if v < state.typing_min_ms:
                    raise ValueError
                state.typing_max_ms = v
            elif key == "randomness_strength":
                v = float(value)
                if not 0.0 <= v <= 3.0:
                    raise ValueError
                state.randomness_strength = v
            elif key == "markov_order":
                v = int(value)
                if v not in {2, 3}:
                    raise ValueError
                if state.backoff_min_order >= v:
                    raise ValueError
                state.markov_order = v
            elif key == "enable_backoff":
                v_bool = parse_bool(value)
                if v_bool is None:
                    raise ValueError
                state.enable_backoff = v_bool
            elif key == "backoff_min_order":
                v = int(value)
                if v not in {1, 2} or v >= state.markov_order:
                    raise ValueError
                state.backoff_min_order = v
            else:
                await reply_humanized(
                    message,
                    (
                        "Неизвестный ключ.\n"
                        "Доступно: reply_probability, min_cooldown_sec, "
                        "min_tokens_for_model, max_reply_chars, normalize_lower, "
                        "typing_min_ms, typing_max_ms, randomness_strength, "
                        "markov_order, enable_backoff, backoff_min_order"
                    ),
                    state.typing_min_ms,
                    state.typing_max_ms,
                )
                return
        except ValueError:
            await reply_humanized(
                message,
                "Некорректное значение для этого ключа.",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return

        await reply_humanized(
            message,
            f"Обновлено: {key}={value} (до перезапуска)",
            state.typing_min_ms,
            state.typing_max_ms,
        )

    @dp.message(Command("clear"))
    async def cmd_clear(message: Message) -> None:
        if not is_group_message(message):
            return
        allowed = is_owner(message, settings.owner_id)
        if not allowed and message.from_user is not None:
            try:
                allowed = await is_chat_admin(bot, message.chat.id, message.from_user.id)
            except Exception as exc:
                logger.warning("Cannot verify chat admins for /clear: %s", exc)
                allowed = False
        if not allowed:
            await reply_humanized(
                message,
                "Недостаточно прав. Нужен OWNER_ID или права админа чата.",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return
        await db.clear_chat(message.chat.id)
        generator.invalidate_chat_cache(message.chat.id)
        await reply_humanized(message, "Данные чата очищены.", state.typing_min_ms, state.typing_max_ms)

    @dp.message(Command("setprob"))
    async def cmd_setprob(message: Message) -> None:
        if not await can_manage_settings(message, bot, settings.owner_id, logger):
            await reply_humanized(
                message,
                "Команда доступна OWNER_ID и администраторам чата.",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return

        raw = extract_command_arg(message.text or "")
        if not raw:
            await reply_humanized(
                message, "Использование: /setprob 0.2", state.typing_min_ms, state.typing_max_ms
            )
            return
        try:
            value = float(raw)
        except ValueError:
            await reply_humanized(
                message, "Нужно число в диапазоне 0..1", state.typing_min_ms, state.typing_max_ms
            )
            return

        if not 0.0 <= value <= 1.0:
            await reply_humanized(
                message,
                "Значение должно быть в диапазоне 0..1",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return

        state.reply_probability = value
        await reply_humanized(
            message, f"REPLY_PROBABILITY теперь: {value}", state.typing_min_ms, state.typing_max_ms
        )

    @dp.message(Command("seed"))
    async def cmd_seed(message: Message) -> None:
        if not await can_manage_settings(message, bot, settings.owner_id, logger):
            await reply_humanized(
                message,
                "Команда доступна OWNER_ID и администраторам чата.",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return
        raw = extract_command_arg(message.text or "")
        if not raw:
            await reply_humanized(
                message, 'Использование: /seed "ваш текст"', state.typing_min_ms, state.typing_max_ms
            )
            return

        clean = sanitize_text(raw)
        tokens = tokenize(clean, normalize_lower=state.normalize_lower)
        if not tokens:
            await reply_humanized(
                message, "Не удалось извлечь токены из seed.", state.typing_min_ms, state.typing_max_ms
            )
            return

        state.pending_seed[message.chat.id] = tokens[:3]
        await reply_humanized(
            message,
            "Seed сохранен для следующей генерации (одноразово).",
            state.typing_min_ms,
            state.typing_max_ms,
        )

    @dp.message(F.text)
    async def on_text_message(message: Message) -> None:
        if not is_group_message(message):
            return
        if message.from_user is None:
            return
        if message.from_user.is_bot:
            return
        logger.info(
            "Message received chat=%s user=%s text_len=%s",
            message.chat.id,
            message.from_user.id,
            len(message.text or ""),
        )

        raw_text = message.text or ""
        if raw_text.startswith("/"):
            return

        clean = sanitize_text(raw_text)
        if len(clean) < 3 or len(clean) > 500:
            logger.debug("Skip message by length: chat=%s len=%s", message.chat.id, len(clean))
            return

        tokens = tokenize(clean, normalize_lower=state.normalize_lower)
        token_volume = await db.save_message_and_update_model(
            chat_id=message.chat.id,
            author_id=message.from_user.id,
            raw_text=raw_text,
            tokens=tokens,
        )
        generator.invalidate_chat_cache(message.chat.id)

        now = time.time()
        mentioned = bot_is_mentioned(message, bot_username, me.id)

        enough_data = token_volume >= state.min_tokens_for_model

        if mentioned and not enough_data:
            await reply_humanized(
                message, "Пока мало материала, поболтайте ещё 🙂", state.typing_min_ms, state.typing_max_ms
            )
            return

        if not enough_data:
            logger.debug(
                "Skip reply: not enough model data chat=%s volume=%s min=%s",
                message.chat.id,
                token_volume,
                state.min_tokens_for_model,
            )
            return

        last_ts = state.last_reply_ts.get(message.chat.id, 0.0)
        cooldown_ok = now - last_ts >= state.min_cooldown_sec

        should_reply = False
        if mentioned:
            should_reply = True
        elif cooldown_ok and random.random() < state.reply_probability:
            should_reply = True

        if not should_reply:
            logger.debug(
                "Skip by trigger/cooldown: chat=%s mentioned=%s cooldown_ok=%s prob=%.2f",
                message.chat.id,
                mentioned,
                cooldown_ok,
                state.reply_probability,
            )
            return

        seed = state.pending_seed.pop(message.chat.id, None)
        reply_text = ""
        # Повторяем генерацию несколько раз, чтобы не уходить в "молчание" на разреженной модели.
        for _ in range(4):
            reply_text = await generator.generate_text(
                chat_id=message.chat.id,
                max_chars=state.max_reply_chars,
                seed_tokens=seed,
                randomness_strength=state.randomness_strength,
                markov_order=state.markov_order,
                enable_backoff=state.enable_backoff,
                backoff_min_order=state.backoff_min_order,
            )
            if reply_text:
                break
            seed = None

        if not reply_text:
            if mentioned:
                await reply_humanized(
                    message,
                    "Собираю мысли... Напишите ещё пару сообщений 🙂",
                    state.typing_min_ms,
                    state.typing_max_ms,
                )
                state.last_reply_ts[message.chat.id] = now
            logger.debug("Generation failed: chat=%s mentioned=%s", message.chat.id, mentioned)
            return

        state.last_reply_ts[message.chat.id] = now
        await reply_humanized(message, reply_text, state.typing_min_ms, state.typing_max_ms)

    logger.info("Bot %s started in polling mode", me.username)
    try:
        await dp.start_polling(bot)
    finally:
        await db.close()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except (KeyboardInterrupt, SystemExit):
        pass
