from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Optional

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.types import BotCommand, Message
from aiogram.enums import ChatAction

from bot_messages import (
    TELEGRAM_COMMANDS,
    format_clear_confirmation_message,
    format_config_message,
    format_help_message,
    format_set_help_message,
    format_stats_message,
)
from bot_policy import (
    bot_is_mentioned,
    cooldown_allows_reply,
    has_enough_model_data,
    should_reply_to_message,
)
from db import Database
from markov import MarkovGenerator, tokenize
from runtime_config import (
    InvalidRuntimeSettingValueError,
    UNKNOWN_RUNTIME_KEY_MESSAGE,
    UnknownRuntimeSettingError,
    apply_runtime_setting,
)
from runtime_state import runtime_state_from_settings
from settings import Settings, load_settings
from text_utils import sanitize_text


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


def extract_context_tokens(
    message: Message,
    current_text: str,
    normalize_lower: bool,
    max_tokens: int,
    only_for_replies: bool,
    include_current_message: bool,
) -> list[str]:
    if only_for_replies and message.reply_to_message is None:
        return []

    context_parts: list[str] = []
    if message.reply_to_message and message.reply_to_message.text:
        context_parts.append(message.reply_to_message.text)
    if include_current_message and current_text:
        context_parts.append(current_text)

    if not context_parts:
        return []

    clean = sanitize_text(" ".join(context_parts))
    if not clean:
        return []

    tokens = tokenize(clean, normalize_lower=normalize_lower)
    return tokens[-max_tokens:] if len(tokens) > max_tokens else tokens


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
    logging.getLogger("aiogram").setLevel(logging.WARNING)

    db = Database(settings.db_path)
    await db.init()
    generator = MarkovGenerator(db=db)
    state = runtime_state_from_settings(settings)

    bot = Bot(token=settings.bot_token)
    me = await bot.get_me()
    bot_username = (me.username or "").lower()
    await bot.delete_webhook(drop_pending_updates=False)
    await bot.set_my_commands(
        [BotCommand(command=command, description=description) for command, description in TELEGRAM_COMMANDS]
    )

    dp = Dispatcher()

    @dp.message(Command("stats"))
    async def cmd_stats(message: Message) -> None:
        if not is_group_message(message):
            return
        stats = await db.get_stats(message.chat.id)
        text = format_stats_message(stats, state.min_tokens_for_model)
        await reply_humanized(message, text, state.typing_min_ms, state.typing_max_ms)

    @dp.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        await reply_humanized(
            message, format_help_message(), state.typing_min_ms, state.typing_max_ms
        )

    @dp.message(Command("ping"))
    async def cmd_ping(message: Message) -> None:
        await message.reply("pong")

    @dp.message(Command("config"))
    async def cmd_config(message: Message) -> None:
        raw = extract_command_arg(message.text or "")
        text = format_config_message(state, full=raw.strip().lower() == "full")
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
        if raw.strip().lower() == "help":
            await reply_humanized(
                message,
                format_set_help_message(),
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return
        if not raw:
            await reply_humanized(
                message,
                "Использование: /set <key> <value>\nПодсказка: /set help",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return
        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            await reply_humanized(
                message,
                "Использование: /set <key> <value>\nПодсказка: /set help",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return

        key, value = parts[0].strip().lower(), parts[1].strip()
        try:
            apply_runtime_setting(state, key, value)
        except UnknownRuntimeSettingError:
            await reply_humanized(
                message,
                f"{UNKNOWN_RUNTIME_KEY_MESSAGE}\n\nПодсказка: /set help",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            return
        except InvalidRuntimeSettingValueError:
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
        raw = extract_command_arg(message.text or "")
        if raw.strip().lower() != "confirm":
            await reply_humanized(
                message,
                format_clear_confirmation_message(),
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
            logger.debug("Skip message by length: chat=%s len=%s", message.chat.id, len(clean))
            return

        tokens = tokenize(clean, normalize_lower=state.normalize_lower)
        token_volume = await db.save_message_and_update_model(
            chat_id=message.chat.id,
            author_id=message.from_user.id,
            raw_text=raw_text,
            tokens=tokens,
        )
        learned = state.learned_messages.get(message.chat.id, 0) + 1
        state.learned_messages[message.chat.id] = learned
        if learned == 1 or learned % 25 == 0:
            logger.info(
                "Прогресс обучения: chat_id=%s, сообщений=%s, объём_модели=%s",
                message.chat.id,
                learned,
                token_volume,
            )
        generator.invalidate_chat_cache(message.chat.id)

        now = time.time()
        mentioned = bot_is_mentioned(message, bot_username, me.id)

        enough_data = has_enough_model_data(token_volume, state.min_tokens_for_model)

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
        cooldown_ok = cooldown_allows_reply(now, last_ts, state.min_cooldown_sec)
        should_reply = should_reply_to_message(
            mentioned=mentioned,
            cooldown_ok=cooldown_ok,
            reply_probability=state.reply_probability,
            random_value=random.random(),
        )

        if not should_reply:
            logger.debug(
                "Skip by trigger/cooldown: chat=%s mentioned=%s cooldown_ok=%s prob=%.2f",
                message.chat.id,
                mentioned,
                cooldown_ok,
                state.reply_probability,
            )
            return

        context_tokens: list[str] = []
        if state.use_reply_context:
            context_tokens = extract_context_tokens(
                message=message,
                current_text=raw_text,
                normalize_lower=state.normalize_lower,
                max_tokens=state.reply_context_max_tokens,
                only_for_replies=state.reply_context_only_for_replies,
                include_current_message=state.reply_context_include_current_message,
            )

        seed = None
        if seed is None and context_tokens:
            seed = context_tokens[-state.reply_context_last_tokens :]
            logger.debug(
                "Reply context prepared: chat=%s context_tokens=%s seed=%s",
                message.chat.id,
                len(context_tokens),
                seed,
            )
        reply_text = ""
        # Повторяем генерацию несколько раз, чтобы не уходить в "молчание" на разреженной модели.
        for attempt in range(4):
            attempt_context_tokens = context_tokens if attempt < 2 else None
            reply_text = await generator.generate_text(
                chat_id=message.chat.id,
                max_chars=state.max_reply_chars,
                max_tokens=state.max_reply_tokens,
                seed_tokens=seed,
                context_tokens=attempt_context_tokens,
                context_bias=state.reply_context_bias,
                context_start_bias=state.reply_context_start_bias,
                randomness_strength=state.randomness_strength,
                repetition_penalty_strength=state.repetition_penalty_strength,
                markov_order=state.markov_order,
                enable_backoff=state.enable_backoff,
                backoff_min_order=state.backoff_min_order,
            )
            if reply_text:
                break
            logger.debug(
                "Generation attempt failed: chat=%s attempt=%s context=%s seed_len=%s",
                message.chat.id,
                attempt + 1,
                bool(attempt_context_tokens),
                len(seed or []),
            )
            if attempt == 0 and context_tokens:
                seed = context_tokens[-state.reply_context_last_tokens :]
            else:
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

    logger.info("Бот %s запущен (polling).", me.username)
    logger.info("Статус: работает.")
    try:
        await dp.start_polling(bot)
    finally:
        logger.info("Статус: остановка...")
        await db.close()
        await bot.session.close()
        logger.info("Статус: остановлен.")


if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except (KeyboardInterrupt, SystemExit):
        pass
