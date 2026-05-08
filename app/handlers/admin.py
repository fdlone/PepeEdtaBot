from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.filters import AdminOrOwner, GroupOnly
from app.handlers._helpers import reply_humanized
from bot_messages import (
    format_clear_confirmation_message,
    format_config_message,
    format_set_help_message,
)
from db import Database
from markov import MarkovGenerator
from runtime_config import (
    InvalidRuntimeSettingValueError,
    UNKNOWN_RUNTIME_KEY_MESSAGE,
    UnknownRuntimeSettingError,
    apply_runtime_setting,
)
from runtime_state import RuntimeState
from settings import Settings


router = Router(name="admin")
logger = logging.getLogger("chat_markov")


def _extract_command_arg(text: str) -> str:
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) >= 2 else ""


@router.message(Command("config"))
async def cmd_config(message: Message, state: RuntimeState) -> None:
    raw = _extract_command_arg(message.text or "")
    text = format_config_message(state, full=raw.strip().lower() == "full")
    await reply_humanized(message, text, state.typing_min_ms, state.typing_max_ms)


@router.message(Command("set"), GroupOnly(), AdminOrOwner())
async def cmd_set(message: Message, state: RuntimeState, settings: Settings) -> None:
    raw = _extract_command_arg(message.text or "")
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


@router.message(Command("setprob"), GroupOnly(), AdminOrOwner())
async def cmd_setprob(message: Message, state: RuntimeState, settings: Settings) -> None:
    raw = _extract_command_arg(message.text or "")
    if not raw:
        await reply_humanized(
            message,
            "Использование: /setprob 0.2",
            state.typing_min_ms,
            state.typing_max_ms,
        )
        return
    try:
        value = float(raw)
    except ValueError:
        await reply_humanized(
            message,
            "Нужно число в диапазоне 0..1",
            state.typing_min_ms,
            state.typing_max_ms,
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
        message,
        f"REPLY_PROBABILITY теперь: {value}",
        state.typing_min_ms,
        state.typing_max_ms,
    )


@router.message(Command("clear"), GroupOnly(), AdminOrOwner())
async def cmd_clear(
    message: Message,
    db: Database,
    state: RuntimeState,
    generator: MarkovGenerator,
    settings: Settings,
) -> None:
    raw = _extract_command_arg(message.text or "")
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
    await reply_humanized(
        message, "Данные чата очищены.", state.typing_min_ms, state.typing_max_ms
    )
