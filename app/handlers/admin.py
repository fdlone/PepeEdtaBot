from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.config.runtime_config import (
    UNKNOWN_RUNTIME_KEY_MESSAGE,
    InvalidRuntimeSettingValueError,
    UnknownRuntimeSettingError,
    apply_runtime_setting,
)
from app.config.runtime_state import RuntimeState
from app.config.settings import Settings
from app.core.markov import MarkovGenerator
from app.filters import AdminOrOwner, GroupOnly
from app.handlers._helpers import reply_humanized
from app.infrastructure.database import Database
from app.presentation.bot_messages import (
    format_clear_confirmation_message,
    format_config_message,
    format_set_help_message,
)

router = Router(name="admin")
logger = logging.getLogger("chat_markov")


def _extract_command_arg(text: str) -> str:
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) >= 2 else ""


async def _reply_no_permission(
    message: Message, runtime_state: RuntimeState
) -> None:
    """Отказ для команд, требующих OWNER_ID или прав админа чата."""
    await reply_humanized(
        message,
        "Команда доступна OWNER_ID и администраторам чата.",
        runtime_state.typing_min_ms,
        runtime_state.typing_max_ms,
    )


@router.message(Command("config"))
async def cmd_config(message: Message, runtime_state: RuntimeState) -> None:
    raw = _extract_command_arg(message.text or "")
    text = format_config_message(runtime_state, full=raw.strip().lower() == "full")
    await reply_humanized(
        message,
        text,
        runtime_state.typing_min_ms,
        runtime_state.typing_max_ms,
    )


@router.message(Command("set"), GroupOnly(), AdminOrOwner())
async def cmd_set(
    message: Message, runtime_state: RuntimeState, settings: Settings
) -> None:
    raw = _extract_command_arg(message.text or "")
    if raw.strip().lower() == "help":
        await reply_humanized(
            message,
            format_set_help_message(),
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
        )
        return
    if not raw:
        await reply_humanized(
            message,
            "Использование: /set <key> <value>\nПодсказка: /set help",
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
        )
        return
    parts = raw.split(maxsplit=1)
    if len(parts) != 2:
        await reply_humanized(
            message,
            "Использование: /set <key> <value>\nПодсказка: /set help",
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
        )
        return

    key, value = parts[0].strip().lower(), parts[1].strip()
    try:
        apply_runtime_setting(runtime_state, key, value)
    except UnknownRuntimeSettingError:
        await reply_humanized(
            message,
            f"{UNKNOWN_RUNTIME_KEY_MESSAGE}\n\nПодсказка: /set help",
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
        )
        return
    except InvalidRuntimeSettingValueError:
        await reply_humanized(
            message,
            "Некорректное значение для этого ключа.",
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
        )
        return

    await reply_humanized(
        message,
        f"Обновлено: {key}={value} (до перезапуска)",
        runtime_state.typing_min_ms,
        runtime_state.typing_max_ms,
    )


@router.message(Command("set"), GroupOnly())
async def cmd_set_denied(message: Message, runtime_state: RuntimeState) -> None:
    """Fallback: вызывается, когда AdminOrOwner отказал в правах для /set."""
    await _reply_no_permission(message, runtime_state)


@router.message(Command("setprob"), GroupOnly(), AdminOrOwner())
async def cmd_setprob(
    message: Message, runtime_state: RuntimeState, settings: Settings
) -> None:
    raw = _extract_command_arg(message.text or "")
    if not raw:
        await reply_humanized(
            message,
            "Использование: /setprob 0.2",
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
        )
        return
    try:
        value = float(raw)
    except ValueError:
        await reply_humanized(
            message,
            "Нужно число в диапазоне 0..1",
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
        )
        return

    if not 0.0 <= value <= 1.0:
        await reply_humanized(
            message,
            "Значение должно быть в диапазоне 0..1",
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
        )
        return

    runtime_state.reply_probability = value
    await reply_humanized(
        message,
        f"REPLY_PROBABILITY теперь: {value}",
        runtime_state.typing_min_ms,
        runtime_state.typing_max_ms,
    )


@router.message(Command("setprob"), GroupOnly())
async def cmd_setprob_denied(
    message: Message, runtime_state: RuntimeState
) -> None:
    """Fallback: вызывается, когда AdminOrOwner отказал в правах для /setprob."""
    await _reply_no_permission(message, runtime_state)


@router.message(Command("clear"), GroupOnly(), AdminOrOwner())
async def cmd_clear(
    message: Message,
    db: Database,
    runtime_state: RuntimeState,
    generator: MarkovGenerator,
    settings: Settings,
) -> None:
    raw = _extract_command_arg(message.text or "")
    if raw.strip().lower() != "confirm":
        await reply_humanized(
            message,
            format_clear_confirmation_message(),
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
        )
        return
    await db.clear_chat(message.chat.id)
    generator.invalidate_chat_cache(message.chat.id)
    runtime_state.forget_chat(message.chat.id)
    await reply_humanized(
        message,
        "Данные чата очищены.",
        runtime_state.typing_min_ms,
        runtime_state.typing_max_ms,
    )


@router.message(Command("clear"), GroupOnly())
async def cmd_clear_denied(message: Message, runtime_state: RuntimeState) -> None:
    """Fallback: вызывается, когда AdminOrOwner отказал в правах для /clear."""
    await reply_humanized(
        message,
        "Недостаточно прав. Нужен OWNER_ID или права админа чата.",
        runtime_state.typing_min_ms,
        runtime_state.typing_max_ms,
    )
