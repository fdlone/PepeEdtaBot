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
from app.filters import AdminOrOwner, GroupOnly, OwnerOnly
from app.handlers._helpers import reply_humanized_state
from app.infrastructure.database import Database
from app.presentation.bot_messages import (
    format_clear_confirmation_message,
    format_config_message,
    format_set_help_message,
)
from app.services import PivoService

router = Router(name="admin")
logger = logging.getLogger("chat_markov")


def _extract_command_arg(text: str) -> str:
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) >= 2 else ""


async def _reply_no_permission(
    message: Message, runtime_state: RuntimeState
) -> None:
    """Отказ для /set и /setprob (OWNER_ID-only, см. O5 в docs/OPEN.md)."""
    await reply_humanized_state(
        message, "Команда доступна только OWNER_ID.", runtime_state
    )


@router.message(Command("config"))
async def cmd_config(message: Message, runtime_state: RuntimeState) -> None:
    raw = _extract_command_arg(message.text or "")
    text = format_config_message(runtime_state, full=raw.strip().lower() == "full")
    await reply_humanized_state(message, text, runtime_state)


# O5 (docs/OPEN.md): RuntimeState is one instance per process, not per chat,
# so a knob change here is visible to every chat the bot is in. AdminOrOwner
# would let any chat's own admins reach that process-global effect; OWNER_ID
# is the interim fix. Proper fix is scoping RuntimeState overlays by chat_id
# (deferred — bigger refactor, see docs/OPEN.md backlog) after which this can
# go back to AdminOrOwner.
@router.message(Command("set"), GroupOnly(), OwnerOnly())
async def cmd_set(
    message: Message, runtime_state: RuntimeState, settings: Settings
) -> None:
    raw = _extract_command_arg(message.text or "")
    if raw.strip().lower() == "help":
        await reply_humanized_state(
            message, format_set_help_message(), runtime_state
        )
        return
    usage = "Использование: /set <key> <value>\nПодсказка: /set help"
    if not raw:
        await reply_humanized_state(message, usage, runtime_state)
        return
    parts = raw.split(maxsplit=1)
    if len(parts) != 2:
        await reply_humanized_state(message, usage, runtime_state)
        return

    key, value = parts[0].strip().lower(), parts[1].strip()
    try:
        apply_runtime_setting(runtime_state, key, value)
    except UnknownRuntimeSettingError:
        await reply_humanized_state(
            message,
            f"{UNKNOWN_RUNTIME_KEY_MESSAGE}\n\nПодсказка: /set help",
            runtime_state,
        )
        return
    except InvalidRuntimeSettingValueError:
        await reply_humanized_state(
            message, "Некорректное значение для этого ключа.", runtime_state
        )
        return

    await reply_humanized_state(
        message, f"Обновлено: {key}={value} (до перезапуска)", runtime_state
    )


@router.message(Command("set"), GroupOnly())
async def cmd_set_denied(message: Message, runtime_state: RuntimeState) -> None:
    """Fallback: вызывается, когда AdminOrOwner отказал в правах для /set."""
    await _reply_no_permission(message, runtime_state)


# Same O5 rationale as cmd_set above: reply_probability is process-global.
@router.message(Command("setprob"), GroupOnly(), OwnerOnly())
async def cmd_setprob(
    message: Message, runtime_state: RuntimeState, settings: Settings
) -> None:
    raw = _extract_command_arg(message.text or "")
    if not raw:
        await reply_humanized_state(
            message, "Использование: /setprob 0.2", runtime_state
        )
        return
    try:
        value = float(raw)
    except ValueError:
        await reply_humanized_state(
            message, "Нужно число в диапазоне 0..1", runtime_state
        )
        return

    if not 0.0 <= value <= 1.0:
        await reply_humanized_state(
            message, "Значение должно быть в диапазоне 0..1", runtime_state
        )
        return

    runtime_state.reply_probability = value
    await reply_humanized_state(
        message, f"REPLY_PROBABILITY теперь: {value}", runtime_state
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
    pivo_service: PivoService,
) -> None:
    raw = _extract_command_arg(message.text or "")
    if raw.strip().lower() != "confirm":
        await reply_humanized_state(
            message, format_clear_confirmation_message(), runtime_state
        )
        return
    await db.clear_chat(message.chat.id)
    await pivo_service.clear_chat_data(message.chat.id)
    generator.invalidate_chat_cache(message.chat.id)
    runtime_state.forget_chat(message.chat.id)
    await reply_humanized_state(
        message, "Данные чата очищены (включая подписки /pivo).", runtime_state
    )


@router.message(Command("clear"), GroupOnly())
async def cmd_clear_denied(message: Message, runtime_state: RuntimeState) -> None:
    """Fallback: вызывается, когда AdminOrOwner отказал в правах для /clear."""
    await reply_humanized_state(
        message,
        "Недостаточно прав. Нужен OWNER_ID или права админа чата.",
        runtime_state,
    )
