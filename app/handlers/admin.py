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
    clear_runtime_setting,
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
async def cmd_config(
    message: Message,
    runtime_state: RuntimeState,
    runtime_state_base: RuntimeState,
) -> None:
    raw = _extract_command_arg(message.text or "")
    # runtime_state is already this chat's view, so the values shown are the
    # effective ones; the base state only says which of them were overridden.
    text = format_config_message(
        runtime_state,
        full=raw.strip().lower() == "full",
        overridden=runtime_state_base.chat_overrides.get(message.chat.id, {}),
    )
    await reply_humanized_state(message, text, runtime_state)


# O5 (docs/OPEN.md): RuntimeState is one instance per process. `/set` used to
# retune every chat at once; it now writes into this chat's overlay, so the
# owner can tune one chat without touching the rest. The explicit `global`
# form keeps the old reach, and OWNER_ID remains the gate either way.
@router.message(Command("set"), GroupOnly(), OwnerOnly())
async def cmd_set(
    message: Message,
    runtime_state: RuntimeState,
    settings: Settings,
    runtime_state_base: RuntimeState,
) -> None:
    raw = _extract_command_arg(message.text or "")
    if raw.strip().lower() == "help":
        await reply_humanized_state(
            message, format_set_help_message(), runtime_state
        )
        return
    usage = (
        "Использование: /set <key> <value>\n"
        "Глобально во всех чатах: /set global <key> <value>\n"
        "Вернуть чат к глобальному значению: /set reset <key>\n"
        "Подсказка: /set help"
    )
    if not raw:
        await reply_humanized_state(message, usage, runtime_state)
        return
    parts = raw.split(maxsplit=2)
    scope_word = parts[0].strip().lower()

    if scope_word == "reset":
        if len(parts) != 2:
            await reply_humanized_state(message, usage, runtime_state)
            return
        key = parts[1].strip().lower()
        try:
            cleared = clear_runtime_setting(runtime_state_base, message.chat.id, key)
        except UnknownRuntimeSettingError:
            await reply_humanized_state(
                message,
                f"{UNKNOWN_RUNTIME_KEY_MESSAGE}\n\nПодсказка: /set help",
                runtime_state,
            )
            return
        text = (
            f"{key}: чат вернулся к глобальному значению."
            if cleared
            else f"{key}: в этом чате и так глобальное значение."
        )
        await reply_humanized_state(message, text, runtime_state)
        return

    is_global = scope_word == "global"
    arguments = parts[1:] if is_global else parts
    if len(arguments) != 2:
        await reply_humanized_state(message, usage, runtime_state)
        return

    key, value = arguments[0].strip().lower(), arguments[1].strip()
    try:
        apply_runtime_setting(
            runtime_state_base,
            key,
            value,
            chat_id=None if is_global else message.chat.id,
        )
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

    scope = "глобально, во всех чатах" if is_global else "в этом чате"
    await reply_humanized_state(
        message, f"Обновлено: {key}={value} ({scope}, до перезапуска)", runtime_state
    )


@router.message(Command("set"), GroupOnly())
async def cmd_set_denied(message: Message, runtime_state: RuntimeState) -> None:
    """Fallback: вызывается, когда AdminOrOwner отказал в правах для /set."""
    await _reply_no_permission(message, runtime_state)


# Scoped to this chat like /set; `/setprob global 0.2` still retunes everyone.
@router.message(Command("setprob"), GroupOnly(), OwnerOnly())
async def cmd_setprob(
    message: Message,
    runtime_state: RuntimeState,
    settings: Settings,
    runtime_state_base: RuntimeState,
) -> None:
    raw = _extract_command_arg(message.text or "").strip()
    usage = "Использование: /setprob 0.2 (или /setprob global 0.2)"
    if not raw:
        await reply_humanized_state(message, usage, runtime_state)
        return

    parts = raw.split()
    is_global = parts[0].lower() == "global"
    if is_global:
        parts = parts[1:]
    if len(parts) != 1:
        await reply_humanized_state(message, usage, runtime_state)
        return

    # Goes through the shared apply path instead of assigning the field: that
    # is what routes the value into the right scope and keeps cross-field
    # validation in one place.
    try:
        apply_runtime_setting(
            runtime_state_base,
            "reply_probability",
            parts[0],
            chat_id=None if is_global else message.chat.id,
        )
    except (UnknownRuntimeSettingError, InvalidRuntimeSettingValueError):
        await reply_humanized_state(
            message, "Нужно число в диапазоне 0..1", runtime_state
        )
        return

    scope = "глобально, во всех чатах" if is_global else "в этом чате"
    await reply_humanized_state(
        message, f"REPLY_PROBABILITY теперь: {parts[0]} ({scope})", runtime_state
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
