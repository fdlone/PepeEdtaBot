from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.config.runtime_state import RuntimeState
from app.filters import GroupOnly
from app.handlers._helpers import reply_humanized_state
from app.infrastructure.database import Database
from app.presentation.bot_messages import format_help_message, format_stats_message

router = Router(name="common")


@router.message(Command("ping"))
async def cmd_ping(message: Message) -> None:
    await message.reply("pong")


@router.message(Command("help"))
async def cmd_help(message: Message, runtime_state: RuntimeState) -> None:
    await reply_humanized_state(message, format_help_message(), runtime_state)


@router.message(Command("stats"), GroupOnly())
async def cmd_stats(
    message: Message, db: Database, runtime_state: RuntimeState
) -> None:
    stats = await db.get_stats(message.chat.id)
    await reply_humanized_state(
        message, format_stats_message(stats), runtime_state
    )
