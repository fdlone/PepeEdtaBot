from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.filters import GroupOnly
from app.handlers._helpers import reply_humanized
from bot_messages import format_help_message, format_stats_message
from db import Database
from runtime_state import RuntimeState

router = Router(name="common")


@router.message(Command("ping"))
async def cmd_ping(message: Message) -> None:
    await message.reply("pong")


@router.message(Command("help"))
async def cmd_help(message: Message, state: RuntimeState) -> None:
    await reply_humanized(
        message, format_help_message(), state.typing_min_ms, state.typing_max_ms
    )


@router.message(Command("stats"), GroupOnly())
async def cmd_stats(message: Message, db: Database, state: RuntimeState) -> None:
    stats = await db.get_stats(message.chat.id)
    text = format_stats_message(stats, state.min_tokens_for_model)
    await reply_humanized(message, text, state.typing_min_ms, state.typing_max_ms)
