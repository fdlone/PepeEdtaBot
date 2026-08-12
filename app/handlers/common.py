from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.config.runtime_state import RuntimeState
from app.core.markov import MarkovGenerator
from app.filters import GROUP_ONLY
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


@router.message(Command("stats"), GROUP_ONLY)
async def cmd_stats(
    message: Message,
    db: Database,
    runtime_state: RuntimeState,
    generator: MarkovGenerator,
) -> None:
    # Показывается только объём модели, поэтому и читается только он: полный
    # срез (`get_stats`) — это пять COUNT-ов по чату, из которых команда не
    # печатает ни одного. Телеметрия генерации — счётчики процесса, без
    # обращений к базе. Реестр коллокаций (M2R-300) — один GROUP BY по чату.
    volume = await db.get_chat_token_volume(message.chat.id)
    collocations = await db.collocations.count_by_status(message.chat.id)
    await reply_humanized_state(
        message,
        format_stats_message(
            {"volume": volume},
            telemetry=generator.telemetry.snapshot(),
            collocations=collocations,
        ),
        runtime_state,
    )
