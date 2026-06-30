from __future__ import annotations

import asyncio
import logging
import random

from aiogram.enums import ChatAction, ChatType
from aiogram.types import Message

logger = logging.getLogger("chat_markov")


def is_group_message(message: Message) -> bool:
    return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}


async def reply_humanized(
    message: Message,
    text: str,
    typing_min_ms: int,
    typing_max_ms: int,
    *,
    rng: random.Random | None = None,
) -> None:
    """Имитация «печатает...»: chat action + случайная пауза, затем reply."""
    try:
        if message.bot is not None:
            await message.bot.send_chat_action(
                chat_id=message.chat.id, action=ChatAction.TYPING
            )
        randint = random.randint if rng is None else rng.randint
        delay_ms = randint(typing_min_ms, typing_max_ms)
        await asyncio.sleep(delay_ms / 1000)
    except Exception as exc:
        # Ошибка chat action не должна блокировать отправку обычного ответа.
        logger.debug("send_chat_action/typing delay failed: %s", exc)
    await message.reply(text)
