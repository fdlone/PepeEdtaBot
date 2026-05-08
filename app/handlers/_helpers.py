from __future__ import annotations

import asyncio
import random

from aiogram.enums import ChatAction
from aiogram.types import Message


async def reply_humanized(
    message: Message, text: str, typing_min_ms: int, typing_max_ms: int
) -> None:
    """Имитация «печатает...»: chat action + случайная пауза, затем reply."""
    try:
        await message.bot.send_chat_action(
            chat_id=message.chat.id, action=ChatAction.TYPING
        )
        delay_ms = random.randint(typing_min_ms, typing_max_ms)
        await asyncio.sleep(delay_ms / 1000)
    except Exception:
        # Ошибка chat action не должна блокировать отправку обычного ответа.
        pass
    await message.reply(text)
