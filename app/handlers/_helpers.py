from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Sequence

from aiogram.enums import ChatAction, ChatType
from aiogram.types import Message

logger = logging.getLogger("chat_markov")

# Absolute ceiling for the humanized typing pause so long replies never make
# the bot look frozen (the per-char component grows with text length).
TYPING_HARD_CAP_MS = 4000


def is_group_message(message: Message) -> bool:
    return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}


def compute_typing_delay_ms(
    text_length: int,
    typing_min_ms: int,
    typing_max_ms: int,
    typing_per_char_ms: int = 0,
    *,
    rng: random.Random | None = None,
) -> int:
    """Случайная база + компонент, пропорциональный длине текста, с потолком."""
    randint = random.randint if rng is None else rng.randint
    base = randint(typing_min_ms, typing_max_ms)
    delay = base + typing_per_char_ms * max(0, text_length)
    return min(delay, max(TYPING_HARD_CAP_MS, typing_max_ms))


async def reply_humanized(
    message: Message,
    text: str,
    typing_min_ms: int,
    typing_max_ms: int,
    *,
    typing_per_char_ms: int = 0,
    rng: random.Random | None = None,
) -> None:
    """Имитация «печатает...»: chat action + случайная пауза, затем reply."""
    await reply_humanized_sequence(
        message,
        [text],
        typing_min_ms,
        typing_max_ms,
        typing_per_char_ms=typing_per_char_ms,
        rng=rng,
    )


async def reply_humanized_sequence(
    message: Message,
    texts: Sequence[str],
    typing_min_ms: int,
    typing_max_ms: int,
    *,
    typing_per_char_ms: int = 0,
    rng: random.Random | None = None,
) -> None:
    """Send a sequence of messages with a humanized typing pause before each.

    The first non-empty part replies to the triggering message; follow-ups go
    as plain chat messages (send-then-send — edits look botty in Telegram
    clients). Chat-action/pause failures never block sending.
    """
    first = True
    for text in texts:
        if not text:
            continue
        try:
            if message.bot is not None:
                await message.bot.send_chat_action(
                    chat_id=message.chat.id, action=ChatAction.TYPING
                )
            delay_ms = compute_typing_delay_ms(
                len(text),
                typing_min_ms,
                typing_max_ms,
                typing_per_char_ms,
                rng=rng,
            )
            await asyncio.sleep(delay_ms / 1000)
        except Exception as exc:
            # Ошибка chat action не должна блокировать отправку обычного ответа.
            logger.debug("send_chat_action/typing delay failed: %s", exc)
        if first:
            await message.reply(text)
            first = False
        else:
            await message.answer(text)
