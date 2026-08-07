from __future__ import annotations

import time
from collections.abc import Sequence

from aiogram import F, Router
from aiogram.types import Message

from app.config.runtime_state import RuntimeState
from app.core.markov import MarkovGenerator
from app.core.reply_policy import bot_is_mentioned
from app.handlers._helpers import (
    is_group_message,
    reply_humanized_sequence_state,
)
from app.services import LearningService, PivoService
from app.services.reply_pipeline import IncomingMessage, ReplyPipeline

router = Router(name="learning")


def _reply_to_authored_by_bot(reply_to: object, bot_id: int) -> bool:
    """True when the replied-to message was sent by the bot itself."""
    author = getattr(reply_to, "from_user", None)
    return author is not None and getattr(author, "id", None) == bot_id


def _incoming_message(
    message: Message,
    *,
    bot_username: str,
    bot_id: int,
    bot_text_aliases: frozenset[str],
    monotonic_now: float,
) -> IncomingMessage:
    """Факты о сообщении для конвейера — здесь заканчиваются типы Telegram."""
    assert message.from_user is not None  # проверено вызывающим
    reply_to = message.reply_to_message
    reply_context_text: str | None = None
    if reply_to and reply_to.text and not _reply_to_authored_by_bot(reply_to, bot_id):
        reply_context_text = reply_to.text
    return IncomingMessage(
        chat_id=message.chat.id,
        user_id=message.from_user.id,
        first_name=message.from_user.first_name or "",
        text=message.text or "",
        # Mention check before learning validation: a 1-token reply to the bot
        # (e.g. "ок", "?") should still trigger a response even if too short
        # to learn.
        mentioned=bot_is_mentioned(message, bot_username, bot_id, bot_text_aliases),
        is_reply=reply_to is not None,
        reply_context_text=reply_context_text,
        bot_aliases=bot_text_aliases,
        monotonic_now=monotonic_now,
    )


@router.message(F.text)
async def on_text_message(
    message: Message,
    learning_service: LearningService,
    generator: MarkovGenerator,
    runtime_state: RuntimeState,
    bot_username: str,
    bot_id: int,
    bot_text_aliases: frozenset[str],
    pivo_service: PivoService,
) -> None:
    if not is_group_message(message):
        return
    if message.from_user is None:
        return
    if message.from_user.is_bot:
        return

    # Каждое сообщение — свежий снимок профиля отправителя: /pivo упоминает по
    # @username, а он мог смениться после /pivo_on. No-op для неподписанных.
    await pivo_service.refresh_member(message.chat.id, message.from_user)

    if (message.text or "").startswith("/"):
        return

    incoming = _incoming_message(
        message,
        bot_username=bot_username,
        bot_id=bot_id,
        bot_text_aliases=bot_text_aliases,
        monotonic_now=time.monotonic(),
    )
    pipeline = ReplyPipeline(
        learning_service=learning_service,
        generator=generator,
        runtime_state=runtime_state,
    )
    observation = await pipeline.observe(incoming)
    if observation is None:
        return

    async def send(parts: Sequence[str]) -> None:
        await reply_humanized_sequence_state(
            message, parts, runtime_state, per_char=True
        )

    try:
        await pipeline.respond(incoming, observation, send)
    finally:
        # Обучение в finally намеренно: сообщение выучивается даже если
        # генерация или отправка ответа упали.
        await pipeline.learn(incoming, observation)
