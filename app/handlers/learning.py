from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from aiogram import Bot, F, Router
from aiogram.types import Message

from app.config.runtime_state import RuntimeState
from app.config.settings import Settings
from app.core.markov import MarkovGenerator
from app.core.reply_policy import bot_is_mentioned
from app.handlers._helpers import (
    is_group_message,
    reply_humanized_sequence_state,
)
from app.services import LearningService, PivoService
from app.services.learning_service import MaintenanceAlert
from app.services.reply_pipeline import IncomingMessage, ReplyPipeline

router = Router(name="learning")
logger = logging.getLogger("chat_markov")


def _format_maintenance_alert(alert: MaintenanceAlert) -> str:
    hours = alert.failing_for_sec / 3600.0
    if alert.recovered:
        return (
            "✅ Обслуживание базы снова проходит "
            f"(не проходило ~{hours:.0f} ч).\n"
            "Затухание счётчиков возобновлено."
        )
    reason = alert.reason or "причина неизвестна"
    return (
        f"⚠️ Обслуживание базы не проходит уже ~{hours:.0f} ч.\n"
        "Ответы и обучение работают, но счётчики перестали стареть: "
        "«горячие» n-граммы и эмодзи чата больше не выветриваются.\n"
        f"Причина: {reason}\n"
        "Бот продолжает пробовать; следующее напоминание — не раньше чем "
        "через сутки."
    )


async def _report_maintenance_to_owner(
    alert: MaintenanceAlert, bot: Bot | None, settings: Settings | None
) -> None:
    """Шлёт владельцу сигнал о состоянии обслуживания базы.

    Логи владельцу недоступны, а сбой обслуживания снаружи выглядит как
    «бот жив, просто мемы перестали выветриваться» — без сигнала это
    замечается через недели. Ошибка доставки гасится: сигнал не может
    отменить ни ответ, ни обучение.
    """
    if bot is None or settings is None or settings.owner_id is None:
        return
    try:
        await bot.send_message(settings.owner_id, _format_maintenance_alert(alert))
    except Exception:
        # Самый вероятный случай — владелец не начинал диалог с ботом.
        logger.warning("maintenance alert not delivered", exc_info=True)


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
    *,
    # Только для сигнала владельцу об обслуживании. Keyword-only и с
    # умолчанием: aiogram подставляет оба всегда, а «некому и нечем сигналить»
    # — то же самое, что незаданный OWNER_ID, то есть штатная конфигурация.
    bot: Bot | None = None,
    settings: Settings | None = None,
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
        # Обслуживание — после обучения и отдельным вызовом: планировщика в
        # проекте нет, а его сбой не должен стоить выученного сообщения.
        maintenance_alert = await pipeline.run_due_maintenance()
        if maintenance_alert is not None:
            await _report_maintenance_to_owner(maintenance_alert, bot, settings)
