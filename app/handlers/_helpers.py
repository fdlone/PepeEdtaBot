from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

from aiogram.enums import ChatAction, ChatType
from aiogram.types import Message

from app.log_masking import mask_chat_ids_in_text
from app.services.reply_pipeline import PartialDeliveryError

if TYPE_CHECKING:
    from app.config.runtime_state import RuntimeState

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


async def reply_humanized_state(
    message: Message,
    text: str,
    runtime_state: RuntimeState,
    *,
    per_char: bool = False,
) -> None:
    """Single-message ``reply_humanized_sequence`` sourcing the pause from state.

    Every handler sends its reply with the same runtime typing knobs; this
    wrapper keeps them from reaching into ``runtime_state`` at each call site.
    Command replies keep the flat pause (``per_char=False``); generated and
    fallback replies opt into per-char scaling with ``per_char=True``.
    ``reply_humanized_sequence`` itself stays knob-typed for tests and callers
    that pass explicit timings.
    """
    await reply_humanized_sequence_state(
        message, [text], runtime_state, per_char=per_char
    )


async def reply_humanized_sequence_state(
    message: Message,
    texts: Sequence[str],
    runtime_state: RuntimeState,
    *,
    per_char: bool = False,
) -> int:
    """``reply_humanized_sequence`` sourcing the typing pause from ``runtime_state``.

    The single point that resolves the runtime typing knobs: the single-message
    ``reply_humanized_state`` delegates here rather than reading them itself.

    Число доставленных частей прокидывается наверх — см. контракт ``Sender``.
    """
    return await reply_humanized_sequence(
        message,
        texts,
        runtime_state.typing_min_ms,
        runtime_state.typing_max_ms,
        typing_per_char_ms=runtime_state.typing_per_char_ms if per_char else 0,
    )


async def reply_humanized_sequence(
    message: Message,
    texts: Sequence[str],
    typing_min_ms: int,
    typing_max_ms: int,
    *,
    typing_per_char_ms: int = 0,
    rng: random.Random | None = None,
) -> int:
    """Send a sequence of messages with a humanized typing pause before each.

    The first non-empty part replies to the triggering message; follow-ups go
    as plain chat messages (send-then-send — edits look botty in Telegram
    clients). Chat-action/pause failures never block sending.

    Возвращает число **доставленных** частей. Вызывающему это нужно, чтобы
    отличить «ответа в чате нет» от «ответ ушёл частично»: ответ бота бывает
    многочастным (фальстарт, двойное сообщение, причуда завсегдатая), и сбой
    на второй части оставляет первую в чате. Без этого числа откат бюджета и
    анти-повтор не могли решить, состоялся ответ или нет, и трактовали
    частичную доставку как полное отсутствие ответа.

    При исключении число уже доставленных частей уходит вместе с ним
    (``PartialDeliveryError``), иначе оно теряется при пробросе.
    """
    first = True
    delivered = 0
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
            # Текст маскируется по тем же основаниям, что в errors.py:
            # send_chat_action принимает chat_id, и aiogram кладёт его сырым в
            # сообщение части исключений, а этот except до общего обработчика
            # не доводит. Путь горячий — вызывается перед каждым ответом.
            logger.debug(
                "send_chat_action/typing delay failed: %s",
                mask_chat_ids_in_text(str(exc)),
            )
        try:
            if first:
                await message.reply(text)
                first = False
            else:
                await message.answer(text)
        except Exception as exc:
            # Оборачиваем только частичную доставку — случай, ради которого
            # исключение и заведено. Когда не ушло ничего, исходное исключение
            # летит как прежде: его тип читают и обработчик ошибок
            # (TelegramAPIError), и все команды, которые шлют ответ через эту
            # же функцию. Подменять его всюду ради одной ветки конвейера
            # значило бы менять поведение там, где менять нечего.
            if delivered:
                raise PartialDeliveryError(delivered) from exc
            raise
        delivered += 1
    return delivered
