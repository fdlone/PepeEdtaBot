from __future__ import annotations

import logging
import traceback

from aiogram import Router
from aiogram.types import ErrorEvent

from app.log_masking import mask_chat_ids_in_text

router = Router(name="errors")
logger = logging.getLogger("chat_markov")


@router.error()
async def handle_error(event: ErrorEvent) -> None:
    exc = event.exception
    # aiogram bakes a raw chat_id into the message text of some errors
    # (TelegramRetryAfter: "Flood control exceeded … in chat -100…",
    # TelegramMigrateToChat). Masking the arguments is not enough here —
    # the identifier arrives inside the string, so the text itself is
    # sanitised.
    #
    # Маскируется ВСЁ и всегда, а не только `TelegramAPIError`, и не только
    # само исключение, а вся цепочка. Прежняя форма выбирала ветку по типу, и
    # это оказалось хрупким по построению: обёртка над отправкой
    # (`PartialDeliveryError`, добавленная 2026-08-26) подменила тип, и
    # исключение с сырым chat_id внутри `__cause__` пошло в ветку
    # `exc_info=exc` — то есть в traceback, который печатает и причину.
    # Инвариант §4 CLAUDE.md не должен зависеть от того, не завернул ли кто-то
    # исключение по дороге; перечисление типов такую зависимость создаёт.
    trace = "".join(traceback.format_exception(exc))
    logger.error(
        "Exception in handler: %s", mask_chat_ids_in_text(trace).rstrip()
    )
