from __future__ import annotations

import logging

from aiogram import Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import ErrorEvent

from app.log_masking import mask_chat_ids_in_text

router = Router(name="errors")
logger = logging.getLogger("chat_markov")


@router.error()
async def handle_error(event: ErrorEvent) -> None:
    exc = event.exception
    if isinstance(exc, TelegramAPIError):
        # aiogram bakes a raw chat_id into the message text of some errors
        # (TelegramRetryAfter: "Flood control exceeded … in chat -100…",
        # TelegramMigrateToChat). Masking the arguments is not enough here —
        # the identifier arrives inside the string, so the text itself is
        # sanitised. Applied to every TelegramAPIError rather than the two
        # known classes: the list belongs to aiogram and can grow.
        logger.error(
            "Telegram API error in handler: %s", mask_chat_ids_in_text(str(exc))
        )
    else:
        logger.error("Unhandled exception in handler", exc_info=exc)
