from __future__ import annotations

import logging

from aiogram import Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import ErrorEvent

router = Router(name="errors")
logger = logging.getLogger("chat_markov")


@router.error()
async def handle_error(event: ErrorEvent) -> None:
    exc = event.exception
    if isinstance(exc, TelegramAPIError):
        logger.error("Telegram API error in handler: %s", exc)
    else:
        logger.error("Unhandled exception in handler", exc_info=exc)
