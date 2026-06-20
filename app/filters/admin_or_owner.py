from __future__ import annotations

import logging

from aiogram import Bot
from aiogram.enums import ChatType
from aiogram.filters import Filter
from aiogram.types import Message

from app.config.settings import Settings

logger = logging.getLogger("chat_markov")


async def is_admin_or_owner(message: Message, bot: Bot, settings: Settings) -> bool:
    if message.from_user is None:
        return False
    user_id = message.from_user.id
    if settings.owner_id is not None and user_id == settings.owner_id:
        return True
    if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return False
    try:
        admins = await bot.get_chat_administrators(message.chat.id)
        return any(a.user.id == user_id for a in admins)
    except Exception as exc:
        logger.warning("Cannot verify chat admins: %s", exc)
        return False


class AdminOrOwner(Filter):
    async def __call__(self, message: Message, bot: Bot, settings: Settings) -> bool:
        return await is_admin_or_owner(message, bot, settings)
