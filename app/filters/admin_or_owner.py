from __future__ import annotations

import logging
import time

from aiogram import Bot
from aiogram.enums import ChatType
from aiogram.filters import Filter
from aiogram.types import Message

from app.config.settings import Settings

logger = logging.getLogger("chat_markov")

# Short TTL cache of admin user-id sets per chat. Admin commands are rare but
# get_chat_administrators is a network round-trip subject to Telegram rate
# limits (audit S4/P5); a few-second cache absorbs bursts without making stale
# membership decisions noticeable. INVARIANT: single-instance — this in-memory
# state is not shared across workers/instances (audit A4).
ADMIN_CACHE_TTL_SECONDS = 60.0
_admin_cache: dict[int, tuple[float, frozenset[int]]] = {}


async def _get_admin_ids(bot: Bot, chat_id: int) -> frozenset[int]:
    now = time.monotonic()
    cached = _admin_cache.get(chat_id)
    if cached is not None and now - cached[0] < ADMIN_CACHE_TTL_SECONDS:
        return cached[1]
    admins = await bot.get_chat_administrators(chat_id)
    admin_ids = frozenset(a.user.id for a in admins)
    _admin_cache[chat_id] = (now, admin_ids)
    return admin_ids


async def is_admin_or_owner(message: Message, bot: Bot, settings: Settings) -> bool:
    if message.from_user is None:
        return False
    user_id = message.from_user.id
    if settings.owner_id is not None and user_id == settings.owner_id:
        return True
    if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return False
    try:
        return user_id in await _get_admin_ids(bot, message.chat.id)
    except Exception as exc:
        logger.warning("Cannot verify chat admins: %s", exc)
        return False


class AdminOrOwner(Filter):
    async def __call__(self, message: Message, bot: Bot, settings: Settings) -> bool:
        return await is_admin_or_owner(message, bot, settings)
