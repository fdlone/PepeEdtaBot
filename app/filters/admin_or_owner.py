from __future__ import annotations

import logging
import time

from aiogram import Bot
from aiogram.enums import ChatType
from aiogram.filters import Filter
from aiogram.types import Message

from app.config.settings import Settings
from app.log_masking import mask_chat_ids_in_text

logger = logging.getLogger("chat_markov")

# Short TTL cache of admin user-id sets per chat. Admin commands are rare but
# get_chat_administrators is a network round-trip subject to Telegram rate
# limits (audit S4/P5); a 60-second cache absorbs bursts at the cost of a
# demoted admin keeping rights for up to the TTL. INVARIANT: single-instance —
# this in-memory state is not shared across workers/instances (audit A4).
ADMIN_CACHE_TTL_SECONDS = 60.0
# An entry stays cached only while it is being used: the TTL above governs
# freshness, these two govern growth. Without them the dict keeps a row for
# every chat that ever ran an admin command, and there is no chat allowlist,
# so the key set is driven from outside. Same policy as the neighbouring
# in-memory states (``RuntimeState.prune_inactive``,
# ``ThrottlingMiddleware._prune_state``); the numbers are generous because an
# entry is one small tuple and eviction only costs a refetch.
ADMIN_CACHE_MAX_CHATS = 512
_ADMIN_CACHE_CLEANUP_EVERY = 64

_admin_cache: dict[int, tuple[float, frozenset[int]]] = {}
_cleanup_tick = 0


def _prune_admin_cache(now: float) -> None:
    """Drop entries nobody refreshed, then cap what is left.

    Eviction never changes an access decision: a chat that comes back simply
    gets its admin list refetched, exactly as it would after the TTL.
    """
    cutoff = now - ADMIN_CACHE_TTL_SECONDS
    stale_keys = [
        chat_id
        for chat_id, (cached_at, _) in _admin_cache.items()
        if cached_at < cutoff
    ]
    for chat_id in stale_keys:
        _admin_cache.pop(chat_id, None)

    overflow = len(_admin_cache) - ADMIN_CACHE_MAX_CHATS
    if overflow > 0:
        oldest = sorted(_admin_cache.items(), key=lambda item: item[1][0])[:overflow]
        for chat_id, _ in oldest:
            _admin_cache.pop(chat_id, None)


def reset_admin_cache() -> None:
    """Drop the whole cache. Tests use this to start from a known state."""
    global _cleanup_tick
    _admin_cache.clear()
    _cleanup_tick = 0


async def _get_admin_ids(bot: Bot, chat_id: int) -> frozenset[int]:
    global _cleanup_tick
    now = time.monotonic()

    _cleanup_tick += 1
    if _cleanup_tick >= _ADMIN_CACHE_CLEANUP_EVERY or (
        len(_admin_cache) > ADMIN_CACHE_MAX_CHATS
    ):
        _cleanup_tick = 0
        _prune_admin_cache(now)

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
        # `get_chat_administrators` принимает chat_id, а aiogram зашивает сырой
        # chat_id в текст части исключений (TelegramRetryAfter,
        # TelegramMigrateToChat). Сюда общий обработчик ошибок не достаёт: этот
        # except глушит исключение раньше, — поэтому маскирование применяется
        # здесь, как в errors.py.
        logger.warning("Cannot verify chat admins: %s", mask_chat_ids_in_text(str(exc)))
        return False


class AdminOrOwner(Filter):
    async def __call__(self, message: Message, bot: Bot, settings: Settings) -> bool:
        return await is_admin_or_owner(message, bot, settings)
