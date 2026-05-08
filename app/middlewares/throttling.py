from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject


class ThrottlingMiddleware(BaseMiddleware):
    """Per-user per-command cooldown. Silently drops throttled updates."""

    def __init__(self, limits: dict[str, float]) -> None:
        # limits: command name (without /) -> cooldown in seconds
        self._limits = limits
        # (chat_id, user_id, command) -> last allowed timestamp
        self._last_used: dict[tuple[int, int, str], float] = {}

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message):
            return await handler(event, data)

        text = event.text or ""
        if not text.startswith("/") or event.from_user is None:
            return await handler(event, data)

        command = text.split()[0].lstrip("/").split("@")[0].lower()
        limit = self._limits.get(command)
        if limit is None:
            return await handler(event, data)

        key = (event.chat.id, event.from_user.id, command)
        now = time.monotonic()
        if now - self._last_used.get(key, 0.0) < limit:
            return None  # throttled — silently drop

        self._last_used[key] = now
        return await handler(event, data)
