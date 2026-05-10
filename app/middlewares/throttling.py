from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject


class ThrottlingMiddleware(BaseMiddleware):
    """Per-user per-command cooldown.

    By default throttled updates are silently dropped. Commands listed in
    ``notify_on_throttle`` instead get a short reply telling the user how
    long is left — useful for explicit user-driven commands like ``/clear``,
    where silence reads as "the bot is broken".
    """

    def __init__(
        self,
        limits: dict[str, float],
        notify_on_throttle: set[str] | None = None,
    ) -> None:
        # limits: command name (without /) -> cooldown in seconds
        self._limits = limits
        self._notify_on_throttle = notify_on_throttle or set()
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

        parts = text.split(maxsplit=1)
        command = parts[0].lstrip("/").split("@")[0].lower()
        limit = self._limits.get(command)
        if limit is None:
            return await handler(event, data)

        throttle_key = command
        if command == "clear" and len(parts) > 1 and parts[1].strip().lower() == "confirm":
            throttle_key = "clear:confirm"

        key = (event.chat.id, event.from_user.id, throttle_key)
        now = time.monotonic()
        elapsed = now - self._last_used.get(key, 0.0)
        if elapsed < limit:
            if command in self._notify_on_throttle:
                remaining = max(1, math.ceil(limit - elapsed))
                await event.reply(
                    f"Слишком часто. Подождите ~{remaining} сек."
                )
            return None  # throttled

        self._last_used[key] = now
        return await handler(event, data)
