"""Substitutes the per-chat view of ``RuntimeState`` into handler context.

Roughly 58 fields are read as ``runtime_state.X`` across the generator and the
handlers. Rewriting every read site to consult an overlay would be both large
and unsafe — a missed site does not fail, it silently keeps using the global
value. So no read site changes: this middleware swaps the object handlers
receive for the one this chat should see.

For a chat without overrides ``effective`` returns the very same instance, so
that path is byte-for-byte what it was before overlays existed.

The base state stays available under ``BASE_STATE_KEY`` for the handlers that
*write* settings (``/set``, ``/setprob``, ``/config``): writing into the
per-chat view would report success and then vanish on the next update.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from app.config.runtime_state import RuntimeState

BASE_STATE_KEY = "runtime_state_base"


class ChatSettingsMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        state = data.get("runtime_state")
        if isinstance(event, Message) and isinstance(state, RuntimeState):
            data[BASE_STATE_KEY] = state
            data["runtime_state"] = state.effective(event.chat.id)
        return await handler(event, data)
