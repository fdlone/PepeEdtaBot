from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import aiosqlite

from text_utils import sanitize_text

ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]


class MessagesRepo:
    """Доступ к таблице messages: чтение и проверка наличия."""

    def __init__(self, conn_provider: ConnProvider, lock: asyncio.Lock) -> None:
        self._conn_provider = conn_provider
        self._lock = lock

    async def exists(self, chat_id: int, text: str) -> bool:
        normalized = sanitize_text(text)
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                "SELECT 1 FROM messages WHERE chat_id = ? AND normalized_text = ? LIMIT 1",
                (chat_id, normalized),
            )
            row = await cursor.fetchone()
        return row is not None

    async def get_all_normalized(self, chat_id: int) -> list[str]:
        async with self._lock:
            db = await self._conn_provider()
            cur = await db.execute(
                "SELECT normalized_text FROM messages WHERE chat_id = ? AND normalized_text != ''",
                (chat_id,),
            )
            rows = await cur.fetchall()
        return [str(row[0]) for row in rows]
