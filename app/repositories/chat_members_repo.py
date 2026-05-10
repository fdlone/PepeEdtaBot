from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import aiosqlite

ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]


class ChatMembersRepo:
    """Доступ к таблице chat_members — каноническое хранилище участников чата.

    Сейчас единственный потребитель — `/pivo` (подписка / упоминания), но
    таблица намеренно не привязана к конкретной фиче: будущие функции,
    которым нужно персистентное состояние участника, ходят сюда же.
    """

    def __init__(self, conn_provider: ConnProvider, lock: asyncio.Lock) -> None:
        self._conn_provider = conn_provider
        self._lock = lock

    async def upsert(
        self,
        *,
        chat_hash: str,
        user_hash: str,
        encrypted_user_id: str,
        encrypted_username: str,
        encrypted_display_name: str,
    ) -> None:
        async with self._lock:
            db = await self._conn_provider()
            await db.execute(
                """
                INSERT INTO chat_members(
                    chat_hash,
                    user_hash,
                    encrypted_user_id,
                    encrypted_username,
                    encrypted_display_name
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_hash, user_hash)
                DO UPDATE SET
                    encrypted_user_id = excluded.encrypted_user_id,
                    encrypted_username = excluded.encrypted_username,
                    encrypted_display_name = excluded.encrypted_display_name,
                    updated_at = datetime('now')
                """,
                (
                    chat_hash,
                    user_hash,
                    encrypted_user_id,
                    encrypted_username,
                    encrypted_display_name,
                ),
            )
            await db.commit()

    async def remove(self, chat_hash: str, user_hash: str) -> None:
        async with self._lock:
            db = await self._conn_provider()
            await db.execute(
                "DELETE FROM chat_members WHERE chat_hash = ? AND user_hash = ?",
                (chat_hash, user_hash),
            )
            await db.commit()

    async def list_members(self, chat_hash: str) -> list[dict[str, object]]:
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                """
                SELECT encrypted_user_id, encrypted_username, encrypted_display_name
                FROM chat_members
                WHERE chat_hash = ?
                ORDER BY created_at, user_hash
                """,
                (chat_hash,),
            )
            rows = await cursor.fetchall()
        return [
            {
                "encrypted_user_id": str(row[0]),
                "encrypted_username": str(row[1]),
                "encrypted_display_name": str(row[2]),
            }
            for row in rows
        ]
