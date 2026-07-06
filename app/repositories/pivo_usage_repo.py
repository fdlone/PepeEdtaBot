from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import aiosqlite

ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]


class PivoUsageRepo:
    """Суточные квоты вызова /pivo по chat/user hash."""

    def __init__(self, conn_provider: ConnProvider, lock: asyncio.Lock) -> None:
        self._conn_provider = conn_provider
        self._lock = lock

    async def consume_daily_call(
        self,
        *,
        chat_hash: str,
        user_hash: str,
        usage_day: str,
        limit: int,
    ) -> tuple[bool, int]:
        """Пытается списать один вызов и возвращает (allowed, used_count)."""
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                """
                SELECT used_count
                FROM pivo_daily_usage
                WHERE chat_hash = ? AND user_hash = ? AND usage_day = ?
                """,
                (chat_hash, user_hash, usage_day),
            )
            row = await cursor.fetchone()

            if row is None:
                await db.execute(
                    """
                    INSERT INTO pivo_daily_usage(chat_hash, user_hash, usage_day, used_count)
                    VALUES (?, ?, ?, 1)
                    """,
                    (chat_hash, user_hash, usage_day),
                )
                await db.commit()
                return True, 1

            used_count = int(row[0])
            if used_count >= limit:
                return False, used_count

            used_count += 1
            await db.execute(
                """
                UPDATE pivo_daily_usage
                SET used_count = ?, updated_at = datetime('now')
                WHERE chat_hash = ? AND user_hash = ? AND usage_day = ?
                """,
                (used_count, chat_hash, user_hash, usage_day),
            )
            await db.commit()
            return True, used_count

    async def refund_daily_call(
        self,
        *,
        chat_hash: str,
        user_hash: str,
        usage_day: str,
    ) -> None:
        """Возвращает один списанный вызов, если ответ /pivo не был доставлен."""
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                """
                SELECT used_count
                FROM pivo_daily_usage
                WHERE chat_hash = ? AND user_hash = ? AND usage_day = ?
                """,
                (chat_hash, user_hash, usage_day),
            )
            row = await cursor.fetchone()
            if row is None:
                return

            used_count = int(row[0])
            if used_count <= 1:
                await db.execute(
                    """
                    DELETE FROM pivo_daily_usage
                    WHERE chat_hash = ? AND user_hash = ? AND usage_day = ?
                    """,
                    (chat_hash, user_hash, usage_day),
                )
            else:
                await db.execute(
                    """
                    UPDATE pivo_daily_usage
                    SET used_count = ?, updated_at = datetime('now')
                    WHERE chat_hash = ? AND user_hash = ? AND usage_day = ?
                    """,
                    (used_count - 1, chat_hash, user_hash, usage_day),
                )
            await db.commit()

    async def delete_chat_usage(self, chat_hash: str) -> None:
        """Удаляет все квоты чата (используется /clear)."""
        async with self._lock:
            db = await self._conn_provider()
            await db.execute(
                "DELETE FROM pivo_daily_usage WHERE chat_hash = ?",
                (chat_hash,),
            )
            await db.commit()

    async def delete_usage_before(self, cutoff_day: str) -> int:
        """Deletes quota rows older than cutoff_day and returns deleted row count."""
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                """
                DELETE FROM pivo_daily_usage
                WHERE usage_day < ?
                """,
                (cutoff_day,),
            )
            await db.commit()
            return cursor.rowcount
