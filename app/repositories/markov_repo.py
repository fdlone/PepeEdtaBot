from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import aiosqlite

ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]


class MarkovRepo:
    """Read-only доступ к таблицам starts/transitions/transitions3/transitions1."""

    def __init__(self, conn_provider: ConnProvider, lock: asyncio.Lock) -> None:
        self._conn_provider = conn_provider
        self._lock = lock

    async def get_starts(self, chat_id: int) -> list[tuple[str, str, int]]:
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                """
                SELECT w1, w2, cnt
                FROM starts
                WHERE chat_id = ?
                ORDER BY w1, w2
                """,
                (chat_id,),
            )
            rows = await cursor.fetchall()
        return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]

    async def get_starts3(self, chat_id: int) -> list[tuple[str, str, str, int]]:
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                """
                SELECT w1, w2, w3, cnt
                FROM starts3
                WHERE chat_id = ?
                ORDER BY w1, w2, w3
                """,
                (chat_id,),
            )
            rows = await cursor.fetchall()
        return [(str(r[0]), str(r[1]), str(r[2]), int(r[3])) for r in rows]

    async def get_start_if_exists(
        self, chat_id: int, w1: str, w2: str
    ) -> tuple[str, str, int] | None:
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                "SELECT w1, w2, cnt FROM starts WHERE chat_id = ? AND w1 = ? AND w2 = ?",
                (chat_id, w1, w2),
            )
            row = await cursor.fetchone()
        if not row:
            return None
        return str(row[0]), str(row[1]), int(row[2])

    async def get_start3_if_exists(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> tuple[str, str, str, int] | None:
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                """
                SELECT w1, w2, w3, cnt
                FROM starts3
                WHERE chat_id = ? AND w1 = ? AND w2 = ? AND w3 = ?
                """,
                (chat_id, w1, w2, w3),
            )
            row = await cursor.fetchone()
        if not row:
            return None
        return str(row[0]), str(row[1]), str(row[2]), int(row[3])

    async def get_transitions(
        self, chat_id: int, w1: str, w2: str
    ) -> list[tuple[str, int]]:
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                """
                SELECT w3, cnt
                FROM transitions
                WHERE chat_id = ? AND w1 = ? AND w2 = ?
                ORDER BY w3
                """,
                (chat_id, w1, w2),
            )
            rows = await cursor.fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]

    async def get_transitions3(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> list[tuple[str, int]]:
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                """
                SELECT w4, cnt
                FROM transitions3
                WHERE chat_id = ? AND w1 = ? AND w2 = ? AND w3 = ?
                ORDER BY w4
                """,
                (chat_id, w1, w2, w3),
            )
            rows = await cursor.fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]

    async def get_transitions1(self, chat_id: int, w1: str) -> list[tuple[str, int]]:
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                """
                SELECT w2, cnt
                FROM transitions1
                WHERE chat_id = ? AND w1 = ?
                ORDER BY w2
                """,
                (chat_id, w1),
            )
            rows = await cursor.fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]

    async def get_states(
        self,
        chat_id: int,
        order: int,
    ) -> list[tuple[tuple[str, ...], int]]:
        if order not in {2, 3}:
            raise ValueError("order must be 2 or 3")

        async with self._lock:
            db = await self._conn_provider()
            if order == 3:
                cursor = await db.execute(
                    """
                    SELECT w1, w2, w3, SUM(cnt)
                    FROM transitions3
                    WHERE chat_id = ?
                    GROUP BY w1, w2, w3
                    ORDER BY w1, w2, w3
                    """,
                    (chat_id,),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT w1, w2, SUM(cnt)
                    FROM transitions
                    WHERE chat_id = ?
                    GROUP BY w1, w2
                    ORDER BY w1, w2
                    """,
                    (chat_id,),
                )
            rows = await cursor.fetchall()

        state_size = order
        return [
            (
                tuple(str(row[index]) for index in range(state_size)),
                int(row[state_size]),
            )
            for row in rows
        ]

    async def get_chat_token_volume(self, chat_id: int) -> int:
        async with self._lock:
            db = await self._conn_provider()
            cursor3 = await db.execute(
                "SELECT COALESCE(SUM(cnt), 0) FROM transitions3 WHERE chat_id = ?",
                (chat_id,),
            )
            row3 = await cursor3.fetchone()
            if row3 is None:
                raise RuntimeError("COALESCE query returned None in get_chat_token_volume")
            volume3 = int(row3[0] or 0)
            if volume3 > 0:
                return volume3

            cursor2 = await db.execute(
                "SELECT COALESCE(SUM(cnt), 0) FROM transitions WHERE chat_id = ?",
                (chat_id,),
            )
            row2 = await cursor2.fetchone()
            if row2 is None:
                raise RuntimeError("COALESCE query returned None in get_chat_token_volume")
            return int(row2[0] or 0)
