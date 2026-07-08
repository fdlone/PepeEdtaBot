from __future__ import annotations

import aiosqlite

from app.repositories.base_repo import BaseRepo


class MarkovRepo(BaseRepo):
    """Read-only доступ к таблицам starts/transitions/transitions3/transitions1."""

    async def get_starts(self, chat_id: int) -> list[tuple[str, str, int]]:
        rows = await self._fetch_all(
            """
            SELECT w1, w2, cnt
            FROM starts
            WHERE chat_id = ?
            ORDER BY w1, w2
            """,
            (chat_id,),
        )
        return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]

    async def get_starts3(self, chat_id: int) -> list[tuple[str, str, str, int]]:
        rows = await self._fetch_all(
            """
            SELECT w1, w2, w3, cnt
            FROM starts3
            WHERE chat_id = ?
            ORDER BY w1, w2, w3
            """,
            (chat_id,),
        )
        return [(str(r[0]), str(r[1]), str(r[2]), int(r[3])) for r in rows]

    async def get_start_if_exists(
        self, chat_id: int, w1: str, w2: str
    ) -> tuple[str, str, int] | None:
        row = await self._fetch_one(
            "SELECT w1, w2, cnt FROM starts WHERE chat_id = ? AND w1 = ? AND w2 = ?",
            (chat_id, w1, w2),
        )
        if not row:
            return None
        return str(row[0]), str(row[1]), int(row[2])

    async def get_start3_if_exists(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> tuple[str, str, str, int] | None:
        row = await self._fetch_one(
            """
            SELECT w1, w2, w3, cnt
            FROM starts3
            WHERE chat_id = ? AND w1 = ? AND w2 = ? AND w3 = ?
            """,
            (chat_id, w1, w2, w3),
        )
        if not row:
            return None
        return str(row[0]), str(row[1]), str(row[2]), int(row[3])

    async def get_transitions(
        self, chat_id: int, w1: str, w2: str
    ) -> list[tuple[str, int]]:
        rows = await self._fetch_all(
            """
            SELECT w3, cnt
            FROM transitions
            WHERE chat_id = ? AND w1 = ? AND w2 = ?
            ORDER BY w3
            """,
            (chat_id, w1, w2),
        )
        return [(str(r[0]), int(r[1])) for r in rows]

    async def get_transitions3(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> list[tuple[str, int]]:
        rows = await self._fetch_all(
            """
            SELECT w4, cnt
            FROM transitions3
            WHERE chat_id = ? AND w1 = ? AND w2 = ? AND w3 = ?
            ORDER BY w4
            """,
            (chat_id, w1, w2, w3),
        )
        return [(str(r[0]), int(r[1])) for r in rows]

    async def get_transitions1(self, chat_id: int, w1: str) -> list[tuple[str, int]]:
        rows = await self._fetch_all(
            """
            SELECT w2, cnt
            FROM transitions1
            WHERE chat_id = ? AND w1 = ?
            ORDER BY w2
            """,
            (chat_id, w1),
        )
        return [(str(r[0]), int(r[1])) for r in rows]

    async def get_states(
        self,
        chat_id: int,
        order: int,
    ) -> list[tuple[tuple[str, ...], int]]:
        if order not in {2, 3}:
            raise ValueError("order must be 2 or 3")

        table, columns = (
            ("transitions3", "w1, w2, w3") if order == 3 else ("transitions", "w1, w2")
        )
        rows = await self._fetch_all(
            f"""
            SELECT {columns}, SUM(cnt)
            FROM {table}
            WHERE chat_id = ?
            GROUP BY {columns}
            ORDER BY {columns}
            """,  # nosec B608 - table/columns picked from two literals above
            (chat_id,),
        )
        return [
            (
                tuple(str(row[index]) for index in range(order)),
                int(row[order]),
            )
            for row in rows
        ]

    async def get_chat_token_volume(self, chat_id: int) -> int:
        # Both sums share one lock acquisition: the 2-gram fallback only fires
        # when the 3-gram table is empty, and this runs on the per-message path.
        async with self._connection() as db:
            volume3 = await self._sum_cnt(db, "transitions3", chat_id)
            if volume3 > 0:
                return volume3
            return await self._sum_cnt(db, "transitions", chat_id)

    @staticmethod
    async def _sum_cnt(
        db: aiosqlite.Connection, table: str, chat_id: int
    ) -> int:
        cursor = await db.execute(
            f"SELECT COALESCE(SUM(cnt), 0) FROM {table} WHERE chat_id = ?",  # nosec B608 - table is a hardcoded caller constant
            (chat_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("COALESCE query returned None in get_chat_token_volume")
        return int(row[0] or 0)
