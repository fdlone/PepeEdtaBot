from __future__ import annotations

import asyncio
from collections import Counter
from typing import Optional

import aiosqlite


class Database:
    """SQLite gateway for message storage and variable-order Markov statistics."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not initialized. Call init() first.")
        return self._conn

    async def init(self) -> None:
        if self._conn is not None:
            return

        self._conn = await aiosqlite.connect(self.path)
        db = await self._get_conn()

        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA foreign_keys=ON;")

        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )

        # Legacy/fallback n=2 tables.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS starts (
                chat_id INTEGER NOT NULL,
                w1 TEXT NOT NULL,
                w2 TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(chat_id, w1, w2)
            );
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS transitions (
                chat_id INTEGER NOT NULL,
                w1 TEXT NOT NULL,
                w2 TEXT NOT NULL,
                w3 TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(chat_id, w1, w2, w3)
            );
            """
        )

        # New n=3 tables.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS starts3 (
                chat_id INTEGER NOT NULL,
                w1 TEXT NOT NULL,
                w2 TEXT NOT NULL,
                w3 TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(chat_id, w1, w2, w3)
            );
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS transitions3 (
                chat_id INTEGER NOT NULL,
                w1 TEXT NOT NULL,
                w2 TEXT NOT NULL,
                w3 TEXT NOT NULL,
                w4 TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(chat_id, w1, w2, w3, w4)
            );
            """
        )

        # n=1 backoff table: (w1 -> w2)
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS transitions1 (
                chat_id INTEGER NOT NULL,
                w1 TEXT NOT NULL,
                w2 TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(chat_id, w1, w2)
            );
            """
        )

        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);")
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_starts_lookup ON starts(chat_id, w1, w2);"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_transitions_lookup ON transitions(chat_id, w1, w2);"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_starts3_chat_id ON starts3(chat_id);"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_transitions3_lookup ON transitions3(chat_id, w1, w2, w3);"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_transitions1_lookup ON transitions1(chat_id, w1);"
        )
        await db.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def save_message_and_update_model(
        self, chat_id: int, author_id: int, raw_text: str, tokens: list[str]
    ) -> int:
        """Store raw message and update n=3/n=2/n=1 transition counters atomically."""
        starts2_pair: Optional[tuple[str, str]] = None
        starts3_triplet: Optional[tuple[str, str, str]] = None

        trans2_counter: Counter[tuple[str, str, str]] = Counter()
        trans3_counter: Counter[tuple[str, str, str, str]] = Counter()
        trans1_counter: Counter[tuple[str, str]] = Counter()

        if len(tokens) >= 2:
            starts2_pair = (tokens[0], tokens[1])
            trans1_counter = Counter((tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1))
        if len(tokens) >= 3:
            starts3_triplet = (tokens[0], tokens[1], tokens[2])
            trans2_counter = Counter(
                (tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)
            )
        if len(tokens) >= 4:
            trans3_counter = Counter(
                (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])
                for i in range(len(tokens) - 3)
            )

        async with self._lock:
            db = await self._get_conn()

            await db.execute(
                """
                INSERT INTO messages(chat_id, author_id, text)
                VALUES (?, ?, ?)
                """,
                (chat_id, author_id, raw_text),
            )

            if starts2_pair:
                await db.execute(
                    """
                    INSERT INTO starts(chat_id, w1, w2, cnt)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(chat_id, w1, w2)
                    DO UPDATE SET cnt = cnt + 1
                    """,
                    (chat_id, starts2_pair[0], starts2_pair[1]),
                )
            if starts3_triplet:
                await db.execute(
                    """
                    INSERT INTO starts3(chat_id, w1, w2, w3, cnt)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(chat_id, w1, w2, w3)
                    DO UPDATE SET cnt = cnt + 1
                    """,
                    (chat_id, starts3_triplet[0], starts3_triplet[1], starts3_triplet[2]),
                )
            if trans1_counter:
                await db.executemany(
                    """
                    INSERT INTO transitions1(chat_id, w1, w2, cnt)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(chat_id, w1, w2)
                    DO UPDATE SET cnt = cnt + excluded.cnt
                    """,
                    [(chat_id, w1, w2, cnt) for (w1, w2), cnt in trans1_counter.items()],
                )
            if trans2_counter:
                await db.executemany(
                    """
                    INSERT INTO transitions(chat_id, w1, w2, w3, cnt)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id, w1, w2, w3)
                    DO UPDATE SET cnt = cnt + excluded.cnt
                    """,
                    [
                        (chat_id, w1, w2, w3, cnt)
                        for (w1, w2, w3), cnt in trans2_counter.items()
                    ],
                )
            if trans3_counter:
                await db.executemany(
                    """
                    INSERT INTO transitions3(chat_id, w1, w2, w3, w4, cnt)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id, w1, w2, w3, w4)
                    DO UPDATE SET cnt = cnt + excluded.cnt
                    """,
                    [
                        (chat_id, w1, w2, w3, w4, cnt)
                        for (w1, w2, w3, w4), cnt in trans3_counter.items()
                    ],
                )

            cursor3 = await db.execute(
                "SELECT COALESCE(SUM(cnt), 0) FROM transitions3 WHERE chat_id = ?",
                (chat_id,),
            )
            volume3 = int((await cursor3.fetchone())[0] or 0)
            cursor2 = await db.execute(
                "SELECT COALESCE(SUM(cnt), 0) FROM transitions WHERE chat_id = ?",
                (chat_id,),
            )
            volume2 = int((await cursor2.fetchone())[0] or 0)

            await db.commit()
            return volume3 if volume3 > 0 else volume2

    async def get_starts(self, chat_id: int) -> list[tuple[str, str, int]]:
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                "SELECT w1, w2, cnt FROM starts WHERE chat_id = ?",
                (chat_id,),
            )
            rows = await cursor.fetchall()
        return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]

    async def get_starts3(self, chat_id: int) -> list[tuple[str, str, str, int]]:
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                "SELECT w1, w2, w3, cnt FROM starts3 WHERE chat_id = ?",
                (chat_id,),
            )
            rows = await cursor.fetchall()
        return [(str(r[0]), str(r[1]), str(r[2]), int(r[3])) for r in rows]

    async def get_start_if_exists(
        self, chat_id: int, w1: str, w2: str
    ) -> Optional[tuple[str, str, int]]:
        async with self._lock:
            db = await self._get_conn()
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
    ) -> Optional[tuple[str, str, str, int]]:
        async with self._lock:
            db = await self._get_conn()
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

    async def get_transitions(self, chat_id: int, w1: str, w2: str) -> list[tuple[str, int]]:
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                """
                SELECT w3, cnt
                FROM transitions
                WHERE chat_id = ? AND w1 = ? AND w2 = ?
                """,
                (chat_id, w1, w2),
            )
            rows = await cursor.fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]

    async def get_transitions3(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> list[tuple[str, int]]:
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                """
                SELECT w4, cnt
                FROM transitions3
                WHERE chat_id = ? AND w1 = ? AND w2 = ? AND w3 = ?
                """,
                (chat_id, w1, w2, w3),
            )
            rows = await cursor.fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]

    async def get_transitions1(self, chat_id: int, w1: str) -> list[tuple[str, int]]:
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                """
                SELECT w2, cnt
                FROM transitions1
                WHERE chat_id = ? AND w1 = ?
                """,
                (chat_id, w1),
            )
            rows = await cursor.fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]

    async def get_chat_token_volume(self, chat_id: int) -> int:
        async with self._lock:
            db = await self._get_conn()
            cursor3 = await db.execute(
                "SELECT COALESCE(SUM(cnt), 0) FROM transitions3 WHERE chat_id = ?",
                (chat_id,),
            )
            volume3 = int((await cursor3.fetchone())[0] or 0)
            if volume3 > 0:
                return volume3

            cursor2 = await db.execute(
                "SELECT COALESCE(SUM(cnt), 0) FROM transitions WHERE chat_id = ?",
                (chat_id,),
            )
            return int((await cursor2.fetchone())[0] or 0)

    async def get_stats(self, chat_id: int) -> dict[str, int]:
        async with self._lock:
            db = await self._get_conn()
            msg_count = int(
                (
                    await (
                        await db.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
                    ).fetchone()
                )[0]
            )
            starts2_count = int(
                (await (await db.execute("SELECT COUNT(*) FROM starts WHERE chat_id = ?", (chat_id,))).fetchone())[0]
            )
            starts3_count = int(
                (await (await db.execute("SELECT COUNT(*) FROM starts3 WHERE chat_id = ?", (chat_id,))).fetchone())[0]
            )
            trans2_count = int(
                (
                    await (
                        await db.execute("SELECT COUNT(*) FROM transitions WHERE chat_id = ?", (chat_id,))
                    ).fetchone()
                )[0]
            )
            trans3_count = int(
                (
                    await (
                        await db.execute("SELECT COUNT(*) FROM transitions3 WHERE chat_id = ?", (chat_id,))
                    ).fetchone()
                )[0]
            )
            trans1_count = int(
                (
                    await (
                        await db.execute("SELECT COUNT(*) FROM transitions1 WHERE chat_id = ?", (chat_id,))
                    ).fetchone()
                )[0]
            )
            volume2 = int(
                (
                    await (
                        await db.execute(
                            "SELECT COALESCE(SUM(cnt), 0) FROM transitions WHERE chat_id = ?",
                            (chat_id,),
                        )
                    ).fetchone()
                )[0]
                or 0
            )
            volume3 = int(
                (
                    await (
                        await db.execute(
                            "SELECT COALESCE(SUM(cnt), 0) FROM transitions3 WHERE chat_id = ?",
                            (chat_id,),
                        )
                    ).fetchone()
                )[0]
                or 0
            )
            volume1 = int(
                (
                    await (
                        await db.execute(
                            "SELECT COALESCE(SUM(cnt), 0) FROM transitions1 WHERE chat_id = ?",
                            (chat_id,),
                        )
                    ).fetchone()
                )[0]
                or 0
            )

        return {
            "messages": msg_count,
            "starts2": starts2_count,
            "starts3": starts3_count,
            "transitions2": trans2_count,
            "transitions3": trans3_count,
            "transitions1": trans1_count,
            "volume2": volume2,
            "volume3": volume3,
            "volume1": volume1,
            "volume": volume3 if volume3 > 0 else volume2,
        }

    async def clear_chat(self, chat_id: int) -> None:
        async with self._lock:
            db = await self._get_conn()
            await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            await db.execute("DELETE FROM starts WHERE chat_id = ?", (chat_id,))
            await db.execute("DELETE FROM starts3 WHERE chat_id = ?", (chat_id,))
            await db.execute("DELETE FROM transitions WHERE chat_id = ?", (chat_id,))
            await db.execute("DELETE FROM transitions3 WHERE chat_id = ?", (chat_id,))
            await db.execute("DELETE FROM transitions1 WHERE chat_id = ?", (chat_id,))
            await db.commit()

    async def message_exists(self, chat_id: int, text: str) -> bool:
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                "SELECT 1 FROM messages WHERE chat_id = ? AND text = ? LIMIT 1",
                (chat_id, text),
            )
            row = await cursor.fetchone()
        return row is not None
