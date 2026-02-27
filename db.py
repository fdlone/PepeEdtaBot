from __future__ import annotations

import asyncio
from collections import Counter
from typing import Optional

import aiosqlite


class Database:
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

        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);")
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_transitions_lookup ON transitions(chat_id, w1, w2);"
        )
        await db.execute("CREATE INDEX IF NOT EXISTS idx_starts_chat_id ON starts(chat_id);")
        await db.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def save_message(self, chat_id: int, author_id: int, text: str) -> None:
        async with self._lock:
            db = await self._get_conn()
            await db.execute(
                """
                INSERT INTO messages(chat_id, author_id, text)
                VALUES (?, ?, ?)
                """,
                (chat_id, author_id, text),
            )
            await db.commit()

    async def update_model(self, chat_id: int, tokens: list[str]) -> None:
        if len(tokens) < 3:
            return

        start_pair = (tokens[0], tokens[1])
        transitions_counter: Counter[tuple[str, str, str]] = Counter(
            (tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)
        )

        async with self._lock:
            db = await self._get_conn()
            await db.execute(
                """
                INSERT INTO starts(chat_id, w1, w2, cnt)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(chat_id, w1, w2)
                DO UPDATE SET cnt = cnt + 1
                """,
                (chat_id, start_pair[0], start_pair[1]),
            )

            await db.executemany(
                """
                INSERT INTO transitions(chat_id, w1, w2, w3, cnt)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, w1, w2, w3)
                DO UPDATE SET cnt = cnt + excluded.cnt
                """,
                [
                    (chat_id, w1, w2, w3, cnt)
                    for (w1, w2, w3), cnt in transitions_counter.items()
                ],
            )
            await db.commit()

    async def save_message_and_update_model(
        self, chat_id: int, author_id: int, raw_text: str, tokens: list[str]
    ) -> int:
        transitions_counter: Counter[tuple[str, str, str]] = Counter()
        if len(tokens) >= 3:
            transitions_counter = Counter(
                (tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)
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

            if len(tokens) >= 3:
                await db.execute(
                    """
                    INSERT INTO starts(chat_id, w1, w2, cnt)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(chat_id, w1, w2)
                    DO UPDATE SET cnt = cnt + 1
                    """,
                    (chat_id, tokens[0], tokens[1]),
                )

                await db.executemany(
                    """
                    INSERT INTO transitions(chat_id, w1, w2, w3, cnt)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id, w1, w2, w3)
                    DO UPDATE SET cnt = cnt + excluded.cnt
                    """,
                    [
                        (chat_id, w1, w2, w3, cnt)
                        for (w1, w2, w3), cnt in transitions_counter.items()
                    ],
                )

            cursor = await db.execute(
                """
                SELECT COALESCE(SUM(cnt), 0)
                FROM transitions
                WHERE chat_id = ?
                """,
                (chat_id,),
            )
            row = await cursor.fetchone()
            await db.commit()
            return int(row[0] or 0)

    async def get_starts(self, chat_id: int) -> list[tuple[str, str, int]]:
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                """
                SELECT w1, w2, cnt
                FROM starts
                WHERE chat_id = ?
                """,
                (chat_id,),
            )
            rows = await cursor.fetchall()
            return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]

    async def get_transitions(
        self, chat_id: int, w1: str, w2: str
    ) -> list[tuple[str, int]]:
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

    async def get_start_if_exists(
        self, chat_id: int, w1: str, w2: str
    ) -> Optional[tuple[str, str, int]]:
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                """
                SELECT w1, w2, cnt
                FROM starts
                WHERE chat_id = ? AND w1 = ? AND w2 = ?
                """,
                (chat_id, w1, w2),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return str(row[0]), str(row[1]), int(row[2])

    async def get_chat_token_volume(self, chat_id: int) -> int:
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                """
                SELECT COALESCE(SUM(cnt), 0)
                FROM transitions
                WHERE chat_id = ?
                """,
                (chat_id,),
            )
            row = await cursor.fetchone()
            return int(row[0] or 0)

    async def get_stats(self, chat_id: int) -> dict[str, int]:
        async with self._lock:
            db = await self._get_conn()
            cursor_msg = await db.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)
            )
            msg_count = int((await cursor_msg.fetchone())[0])

            cursor_starts = await db.execute(
                "SELECT COUNT(*) FROM starts WHERE chat_id = ?", (chat_id,)
            )
            starts_count = int((await cursor_starts.fetchone())[0])

            cursor_trans = await db.execute(
                "SELECT COUNT(*) FROM transitions WHERE chat_id = ?", (chat_id,)
            )
            transitions_count = int((await cursor_trans.fetchone())[0])

            cursor_volume = await db.execute(
                "SELECT COALESCE(SUM(cnt), 0) FROM transitions WHERE chat_id = ?",
                (chat_id,),
            )
            volume = int((await cursor_volume.fetchone())[0] or 0)

        return {
            "messages": msg_count,
            "starts": starts_count,
            "transitions": transitions_count,
            "volume": volume,
        }

    async def clear_chat(self, chat_id: int) -> None:
        async with self._lock:
            db = await self._get_conn()
            await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            await db.execute("DELETE FROM starts WHERE chat_id = ?", (chat_id,))
            await db.execute("DELETE FROM transitions WHERE chat_id = ?", (chat_id,))
            await db.commit()

    async def message_exists(self, chat_id: int, text: str) -> bool:
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                """
                SELECT 1
                FROM messages
                WHERE chat_id = ? AND text = ?
                LIMIT 1
                """,
                (chat_id, text),
            )
            row = await cursor.fetchone()
            return row is not None
