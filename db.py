from __future__ import annotations

import asyncio
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import Optional

import aiosqlite

from app.infrastructure import migrator
from app.repositories import ChatMembersRepo, MarkovRepo, MessagesRepo, PivoUsageRepo
from text_utils import sanitize_text

PIVO_DAILY_USAGE_RETENTION_DAYS = 7


class Database:
    """Фасад над SQLite: владеет соединением, миграциями и репозиториями."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self.markov: Optional[MarkovRepo] = None
        self.messages: Optional[MessagesRepo] = None
        self.chat_members: Optional[ChatMembersRepo] = None
        self.pivo_usage: Optional[PivoUsageRepo] = None

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

        await migrator.run(db)

        self.markov = MarkovRepo(self._get_conn, self._lock)
        self.messages = MessagesRepo(self._get_conn, self._lock)
        self.chat_members = ChatMembersRepo(self._get_conn, self._lock)
        self.pivo_usage = PivoUsageRepo(self._get_conn, self._lock)
        await self.cleanup_pivo_daily_usage()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        self.markov = None
        self.messages = None
        self.chat_members = None
        self.pivo_usage = None

    async def save_message_and_update_model(
        self, chat_id: int, raw_text: str, tokens: list[str]
    ) -> int:
        """Атомарно сохраняет raw-сообщение и обновляет счётчики переходов n=3/n=2/n=1."""
        starts2_pair: Optional[tuple[str, str]] = None
        starts3_triplet: Optional[tuple[str, str, str]] = None

        trans2_counter: Counter[tuple[str, str, str]] = Counter()
        trans3_counter: Counter[tuple[str, str, str, str]] = Counter()
        trans1_counter: Counter[tuple[str, str]] = Counter()

        if len(tokens) >= 2:
            starts2_pair = (tokens[0], tokens[1])
            trans1_counter = Counter(
                (tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)
            )
        if len(tokens) >= 3:
            starts3_triplet = (tokens[0], tokens[1], tokens[2])
            trans2_counter = Counter(
                (tokens[i], tokens[i + 1], tokens[i + 2])
                for i in range(len(tokens) - 2)
            )
        if len(tokens) >= 4:
            trans3_counter = Counter(
                (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])
                for i in range(len(tokens) - 3)
            )

        async with self._lock:
            db = await self._get_conn()

            await db.execute(
                "INSERT INTO messages(chat_id, author_id, normalized_text) VALUES (?, ?, ?)",
                (chat_id, 0, sanitize_text(raw_text)),
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
                    (
                        chat_id,
                        starts3_triplet[0],
                        starts3_triplet[1],
                        starts3_triplet[2],
                    ),
                )
            if trans1_counter:
                await db.executemany(
                    """
                    INSERT INTO transitions1(chat_id, w1, w2, cnt)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(chat_id, w1, w2)
                    DO UPDATE SET cnt = cnt + excluded.cnt
                    """,
                    [
                        (chat_id, w1, w2, cnt)
                        for (w1, w2), cnt in trans1_counter.items()
                    ],
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
            row3 = await cursor3.fetchone()
            if row3 is None:
                raise RuntimeError("COALESCE query returned None in save_message_and_update_model")
            volume3 = int(row3[0] or 0)
            cursor2 = await db.execute(
                "SELECT COALESCE(SUM(cnt), 0) FROM transitions WHERE chat_id = ?",
                (chat_id,),
            )
            row2 = await cursor2.fetchone()
            if row2 is None:
                raise RuntimeError("COALESCE query returned None in save_message_and_update_model")
            volume2 = int(row2[0] or 0)

            await db.commit()
            return volume3 if volume3 > 0 else volume2

    # --- Делегаты к MarkovRepo (сохраняем публичный API) ---

    async def get_starts(self, chat_id: int) -> list[tuple[str, str, int]]:
        if self.markov is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.markov.get_starts(chat_id)

    async def get_starts3(self, chat_id: int) -> list[tuple[str, str, str, int]]:
        if self.markov is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.markov.get_starts3(chat_id)

    async def get_start_if_exists(
        self, chat_id: int, w1: str, w2: str
    ) -> Optional[tuple[str, str, int]]:
        if self.markov is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.markov.get_start_if_exists(chat_id, w1, w2)

    async def get_start3_if_exists(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> Optional[tuple[str, str, str, int]]:
        if self.markov is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.markov.get_start3_if_exists(chat_id, w1, w2, w3)

    async def get_transitions(
        self, chat_id: int, w1: str, w2: str
    ) -> list[tuple[str, int]]:
        if self.markov is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.markov.get_transitions(chat_id, w1, w2)

    async def get_transitions3(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> list[tuple[str, int]]:
        if self.markov is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.markov.get_transitions3(chat_id, w1, w2, w3)

    async def get_transitions1(self, chat_id: int, w1: str) -> list[tuple[str, int]]:
        if self.markov is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.markov.get_transitions1(chat_id, w1)

    async def get_chat_token_volume(self, chat_id: int) -> int:
        if self.markov is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.markov.get_chat_token_volume(chat_id)

    # --- Делегаты к MessagesRepo ---

    async def message_exists(self, chat_id: int, text: str) -> bool:
        if self.messages is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.messages.exists(chat_id, text)

    async def get_recent_normalized_messages(
        self, chat_id: int, limit: int
    ) -> list[str]:
        if self.messages is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.messages.get_recent_normalized(chat_id, limit)

    # --- Делегаты к ChatMembersRepo ---

    async def upsert_chat_member(
        self,
        *,
        chat_hash: str,
        user_hash: str,
        encrypted_user_id: str,
        encrypted_username: str,
        encrypted_display_name: str,
    ) -> None:
        if self.chat_members is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        await self.chat_members.upsert(
            chat_hash=chat_hash,
            user_hash=user_hash,
            encrypted_user_id=encrypted_user_id,
            encrypted_username=encrypted_username,
            encrypted_display_name=encrypted_display_name,
        )

    async def remove_chat_member(self, chat_hash: str, user_hash: str) -> None:
        if self.chat_members is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        await self.chat_members.remove(chat_hash, user_hash)

    async def get_chat_members(self, chat_hash: str) -> list[dict[str, object]]:
        if self.chat_members is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.chat_members.list_members(chat_hash)

    async def consume_pivo_daily_call(
        self,
        *,
        chat_hash: str,
        user_hash: str,
        usage_day: str,
        limit: int,
    ) -> tuple[bool, int]:
        if self.pivo_usage is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        return await self.pivo_usage.consume_daily_call(
            chat_hash=chat_hash,
            user_hash=user_hash,
            usage_day=usage_day,
            limit=limit,
        )

    async def refund_pivo_daily_call(
        self,
        *,
        chat_hash: str,
        user_hash: str,
        usage_day: str,
    ) -> None:
        if self.pivo_usage is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        await self.pivo_usage.refund_daily_call(
            chat_hash=chat_hash,
            user_hash=user_hash,
            usage_day=usage_day,
        )

    async def cleanup_pivo_daily_usage(
        self,
        *,
        retention_days: int = PIVO_DAILY_USAGE_RETENTION_DAYS,
        today: date | None = None,
    ) -> int:
        """Deletes /pivo daily quota rows older than retention_days."""
        if retention_days < 0:
            raise ValueError("retention_days must be non-negative")
        if self.pivo_usage is None:
            raise RuntimeError("Database not initialized: call await Database.init() first")
        current_day = today or datetime.now(UTC).date()
        cutoff_day = (current_day - timedelta(days=retention_days)).isoformat()
        return await self.pivo_usage.delete_usage_before(cutoff_day)

    # --- Кросс-доменные операции ---

    async def _fetch_int(
        self, db: aiosqlite.Connection, sql: str, params: tuple[object, ...]
    ) -> int:
        row = await (await db.execute(sql, params)).fetchone()
        return int(row[0] or 0) if row else 0

    async def get_stats(self, chat_id: int) -> dict[str, int]:
        p = (chat_id,)
        async with self._lock:
            db = await self._get_conn()
            f = self._fetch_int

            msg_count    = await f(db, "SELECT COUNT(*) FROM messages WHERE chat_id = ?", p)
            starts2      = await f(db, "SELECT COUNT(*) FROM starts WHERE chat_id = ?", p)
            starts3      = await f(db, "SELECT COUNT(*) FROM starts3 WHERE chat_id = ?", p)
            trans2_count = await f(db, "SELECT COUNT(*) FROM transitions WHERE chat_id = ?", p)
            trans3_count = await f(db, "SELECT COUNT(*) FROM transitions3 WHERE chat_id = ?", p)
            trans1_count = await f(db, "SELECT COUNT(*) FROM transitions1 WHERE chat_id = ?", p)
            volume2 = await f(
                db, "SELECT COALESCE(SUM(cnt), 0) FROM transitions WHERE chat_id = ?", p
            )
            volume3 = await f(
                db, "SELECT COALESCE(SUM(cnt), 0) FROM transitions3 WHERE chat_id = ?", p
            )
            volume1 = await f(
                db, "SELECT COALESCE(SUM(cnt), 0) FROM transitions1 WHERE chat_id = ?", p
            )

        return {
            "messages":     msg_count,
            "starts2":      starts2,
            "starts3":      starts3,
            "transitions2": trans2_count,
            "transitions3": trans3_count,
            "transitions1": trans1_count,
            "volume2":      volume2,
            "volume3":      volume3,
            "volume1":      volume1,
            "volume":       volume3 if volume3 > 0 else volume2,
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
