from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Optional, TypeVar

import aiosqlite

from app.config.defaults import (
    MESSAGES_RETENTION_PER_CHAT,
    SQLITE_BUSY_TIMEOUT_MS,
    SQLITE_WAL_AUTOCHECKPOINT_PAGES,
)
from app.core.text import sanitize_text
from app.infrastructure import migrator
from app.repositories import (
    ChatEmojiStatsRepo,
    ChatMembersRepo,
    MarkovRepo,
    MessagesRepo,
    PivoPoolUsageRepo,
    PivoUsageRepo,
)

PIVO_DAILY_USAGE_RETENTION_DAYS = 7
CHAT_EMOJI_DECAY_DAYS = 7

_RepoT = TypeVar("_RepoT")


class Database:
    """Фасад над SQLite: владеет соединением, миграциями и репозиториями."""

    def __init__(
        self,
        path: str,
        *,
        messages_retention_per_chat: int = MESSAGES_RETENTION_PER_CHAT,
        busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
        wal_autocheckpoint_pages: int = SQLITE_WAL_AUTOCHECKPOINT_PAGES,
    ) -> None:
        if messages_retention_per_chat < 1:
            raise ValueError("messages_retention_per_chat must be at least 1")
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        if wal_autocheckpoint_pages < 0:
            raise ValueError("wal_autocheckpoint_pages must be non-negative")
        self.path = path
        self.messages_retention_per_chat = messages_retention_per_chat
        self.busy_timeout_ms = busy_timeout_ms
        self.wal_autocheckpoint_pages = wal_autocheckpoint_pages
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self.markov: Optional[MarkovRepo] = None
        self.messages: Optional[MessagesRepo] = None
        self.chat_members: Optional[ChatMembersRepo] = None
        self.pivo_usage: Optional[PivoUsageRepo] = None
        self.pivo_pool_usage: Optional[PivoPoolUsageRepo] = None
        self.chat_emoji_stats: Optional[ChatEmojiStatsRepo] = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not initialized. Call init() first.")
        return self._conn

    @staticmethod
    def _require(repo: Optional[_RepoT]) -> _RepoT:
        """Возвращает репозиторий или бросает, если init() ещё не вызван."""
        if repo is None:
            raise RuntimeError(
                "Database not initialized: call await Database.init() first"
            )
        return repo

    async def init(self) -> None:
        if self._conn is not None:
            return

        self._conn = await aiosqlite.connect(self.path)
        db = await self._get_conn()

        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms};")
        await db.execute(
            f"PRAGMA wal_autocheckpoint = {self.wal_autocheckpoint_pages};"
        )
        await db.execute("PRAGMA foreign_keys=ON;")

        await migrator.run(db)

        self.markov = MarkovRepo(self._get_conn, self._lock)
        self.messages = MessagesRepo(self._get_conn, self._lock)
        self.chat_members = ChatMembersRepo(self._get_conn, self._lock)
        self.pivo_usage = PivoUsageRepo(self._get_conn, self._lock)
        self.pivo_pool_usage = PivoPoolUsageRepo(self._get_conn, self._lock)
        self.chat_emoji_stats = ChatEmojiStatsRepo(self._get_conn, self._lock)
        await self.cleanup_pivo_daily_usage()
        await self.decay_chat_emoji_stats()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        self.markov = None
        self.messages = None
        self.chat_members = None
        self.pivo_usage = None
        self.pivo_pool_usage = None
        self.chat_emoji_stats = None

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
            await db.execute(
                """
                DELETE FROM messages
                WHERE chat_id = ?
                  AND id <= (
                      SELECT id
                      FROM messages
                      WHERE chat_id = ?
                      ORDER BY id DESC
                      LIMIT 1 OFFSET ?
                  )
                """,
                (
                    chat_id,
                    chat_id,
                    self.messages_retention_per_chat,
                ),
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

            # Maintain the per-chat model volume incrementally instead of
            # re-summing the whole transitions tables on every message (audit
            # D2). The delta is exactly the number of transition occurrences
            # this message contributed (sum of the per-message counters).
            volume2_delta = sum(trans2_counter.values())
            volume3_delta = sum(trans3_counter.values())
            await db.execute(
                """
                INSERT INTO chat_model_volume (chat_id, volume2, volume3)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    volume2 = volume2 + excluded.volume2,
                    volume3 = volume3 + excluded.volume3
                """,
                (chat_id, volume2_delta, volume3_delta),
            )
            cursor = await db.execute(
                "SELECT volume2, volume3 FROM chat_model_volume WHERE chat_id = ?",
                (chat_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError(
                    "chat_model_volume row missing after upsert in "
                    "save_message_and_update_model"
                )
            volume2 = int(row[0] or 0)
            volume3 = int(row[1] or 0)

            await db.commit()
            return volume3 if volume3 > 0 else volume2

    # --- Делегаты к MarkovRepo (сохраняем публичный API) ---

    async def get_starts(self, chat_id: int) -> list[tuple[str, str, int]]:
        return await self._require(self.markov).get_starts(chat_id)

    async def get_starts3(self, chat_id: int) -> list[tuple[str, str, str, int]]:
        return await self._require(self.markov).get_starts3(chat_id)

    async def get_start_if_exists(
        self, chat_id: int, w1: str, w2: str
    ) -> Optional[tuple[str, str, int]]:
        return await self._require(self.markov).get_start_if_exists(chat_id, w1, w2)

    async def get_start3_if_exists(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> Optional[tuple[str, str, str, int]]:
        return await self._require(self.markov).get_start3_if_exists(
            chat_id, w1, w2, w3
        )

    async def get_transitions(
        self, chat_id: int, w1: str, w2: str
    ) -> list[tuple[str, int]]:
        return await self._require(self.markov).get_transitions(chat_id, w1, w2)

    async def get_transitions3(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> list[tuple[str, int]]:
        return await self._require(self.markov).get_transitions3(chat_id, w1, w2, w3)

    async def get_transitions1(self, chat_id: int, w1: str) -> list[tuple[str, int]]:
        return await self._require(self.markov).get_transitions1(chat_id, w1)

    async def get_markov_states(
        self,
        chat_id: int,
        order: int,
    ) -> list[tuple[tuple[str, ...], int]]:
        return await self._require(self.markov).get_states(chat_id, order)

    async def get_chat_token_volume(self, chat_id: int) -> int:
        return await self._require(self.markov).get_chat_token_volume(chat_id)

    # --- Делегаты к MessagesRepo ---

    async def message_exists(self, chat_id: int, text: str) -> bool:
        return await self._require(self.messages).exists(chat_id, text)

    async def get_recent_normalized_messages(
        self, chat_id: int, limit: int
    ) -> list[str]:
        return await self._require(self.messages).get_recent_normalized(chat_id, limit)

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
        await self._require(self.chat_members).upsert(
            chat_hash=chat_hash,
            user_hash=user_hash,
            encrypted_user_id=encrypted_user_id,
            encrypted_username=encrypted_username,
            encrypted_display_name=encrypted_display_name,
        )

    async def remove_chat_member(self, chat_hash: str, user_hash: str) -> None:
        await self._require(self.chat_members).remove(chat_hash, user_hash)

    async def get_chat_members(self, chat_hash: str) -> list[dict[str, object]]:
        return await self._require(self.chat_members).list_members(chat_hash)

    async def consume_pivo_daily_call(
        self,
        *,
        chat_hash: str,
        user_hash: str,
        usage_day: str,
        limit: int,
    ) -> tuple[bool, int]:
        return await self._require(self.pivo_usage).consume_daily_call(
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
        await self._require(self.pivo_usage).refund_daily_call(
            chat_hash=chat_hash,
            user_hash=user_hash,
            usage_day=usage_day,
        )

    async def get_pivo_pool_usage(self, chat_hash: str) -> dict[str, tuple[int, ...]]:
        return await self._require(self.pivo_pool_usage).get_recent(chat_hash)

    async def record_pivo_pool_usage(
        self,
        chat_hash: str,
        picks: Mapping[str, int],
        *,
        keep: int,
    ) -> None:
        await self._require(self.pivo_pool_usage).record(chat_hash, picks, keep=keep)

    # --- Делегаты к ChatEmojiStatsRepo (M3 emoji channel) ---

    async def record_chat_emojis(
        self, chat_id: int, counts: Mapping[str, int]
    ) -> None:
        await self._require(self.chat_emoji_stats).bump(chat_id, counts)

    async def get_chat_emoji_stats(self, chat_id: int) -> dict[str, int]:
        return await self._require(self.chat_emoji_stats).get_stats(chat_id)

    async def decay_chat_emoji_stats(
        self,
        *,
        decay_days: int = CHAT_EMOJI_DECAY_DAYS,
        now: datetime | None = None,
    ) -> int:
        """Halve emoji counts not bumped within ``decay_days`` so dead memes fade.

        Runs once at init (no scheduler exists), mirroring
        ``cleanup_pivo_daily_usage``. Stale rows are halved and their clock reset;
        rows reaching 0 are removed. Returns the number of rows deleted.
        """
        if decay_days < 0:
            raise ValueError("decay_days must be non-negative")
        current = now or datetime.now(UTC)
        # Match SQLite's datetime('now') text format ("YYYY-MM-DD HH:MM:SS", UTC)
        # so the string comparison in decay_stale is well-defined.
        cutoff = (current - timedelta(days=decay_days)).strftime("%Y-%m-%d %H:%M:%S")
        return await self._require(self.chat_emoji_stats).decay_stale(cutoff)

    async def cleanup_pivo_daily_usage(
        self,
        *,
        retention_days: int = PIVO_DAILY_USAGE_RETENTION_DAYS,
        today: date | None = None,
    ) -> int:
        """Deletes /pivo daily quota rows older than retention_days."""
        if retention_days < 0:
            raise ValueError("retention_days must be non-negative")
        pivo_usage = self._require(self.pivo_usage)
        current_day = today or datetime.now(UTC).date()
        cutoff_day = (current_day - timedelta(days=retention_days)).isoformat()
        return await pivo_usage.delete_usage_before(cutoff_day)

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
            await db.execute("DELETE FROM chat_model_volume WHERE chat_id = ?", (chat_id,))
            await db.execute("DELETE FROM chat_emoji_stats WHERE chat_id = ?", (chat_id,))
            await db.commit()
