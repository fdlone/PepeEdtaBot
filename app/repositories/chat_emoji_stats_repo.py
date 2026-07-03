from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping

import aiosqlite

ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]


class ChatEmojiStatsRepo:
    """Per-chat emoji frequency for the M3 emoji channel.

    Keyed by raw ``chat_id`` to match the Markov model tables. Emojis are a
    per-chat aggregate (no author), so this stays within the existing privacy
    contour and is wiped together with the model in ``clear_chat``.
    """

    def __init__(self, conn_provider: ConnProvider, lock: asyncio.Lock) -> None:
        self._conn_provider = conn_provider
        self._lock = lock

    async def bump(self, chat_id: int, counts: Mapping[str, int]) -> None:
        """Add ``counts`` to the chat's emoji frequencies (no-op if empty)."""
        positive = {emoji: n for emoji, n in counts.items() if n > 0}
        if not positive:
            return
        async with self._lock:
            db = await self._conn_provider()
            await db.executemany(
                """
                INSERT INTO chat_emoji_stats(chat_id, emoji, cnt)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id, emoji) DO UPDATE SET
                    cnt = cnt + excluded.cnt,
                    updated_at = datetime('now')
                """,
                [(chat_id, emoji, n) for emoji, n in positive.items()],
            )
            await db.commit()

    async def get_stats(self, chat_id: int) -> dict[str, int]:
        """Return {emoji: count} for a chat (empty if none learned)."""
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(
                "SELECT emoji, cnt FROM chat_emoji_stats WHERE chat_id = ?",
                (chat_id,),
            )
            rows = await cursor.fetchall()
        return {str(row[0]): int(row[1] or 0) for row in rows}

    async def decay_stale(self, cutoff_iso: str) -> int:
        """Halve counts of rows not bumped since ``cutoff_iso`` so dead memes fade.

        Halved rows have their clock reset so they will not re-decay for another
        window; rows that reach 0 are removed. Returns the number of rows deleted.
        """
        async with self._lock:
            db = await self._conn_provider()
            await db.execute(
                """
                UPDATE chat_emoji_stats
                SET cnt = cnt / 2, updated_at = datetime('now')
                WHERE updated_at < ?
                """,
                (cutoff_iso,),
            )
            cursor = await db.execute(
                "DELETE FROM chat_emoji_stats WHERE cnt <= 0"
            )
            await db.commit()
            return max(0, cursor.rowcount)
