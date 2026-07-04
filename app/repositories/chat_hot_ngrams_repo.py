from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable

import aiosqlite

ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]


class ChatHotNgramsRepo:
    """Sliding-window content n-gram counts for the L1 running-jokes channel.

    Keyed by raw ``chat_id`` to match the Markov model tables; per-chat
    aggregate only (no author). Bigrams are stored with ``w3 = ''``. Hotness
    is the window count's share of the all-time count in ``transitions`` /
    ``transitions1``: a spike means the chat picked the phrase up recently.
    """

    def __init__(self, conn_provider: ConnProvider, lock: asyncio.Lock) -> None:
        self._conn_provider = conn_provider
        self._lock = lock

    async def bump(self, chat_id: int, ngrams: Iterable[tuple[str, ...]]) -> None:
        """Add one occurrence per listed n-gram (no-op if empty)."""
        counts = Counter(ngram for ngram in ngrams if len(ngram) in (2, 3))
        if not counts:
            return
        rows = [
            (chat_id, ngram[0], ngram[1], ngram[2] if len(ngram) == 3 else "", n)
            for ngram, n in counts.items()
        ]
        async with self._lock:
            db = await self._conn_provider()
            await db.executemany(
                """
                INSERT INTO chat_hot_ngrams(chat_id, w1, w2, w3, cnt)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, w1, w2, w3) DO UPDATE SET
                    cnt = cnt + excluded.cnt,
                    updated_at = datetime('now')
                """,
                rows,
            )
            await db.commit()

    async def get_hot(
        self,
        chat_id: int,
        *,
        min_count: int,
        recency_share: float,
        limit: int = 8,
    ) -> list[tuple[str, ...]]:
        """Top n-grams whose window count is a big share of their all-time count.

        ``recency_share`` in (0..1]: 0.5 means at least half of all recorded
        occurrences happened inside the current window. Missing long-term rows
        fall back to the window count itself (share 1.0) so a brand-new meme
        still qualifies.
        """
        query = """
            SELECT h.w1, h.w2, h.w3, h.cnt
            FROM chat_hot_ngrams h
            LEFT JOIN transitions t
              ON t.chat_id = h.chat_id
             AND t.w1 = h.w1 AND t.w2 = h.w2 AND t.w3 = h.w3
            WHERE h.chat_id = ? AND h.w3 <> '' AND h.cnt >= ?
              AND h.cnt * 1.0 / MAX(COALESCE(t.cnt, h.cnt), h.cnt) >= ?
            UNION ALL
            SELECT h.w1, h.w2, h.w3, h.cnt
            FROM chat_hot_ngrams h
            LEFT JOIN transitions1 t1
              ON t1.chat_id = h.chat_id AND t1.w1 = h.w1 AND t1.w2 = h.w2
            WHERE h.chat_id = ? AND h.w3 = '' AND h.cnt >= ?
              AND h.cnt * 1.0 / MAX(COALESCE(t1.cnt, h.cnt), h.cnt) >= ?
            ORDER BY 4 DESC
            LIMIT ?
        """
        params = (
            chat_id,
            min_count,
            recency_share,
            chat_id,
            min_count,
            recency_share,
            limit,
        )
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
        result: list[tuple[str, ...]] = []
        for w1, w2, w3, _cnt in rows:
            if w3:
                result.append((str(w1), str(w2), str(w3)))
            else:
                result.append((str(w1), str(w2)))
        return result

    async def decay_stale(self, cutoff_iso: str) -> int:
        """Halve counts of rows not bumped since ``cutoff_iso``; purge zeros.

        Same contract as ``ChatEmojiStatsRepo.decay_stale``: halved rows get a
        fresh clock so they will not re-decay for another window; returns the
        number of purged rows.
        """
        async with self._lock:
            db = await self._conn_provider()
            await db.execute(
                """
                UPDATE chat_hot_ngrams
                SET cnt = cnt / 2, updated_at = datetime('now')
                WHERE updated_at < ?
                """,
                (cutoff_iso,),
            )
            cursor = await db.execute("DELETE FROM chat_hot_ngrams WHERE cnt <= 0")
            await db.commit()
            return max(0, cursor.rowcount)
