from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

import aiosqlite

ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]


class BaseRepo:
    """Shared connection/lock plumbing for the SQLite-backed repositories.

    Every repository serialises access through the one shared ``asyncio.Lock``
    and resolves the live connection lazily via ``conn_provider``. The helpers
    below centralise that boilerplate so subclasses only carry SQL and
    row-mapping. Multi-statement methods use the ``_connection`` /
    ``_transaction`` context managers directly.
    """

    def __init__(self, conn_provider: ConnProvider, lock: asyncio.Lock) -> None:
        self._conn_provider = conn_provider
        self._lock = lock

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Hold the shared lock and yield the live connection (no auto-commit)."""
        async with self._lock:
            yield await self._conn_provider()

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Hold the lock, yield the connection, and commit on clean exit."""
        async with self._lock:
            db = await self._conn_provider()
            yield db
            await db.commit()

    async def _fetch_all(
        self, sql: str, params: tuple[object, ...]
    ) -> list[aiosqlite.Row]:
        async with self._connection() as db:
            cursor = await db.execute(sql, params)
            return list(await cursor.fetchall())

    async def _fetch_one(
        self, sql: str, params: tuple[object, ...]
    ) -> aiosqlite.Row | None:
        async with self._connection() as db:
            cursor = await db.execute(sql, params)
            return await cursor.fetchone()

    async def _execute(self, sql: str, params: tuple[object, ...]) -> int:
        """Run one write statement in its own transaction; return its rowcount."""
        async with self._transaction() as db:
            cursor = await db.execute(sql, params)
            return cursor.rowcount

    async def _execute_many(
        self, sql: str, seq_params: Sequence[tuple[object, ...]]
    ) -> None:
        """Run one batched write statement in its own transaction."""
        async with self._transaction() as db:
            await db.executemany(sql, seq_params)


class DecayableCountsRepo(BaseRepo):
    """BaseRepo for the flavor tables whose ``cnt`` decays when a chat goes quiet.

    The emoji-stats and hot-ngram tables share the same
    ``(..., cnt, updated_at)`` shape and the same lazy-decay contract, so the
    halving/purge statement lives here rather than in the generic base.
    """

    async def _decay_stale(self, table: str, cutoff_iso: str) -> int:
        """Halve counts of rows not touched since ``cutoff_iso``; purge zeros.

        Stale rows are halved and their ``updated_at`` clock reset so they will
        not re-decay for another window, and rows reaching 0 are removed.
        Returns the number of purged rows. ``table`` is a hardcoded caller
        constant, never user input.
        """
        async with self._transaction() as db:
            await db.execute(
                f"""
                UPDATE {table}
                SET cnt = cnt / 2, updated_at = datetime('now')
                WHERE updated_at < ?
                """,
                (cutoff_iso,),
            )
            cursor = await db.execute(f"DELETE FROM {table} WHERE cnt <= 0")
            return max(0, cursor.rowcount)
