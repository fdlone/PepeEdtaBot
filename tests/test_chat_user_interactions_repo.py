from __future__ import annotations

import time
import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.infrastructure.database import (
    CHAT_USER_INTERACTION_DECAY_DAYS,
    FLAVOR_DECAY_INTERVAL_SEC,
    Database,
)
from app.repositories import ChatUserInteractionsRepo

CHAT = 4242
OTHER_CHAT = 999
USER_HASH = "a" * 64
OTHER_HASH = "b" * 64


def _cutoff_after(days: int) -> str:
    """Cutoff string as if ``days`` days have passed (matches _decay_cutoff)."""
    return (datetime.now(UTC) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


class TestChatUserInteractionsRepo(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_interactions_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        assert self.db.chat_user_interactions is not None
        self.repo: ChatUserInteractionsRepo = self.db.chat_user_interactions

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def test_unknown_user_has_zero_count(self) -> None:
        self.assertEqual(await self.repo.get_count(CHAT, USER_HASH), 0)

    async def test_bump_accumulates(self) -> None:
        await self.repo.bump(CHAT, USER_HASH)
        await self.repo.bump(CHAT, USER_HASH)
        await self.repo.bump(CHAT, USER_HASH)
        self.assertEqual(await self.repo.get_count(CHAT, USER_HASH), 3)

    async def test_counts_are_per_user(self) -> None:
        await self.repo.bump(CHAT, USER_HASH)
        await self.repo.bump(CHAT, OTHER_HASH)
        await self.repo.bump(CHAT, OTHER_HASH)

        self.assertEqual(await self.repo.get_count(CHAT, USER_HASH), 1)
        self.assertEqual(await self.repo.get_count(CHAT, OTHER_HASH), 2)

    async def test_counts_are_per_chat(self) -> None:
        await self.repo.bump(CHAT, USER_HASH)
        await self.repo.bump(OTHER_CHAT, USER_HASH)
        await self.repo.bump(OTHER_CHAT, USER_HASH)

        self.assertEqual(await self.repo.get_count(CHAT, USER_HASH), 1)
        self.assertEqual(await self.repo.get_count(OTHER_CHAT, USER_HASH), 2)

    async def test_decay_halves_stale_rows_and_drops_zeros(self) -> None:
        for _ in range(4):
            await self.repo.bump(CHAT, USER_HASH)
        await self.repo.bump(CHAT, OTHER_HASH)

        deleted = await self.repo.decay_stale(_cutoff_after(30))

        self.assertEqual(await self.repo.get_count(CHAT, USER_HASH), 2)
        self.assertEqual(await self.repo.get_count(CHAT, OTHER_HASH), 0)
        self.assertEqual(deleted, 1)

    async def test_decay_leaves_fresh_rows_untouched(self) -> None:
        for _ in range(4):
            await self.repo.bump(CHAT, USER_HASH)

        deleted = await self.repo.decay_stale(_cutoff_after(-1))

        self.assertEqual(await self.repo.get_count(CHAT, USER_HASH), 4)
        self.assertEqual(deleted, 0)


class TestDatabaseUserInteractionDelegates(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_interactions_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def test_record_and_get_roundtrip(self) -> None:
        await self.db.chat_user_interactions.bump(CHAT, USER_HASH)
        await self.db.chat_user_interactions.bump(CHAT, USER_HASH)

        self.assertEqual(
            await self.db.chat_user_interactions.get_count(CHAT, USER_HASH), 2
        )
        self.assertEqual(
            await self.db.chat_user_interactions.get_count(CHAT, OTHER_HASH), 0
        )

    async def test_decay_window_is_30_days(self) -> None:
        self.assertEqual(CHAT_USER_INTERACTION_DECAY_DAYS, 30)
        await self.db.chat_user_interactions.bump(CHAT, USER_HASH)
        await self.db.chat_user_interactions.bump(CHAT, USER_HASH)

        # 29 days later the row is still fresh; at 31 days it halves.
        fresh = datetime.now(UTC) + timedelta(days=29)
        self.assertEqual(await self.db.decay_chat_user_interactions(now=fresh), 0)
        self.assertEqual(
            await self.db.chat_user_interactions.get_count(CHAT, USER_HASH), 2
        )

        stale = datetime.now(UTC) + timedelta(days=31)
        await self.db.decay_chat_user_interactions(now=stale)
        self.assertEqual(
            await self.db.chat_user_interactions.get_count(CHAT, USER_HASH), 1
        )

    async def test_clear_chat_removes_interactions(self) -> None:
        await self.db.chat_user_interactions.bump(CHAT, USER_HASH)
        await self.db.chat_user_interactions.bump(OTHER_CHAT, USER_HASH)

        await self.db.clear_chat(CHAT)

        self.assertEqual(
            await self.db.chat_user_interactions.get_count(CHAT, USER_HASH), 0
        )
        self.assertEqual(
            await self.db.chat_user_interactions.get_count(OTHER_CHAT, USER_HASH), 1
        )

    async def test_lazy_decay_covers_interactions(self) -> None:
        # decay_flavor_stats_if_due must run the interactions decay too: with
        # the monotonic stamp aged past the interval and rows made stale via
        # a rewound updated_at, one lazy call halves them.
        await self.db.chat_user_interactions.bump(CHAT, USER_HASH)
        await self.db.chat_user_interactions.bump(CHAT, USER_HASH)
        assert self.db._conn is not None
        await self.db._conn.execute(
            "UPDATE chat_user_interactions SET updated_at = ?",
            ((datetime.now(UTC) - timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S"),),
        )
        await self.db._conn.commit()
        self.db._last_flavor_decay_monotonic = (
            time.monotonic() - FLAVOR_DECAY_INTERVAL_SEC - 1.0
        )

        self.assertTrue(await self.db.decay_flavor_stats_if_due())
        self.assertEqual(
            await self.db.chat_user_interactions.get_count(CHAT, USER_HASH), 1
        )

    async def test_stats_aggregate_counts_people_max_and_threshold(self) -> None:
        # Three people with 1, 3 and 5 answered mentions; threshold 3.
        for _ in range(1):
            await self.db.chat_user_interactions.bump(CHAT, "b" * 64)
        for _ in range(3):
            await self.db.chat_user_interactions.bump(CHAT, "c" * 64)
        for _ in range(5):
            await self.db.chat_user_interactions.bump(CHAT, "d" * 64)
        # A neighbouring chat must not leak into the aggregate.
        await self.db.chat_user_interactions.bump(OTHER_CHAT, "e" * 64)

        people, max_count, at_or_above = await self.db.chat_user_interactions.get_stats(
            CHAT, 3
        )

        self.assertEqual(people, 3)
        self.assertEqual(max_count, 5)
        self.assertEqual(at_or_above, 2)

    async def test_stats_on_empty_chat_are_zeros_not_an_error(self) -> None:
        self.assertEqual(
            await self.db.chat_user_interactions.get_stats(OTHER_CHAT, 25), (0, 0, 0)
        )

    async def test_stats_do_not_change_the_counters(self) -> None:
        await self.db.chat_user_interactions.bump(CHAT, USER_HASH)

        before = await self.db.chat_user_interactions.get_count(CHAT, USER_HASH)
        await self.db.chat_user_interactions.get_stats(CHAT, 1)
        after = await self.db.chat_user_interactions.get_count(CHAT, USER_HASH)

        self.assertEqual(before, after)
