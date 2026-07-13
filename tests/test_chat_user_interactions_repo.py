from __future__ import annotations

import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.infrastructure.database import Database
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
