from __future__ import annotations

import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.infrastructure.database import Database

CHAT = 4242


class TestChatEmojiStatsRepo(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_emoji_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def test_empty_chat_has_no_stats(self) -> None:
        self.assertEqual(await self.db.get_chat_emoji_stats(CHAT), {})

    async def test_bump_accumulates_across_calls(self) -> None:
        await self.db.record_chat_emojis(CHAT, {"😂": 2, "🔥": 1})
        await self.db.record_chat_emojis(CHAT, {"😂": 3})

        stats = await self.db.get_chat_emoji_stats(CHAT)
        self.assertEqual(stats, {"😂": 5, "🔥": 1})

    async def test_bump_ignores_non_positive_and_empty(self) -> None:
        await self.db.record_chat_emojis(CHAT, {"😂": 0, "🔥": -2})
        await self.db.record_chat_emojis(CHAT, {})
        self.assertEqual(await self.db.get_chat_emoji_stats(CHAT), {})

    async def test_stats_are_per_chat(self) -> None:
        await self.db.record_chat_emojis(CHAT, {"😂": 1})
        await self.db.record_chat_emojis(999, {"🔥": 1})
        self.assertEqual(await self.db.get_chat_emoji_stats(CHAT), {"😂": 1})

    async def test_clear_chat_removes_emoji_stats(self) -> None:
        await self.db.record_chat_emojis(CHAT, {"😂": 4})
        await self.db.clear_chat(CHAT)
        self.assertEqual(await self.db.get_chat_emoji_stats(CHAT), {})

    async def test_decay_halves_stale_rows_and_drops_zeros(self) -> None:
        await self.db.record_chat_emojis(CHAT, {"😂": 4, "🔥": 1})
        # Decay as if 30 days have passed: 4→2, 1→0 (dropped).
        future = datetime.now(UTC) + timedelta(days=30)
        deleted = await self.db.decay_chat_emoji_stats(now=future)

        stats = await self.db.get_chat_emoji_stats(CHAT)
        self.assertEqual(stats, {"😂": 2})
        self.assertEqual(deleted, 1)

    async def test_decay_leaves_fresh_rows_untouched(self) -> None:
        await self.db.record_chat_emojis(CHAT, {"😂": 4})
        # Decay run "now" — the row was just bumped, so it is not yet stale.
        await self.db.decay_chat_emoji_stats()
        self.assertEqual(await self.db.get_chat_emoji_stats(CHAT), {"😂": 4})


if __name__ == "__main__":
    unittest.main()
