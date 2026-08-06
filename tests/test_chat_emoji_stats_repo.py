from __future__ import annotations

import time
import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.infrastructure.database import FLAVOR_DECAY_INTERVAL_SEC, Database

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

    async def test_lazy_decay_not_due_right_after_init(self) -> None:
        # init() already ran the decay and stamped the clock.
        self.assertFalse(await self.db.decay_flavor_stats_if_due())

    async def test_lazy_decay_runs_once_after_interval(self) -> None:
        # Pretend the last decay happened more than a day ago.
        self.db._last_flavor_decay_monotonic = (
            time.monotonic() - FLAVOR_DECAY_INTERVAL_SEC - 1.0
        )
        self.assertTrue(await self.db.decay_flavor_stats_if_due())
        # The clock was re-stamped: an immediate second call is a no-op.
        self.assertFalse(await self.db.decay_flavor_stats_if_due())

    async def test_failed_decay_does_not_postpone_the_next_attempt(self) -> None:
        """Отметка ставится после выполнения, а не до.

        Иначе сбой внутри decay засчитывался как выполненный прогон: окно
        «горячести» замирало до следующих суток, хотя ни одна строка не была
        обработана.
        """
        from unittest.mock import patch

        stale = time.monotonic() - FLAVOR_DECAY_INTERVAL_SEC - 1.0
        self.db._last_flavor_decay_monotonic = stale

        with patch.object(
            self.db, "decay_chat_hot_ngrams", side_effect=RuntimeError("disk gone")
        ):
            with self.assertRaises(RuntimeError):
                await self.db.decay_flavor_stats_if_due()

        self.assertEqual(self.db._last_flavor_decay_monotonic, stale)
        # Следующая попытка — сразу, а не через сутки.
        self.assertTrue(await self.db.decay_flavor_stats_if_due())


if __name__ == "__main__":
    unittest.main()
