from __future__ import annotations

import time
import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.infrastructure.database import (
    FLAVOR_DECAY_INTERVAL_SEC,
    FLAVOR_DECAY_RETRY_INTERVAL_SEC,
    Database,
    MaintenanceOutcome,
)

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
        self.assertEqual(await self.db.chat_emoji_stats.get_stats(CHAT), {})

    async def test_bump_accumulates_across_calls(self) -> None:
        await self.db.chat_emoji_stats.bump(CHAT, {"😂": 2, "🔥": 1})
        await self.db.chat_emoji_stats.bump(CHAT, {"😂": 3})

        stats = await self.db.chat_emoji_stats.get_stats(CHAT)
        self.assertEqual(stats, {"😂": 5, "🔥": 1})

    async def test_bump_ignores_non_positive_and_empty(self) -> None:
        await self.db.chat_emoji_stats.bump(CHAT, {"😂": 0, "🔥": -2})
        await self.db.chat_emoji_stats.bump(CHAT, {})
        self.assertEqual(await self.db.chat_emoji_stats.get_stats(CHAT), {})

    async def test_stats_are_per_chat(self) -> None:
        await self.db.chat_emoji_stats.bump(CHAT, {"😂": 1})
        await self.db.chat_emoji_stats.bump(999, {"🔥": 1})
        self.assertEqual(await self.db.chat_emoji_stats.get_stats(CHAT), {"😂": 1})

    async def test_clear_chat_removes_emoji_stats(self) -> None:
        await self.db.chat_emoji_stats.bump(CHAT, {"😂": 4})
        await self.db.clear_chat(CHAT)
        self.assertEqual(await self.db.chat_emoji_stats.get_stats(CHAT), {})

    async def test_decay_halves_stale_rows_and_drops_zeros(self) -> None:
        await self.db.chat_emoji_stats.bump(CHAT, {"😂": 4, "🔥": 1})
        # Decay as if 30 days have passed: 4→2, 1→0 (dropped).
        future = datetime.now(UTC) + timedelta(days=30)
        deleted = await self.db.decay_chat_emoji_stats(now=future)

        stats = await self.db.chat_emoji_stats.get_stats(CHAT)
        self.assertEqual(stats, {"😂": 2})
        self.assertEqual(deleted, 1)

    async def test_decay_leaves_fresh_rows_untouched(self) -> None:
        await self.db.chat_emoji_stats.bump(CHAT, {"😂": 4})
        # Decay run "now" — the row was just bumped, so it is not yet stale.
        await self.db.decay_chat_emoji_stats()
        self.assertEqual(await self.db.chat_emoji_stats.get_stats(CHAT), {"😂": 4})

    async def test_lazy_decay_not_due_right_after_init(self) -> None:
        # init() already ran the decay and stamped the clock.
        self.assertIs(
            await self.db.decay_flavor_stats_if_due(), MaintenanceOutcome.SKIPPED
        )

    async def test_lazy_decay_runs_once_after_interval(self) -> None:
        # Pretend the last decay happened more than a day ago.
        self.db._next_flavor_decay_monotonic = time.monotonic() - 1.0
        self.assertIs(
            await self.db.decay_flavor_stats_if_due(), MaintenanceOutcome.DONE
        )
        # The clock was re-stamped: an immediate second call is a no-op.
        self.assertIs(
            await self.db.decay_flavor_stats_if_due(), MaintenanceOutcome.SKIPPED
        )

    async def test_failed_decay_is_swallowed_and_retried_soon(self) -> None:
        """Сбой обслуживания не пробрасывается и не откладывается на сутки.

        Пробрасывать нельзя: вызывающий — learn-путь, и исключение стоило бы
        выученного сообщения. Откладывать на сутки тоже нельзя: одна занятая
        база заморозила бы окно «горячести» на день. Отсюда короткая пауза.
        """
        from unittest.mock import patch

        self.db._next_flavor_decay_monotonic = time.monotonic() - 1.0

        with patch.object(
            self.db, "decay_chat_hot_ngrams", side_effect=RuntimeError("disk gone")
        ):
            with self.assertLogs("chat_markov", level="ERROR"):
                self.assertIs(
                    await self.db.decay_flavor_stats_if_due(),
                    MaintenanceOutcome.FAILED,
                )

        assert self.db._next_flavor_decay_monotonic is not None
        wait = self.db._next_flavor_decay_monotonic - time.monotonic()
        self.assertGreater(wait, 0.0)
        self.assertLessEqual(wait, FLAVOR_DECAY_RETRY_INTERVAL_SEC)
        self.assertLess(wait, FLAVOR_DECAY_INTERVAL_SEC)
        # Повтора на следующем же вызове нет — иначе занятая база стоила бы
        # busy_timeout на каждом сообщении.
        self.assertIs(
            await self.db.decay_flavor_stats_if_due(), MaintenanceOutcome.SKIPPED
        )

    async def test_saving_a_message_does_not_run_maintenance(self) -> None:
        """Обслуживание отвязано от записи по устройству кода.

        Пока вызов затухания стоял первой строкой внутри
        ``save_message_and_update_model``, сбой обслуживания останавливал
        обучение во всех чатах сразу.
        """
        from unittest.mock import AsyncMock, patch

        self.db._next_flavor_decay_monotonic = time.monotonic() - 1.0
        with patch.object(
            self.db, "decay_flavor_stats_if_due", new=AsyncMock()
        ) as maintenance:
            await self.db.save_message_and_update_model(
                CHAT, "привет всем в этом чате", ["привет", "всем", "в", "этом", "чате"]
            )

        maintenance.assert_not_awaited()

    async def test_broken_maintenance_does_not_cost_the_message(self) -> None:
        """Сообщение выучивается, даже когда обслуживание падает."""
        from unittest.mock import patch

        self.db._next_flavor_decay_monotonic = time.monotonic() - 1.0
        tokens = ["сегодня", "хорошая", "погода", "в", "городе"]

        with patch.object(
            self.db, "decay_chat_emoji_stats", side_effect=RuntimeError("disk gone")
        ):
            # Тот же порядок, что у конвейера: сначала обучение, потом
            # обслуживание отдельным шагом.
            volume = await self.db.save_message_and_update_model(
                CHAT, "сегодня хорошая погода в городе", tokens
            )
            with self.assertLogs("chat_markov", level="ERROR"):
                await self.db.decay_flavor_stats_if_due()

        self.assertGreater(volume, 0)
        self.assertEqual((await self.db.get_stats(CHAT))["messages"], 1)

    async def test_successful_decay_postpones_by_the_daily_interval(self) -> None:
        self.db._next_flavor_decay_monotonic = time.monotonic() - 1.0

        self.assertIs(
            await self.db.decay_flavor_stats_if_due(), MaintenanceOutcome.DONE
        )

        assert self.db._next_flavor_decay_monotonic is not None
        wait = self.db._next_flavor_decay_monotonic - time.monotonic()
        self.assertGreater(wait, FLAVOR_DECAY_RETRY_INTERVAL_SEC)
        self.assertLessEqual(wait, FLAVOR_DECAY_INTERVAL_SEC)


if __name__ == "__main__":
    unittest.main()
