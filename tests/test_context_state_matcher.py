from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from app.core.context_state_matcher import ContextStateMatcher
from app.repositories import MarkovRepo


class TestContextStateMatcher(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.db = AsyncMock(spec=MarkovRepo)
        self.matcher = ContextStateMatcher(self.db, cache_limit=4)

    async def test_repeated_lookup_uses_cached_index(self) -> None:
        self.db.get_states.return_value = [(("alpha", "beta"), 3)]

        await self.matcher.match(3, ("alpha", "beta"), 2)
        await self.matcher.match(3, ("alpha", "beta"), 2)

        self.db.get_states.assert_awaited_once_with(3, 2)

    async def test_invalidation_rebuilds_all_orders_for_chat(self) -> None:
        self.db.get_states.side_effect = [
            [(("alpha", "beta"), 3)],
            [(("alpha", "beta", "gamma"), 4)],
            [(("alpha", "beta"), 5)],
            [(("alpha", "beta", "gamma"), 6)],
        ]

        await self.matcher.match(4, ("alpha", "beta"), 2)
        await self.matcher.match(4, ("alpha", "beta", "gamma"), 3)
        self.matcher.invalidate_chat_cache(4)
        matches2 = await self.matcher.match(4, ("alpha", "beta"), 2)
        matches3 = await self.matcher.match(4, ("alpha", "beta", "gamma"), 3)

        self.assertEqual(matches2[0].transition_count, 5)
        self.assertEqual(matches3[0].transition_count, 6)
        self.assertEqual(self.db.get_states.await_count, 4)

    async def test_rejects_invalid_order_and_window_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "order"):
            await self.matcher.match(1, ("alpha",), 1)
        with self.assertRaisesRegex(ValueError, "length"):
            await self.matcher.match(1, ("alpha",), 2)

    async def test_inflected_variants_do_not_match(self) -> None:
        # The stem tier was removed 2026-07-14 (no starts on prod data) and the
        # casefold tier 2026-09-11 (inert: the corpus learns in lower case).
        # Matching is exact; morphology lives in context_start_affinity and IDF
        # relevance instead.
        self.db.get_states.return_value = [
            (("тренировка", "помогает"), 10),
        ]

        matches = await self.matcher.match(5, ("тренировки", "помогают"), 2)

        self.assertEqual(matches, [])
