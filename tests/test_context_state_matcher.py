from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from app.core.context_state_matcher import ContextStateMatcher
from app.infrastructure.database import Database


class TestContextStateMatcher(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.db = AsyncMock(spec=Database)
        self.matcher = ContextStateMatcher(self.db, cache_limit=4)

    async def test_exact_precedes_casefold_matches(self) -> None:
        self.db.get_markov_states.return_value = [
            (("Alpha", "Beta"), 3),
            (("alpha", "beta"), 8),
        ]

        matches = await self.matcher.match(1, ("Alpha", "Beta"), 2)

        self.assertEqual(
            [(match.state, match.match_kind) for match in matches],
            [
                (("Alpha", "Beta"), "exact"),
                (("alpha", "beta"), "casefold"),
            ],
        )

    async def test_casefold_groups_variants_and_orders_deterministically(self) -> None:
        self.db.get_markov_states.return_value = [
            (("ALPHA", "BETA", "GAMMA"), 2),
            (("Alpha", "Beta", "Gamma"), 7),
            (("alpha", "beta", "gamma"), 7),
        ]

        matches = await self.matcher.match(
            2,
            ("aLpHa", "bEtA", "gAmMa"),
            3,
        )

        self.assertEqual(
            [(match.state, match.transition_count) for match in matches],
            [
                (("Alpha", "Beta", "Gamma"), 7),
                (("alpha", "beta", "gamma"), 7),
                (("ALPHA", "BETA", "GAMMA"), 2),
            ],
        )
        self.assertTrue(all(match.match_kind == "casefold" for match in matches))

    async def test_repeated_lookup_uses_cached_index(self) -> None:
        self.db.get_markov_states.return_value = [(("Alpha", "Beta"), 3)]

        await self.matcher.match(3, ("alpha", "beta"), 2)
        await self.matcher.match(3, ("ALPHA", "BETA"), 2)

        self.db.get_markov_states.assert_awaited_once_with(3, 2)

    async def test_invalidation_rebuilds_all_orders_for_chat(self) -> None:
        self.db.get_markov_states.side_effect = [
            [(("Alpha", "Beta"), 3)],
            [(("Alpha", "Beta", "Gamma"), 4)],
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
        self.assertEqual(self.db.get_markov_states.await_count, 4)

    async def test_rejects_invalid_order_and_window_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "order"):
            await self.matcher.match(1, ("alpha",), 1)
        with self.assertRaisesRegex(ValueError, "length"):
            await self.matcher.match(1, ("alpha",), 2)
