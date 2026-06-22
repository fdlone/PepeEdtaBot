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

    async def test_prefix_frequency_prior_beats_rare_closer_state(self) -> None:
        self.db.get_markov_states.return_value = [
            (("хоссейн", "джаббар"), 10),
            (("хоссейно", "джаббаров"), 1),
        ]

        matches = await self.matcher.match(
            5,
            ("хоссейном", "джаббаровичем"),
            2,
            include_prefix=True,
        )

        prefix_matches = [
            match for match in matches if match.match_kind == "prefix"
        ]
        self.assertEqual(
            [match.state for match in prefix_matches],
            [("хоссейн", "джаббар")],
        )
        self.assertEqual(prefix_matches[0].transition_count, 10)

    async def test_prefix_requires_cyrillic_length_and_common_prefix_guards(
        self,
    ) -> None:
        self.db.get_markov_states.return_value = [
            (("alpha", "джаббар"), 10),
            (("котик", "джаббар"), 10),
            (("хоссейн", "джаббар"), 10),
        ]

        latin = await self.matcher.match(
            6,
            ("alphabet", "джаббаровичем"),
            2,
            include_prefix=True,
        )
        short = await self.matcher.match(
            6,
            ("котика", "джаббаровичем"),
            2,
            include_prefix=True,
        )
        low_coverage = await self.matcher.match(
            6,
            ("хоссейновичдлинный", "джаббаровичем"),
            2,
            include_prefix=True,
        )

        self.assertFalse(any(match.match_kind == "prefix" for match in latin))
        self.assertFalse(any(match.match_kind == "prefix" for match in short))
        self.assertFalse(
            any(match.match_kind == "prefix" for match in low_coverage)
        )

    async def test_prefix_rejects_low_confidence_singleton(self) -> None:
        self.db.get_markov_states.return_value = [
            (("хоссейн", "говорит"), 1),
        ]

        matches = await self.matcher.match(
            7,
            ("хоссейном", "говорит"),
            2,
            include_prefix=True,
        )

        self.assertFalse(any(match.match_kind == "prefix" for match in matches))

    async def test_exact_and_casefold_precede_prefix(self) -> None:
        self.db.get_markov_states.return_value = [
            (("Хоссейн", "Джаббар"), 8),
            (("хоссейн", "джаббар"), 10),
            (("хоссейна", "джаббара"), 10),
        ]

        matches = await self.matcher.match(
            8,
            ("Хоссейн", "Джаббар"),
            2,
            include_prefix=True,
        )

        self.assertEqual(matches[0].match_kind, "exact")
        self.assertEqual(matches[1].match_kind, "casefold")
        self.assertEqual(matches[2].match_kind, "prefix")

    async def test_prefix_index_is_built_only_when_requested(self) -> None:
        self.db.get_markov_states.return_value = [
            (("хоссейн", "джаббар"), 10),
        ]

        await self.matcher.match(9, ("ХОССЕЙН", "ДЖАББАР"), 2)
        index = self.matcher._cache[(9, 2)]
        self.assertIsNone(index.prefix_tokens)

        await self.matcher.match(
            9,
            ("хоссейном", "джаббаровичем"),
            2,
            include_prefix=True,
        )
        self.assertIsNotNone(index.prefix_tokens)

    async def test_prefix_token_candidates_are_bounded(self) -> None:
        self.db.get_markov_states.return_value = [
            ((f"хоссейн{suffix}", "джаббар"), count)
            for suffix, count in zip(
                ("а", "б", "в", "г", "д", "е", "ж", "з"),
                range(10, 2, -1),
            )
        ]

        matches = await self.matcher.match(
            10,
            ("хоссейном", "джаббар"),
            2,
            include_prefix=True,
        )
        prefix_matches = [
            match for match in matches if match.match_kind == "prefix"
        ]

        self.assertLessEqual(len(prefix_matches), 6)
        self.assertIn(("хоссейна", "джаббар"), {
            match.state for match in prefix_matches
        })
