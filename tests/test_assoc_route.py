"""Associative route pilot (M3R-200, change assoc-route-pilot).

Two layers. The ranking (`rank_associates`) is checked on a real chain built
by hand, so the PMI neighbour on each side of an anchor is known. The route
itself is checked at the ResponseGenerator level with mocks, the way the
seeded and hot routes are: pool composition is captured at selection.
"""
from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.generation_telemetry import CandidateRoute
from app.core.markov import MarkovGenerator, tokenize
from app.core.response_generator import GenerationRequest, ResponseGenerator
from app.infrastructure.database import Database
from tests.test_response_generator import (
    _learning_service,
    _runtime_state,
    _score,
    _traced_generator,
)
from tests.test_route_slot_budget import PoolCompositionTestCase

CHAT = 4242


class TestRankAssociates(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_assoc_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        self.generator = MarkovGenerator(self.db.markov)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)
        for suffix in ("-shm", "-wal"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)

    async def _learn(self, *texts: str, times: int = 1) -> None:
        for _ in range(times):
            for text in texts:
                await self.db.save_message_and_update_model(
                    chat_id=CHAT, raw_text=text, tokens=tokenize(text)
                )

    async def _rank(self, message: str, *, slots: int = 2) -> list[str]:
        return await self.generator.rank_associates(
            CHAT, tokenize(message), slots=slots, min_token_len=3
        )

    # Rows are (w1, w2) -> w3, so a pair is visible as (w1, w2) only when a
    # third token follows, and an associate needs a forward row of its own to
    # be assembled around: the hand-built messages carry five tokens so both
    # neighbours of the anchor sit inside with a continuation.
    async def test_neighbours_on_both_sides_are_found(self) -> None:
        # "холодное пиво" (left neighbour) and "пиво вечером" (right neighbour)
        # recur; filler pairs seen once never qualify.
        await self._learn("холодное пиво вечером зашло отлично", times=3)
        await self._learn("пиво кончилось быстро вчера", "странное пиво было тут")
        picked = await self._rank("пиво", slots=3)
        self.assertEqual(set(picked), {"холодное", "вечером"})

    async def test_message_tokens_are_never_associates(self) -> None:
        await self._learn("холодное пиво вечером зашло отлично", times=3)
        picked = await self._rank("холодное пиво", slots=3)
        self.assertNotIn("холодное", picked)
        self.assertIn("вечером", picked)

    async def test_round_robin_across_anchors(self) -> None:
        await self._learn("холодное пиво вечером зашло отлично", times=3)
        await self._learn("громкая музыка играла долго вчера", times=3)
        picked = await self._rank("пиво музыка", slots=2)
        # One neighbour per anchor before a second from the same anchor.
        self.assertEqual(len(picked), 2)
        self.assertTrue({"холодное", "вечером"} & set(picked))
        self.assertTrue({"громкая", "играла"} & set(picked))

    async def test_empty_chain_gives_nothing(self) -> None:
        self.assertEqual(await self._rank("пиво"), [])

    async def test_backward_read_is_ordered(self) -> None:
        # CLAUDE.md §5: a SQL read that can feed a draw carries a full ORDER BY.
        await self._learn("яркое пиво было", "тёмное пиво было", "светлое пиво было", times=2)
        rows = await self.db.markov.get_seed_backward(CHAT, "пиво")
        self.assertEqual(len(rows), 3)
        self.assertEqual([w for w, _ in rows], sorted(w for w, _ in rows))
        self.assertTrue(all(cnt == 2 for _, cnt in rows))


def _request() -> GenerationRequest:
    return GenerationRequest(
        chat_id=123,
        context_tokens=["пиво", "сегодня"],
        seed=None,
        current_message_normalized="пиво сегодня",
    )


class AssocRouteTestCase(PoolCompositionTestCase):
    @staticmethod
    def _state(ratio: float) -> MagicMock:
        state = _runtime_state()
        state.assoc_slot_ratio = ratio
        state.markov_seed_min_token_len = 3
        state.markov_seed_head_share = 0.5
        state.slot_mutation_probability = 0.0
        return state

    @staticmethod
    def _generator(associates: list[str]) -> AsyncMock:
        generator = _traced_generator()
        counter = iter(range(100))
        generator.generate_text = AsyncMock(
            side_effect=lambda *a, **k: f"обычный кандидат номер {next(counter)} тут"
        )
        generator.rank_associates = AsyncMock(return_value=list(associates))
        generator.generate_seeded_candidate = AsyncMock(
            side_effect=lambda _chat, anchor, **k: ["вот", anchor, "и", "всё", "такое"]
        )
        return generator

    async def _run(
        self, state: MagicMock, generator: AsyncMock, *, target: int = 5
    ) -> list[object]:
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            await response_generator.generate_with_result(
                _request(), rng=random.Random(11), candidate_target=target
            )
        return self.captured[-1] if self.captured else []


class TestAssocRoute(AssocRouteTestCase):
    async def test_ratio_zero_reads_nothing(self) -> None:
        generator = self._generator(["холодное"])
        await self._run(self._state(0.0), generator)
        generator.rank_associates.assert_not_awaited()
        generator.generate_seeded_candidate.assert_not_awaited()
        self.assertEqual(generator.telemetry.assoc_draws, 0)
        self.assertEqual(generator.telemetry.route_breakdown()["assoc"]["attempts"], 0)

    async def test_two_slots_two_associates(self) -> None:
        generator = self._generator(["холодное", "вечером"])
        pool = await self._run(self._state(0.4), generator)
        self.assertEqual(generator.rank_associates.await_args.kwargs["slots"], 2)
        routes = [candidate.route for candidate in pool]
        self.assertEqual(routes.count(CandidateRoute.ASSOC), 2)
        self.assertIn(CandidateRoute.VANILLA, routes)
        self.assertLessEqual(len(pool), 5)
        breakdown = generator.telemetry.route_breakdown()["assoc"]
        self.assertEqual(breakdown["attempts"], 1)
        self.assertEqual(breakdown["present"], 1)
        self.assertEqual(generator.telemetry.assoc_draws, 1)
        self.assertEqual(generator.telemetry.assoc_empty, 0)

    async def test_empty_ranking_is_attempted_not_present(self) -> None:
        generator = self._generator([])
        pool = await self._run(self._state(0.4), generator)
        generator.generate_seeded_candidate.assert_not_awaited()
        breakdown = generator.telemetry.route_breakdown()["assoc"]
        self.assertEqual(breakdown["attempts"], 1)
        self.assertEqual(breakdown["present"], 0)
        self.assertEqual(generator.telemetry.assoc_empty, 1)
        self.assertEqual(len(pool), 5)

    async def test_walk_keeps_a_slot_with_every_route_on(self) -> None:
        # seeded 0.4 (2) + assoc 0.4 (2) at target 5: assoc is clamped so the
        # walk keeps its slot (design D3).
        state = self._state(0.4)
        state.markov_seeded_candidate_ratio = 0.4
        state.markov_seed_min_score = 0.0
        generator = self._generator(["холодное", "вечером"])
        generator.rank_seeds = AsyncMock(
            return_value=[MagicMock(token=f"якорь{i}", score=1.0) for i in range(3)]
        )
        pool = await self._run(state, generator)
        routes = [candidate.route for candidate in pool]
        self.assertLessEqual(len(pool), 5)
        self.assertIn(CandidateRoute.VANILLA, routes)
        self.assertEqual(routes.count(CandidateRoute.SEEDED), 2)
        self.assertEqual(routes.count(CandidateRoute.ASSOC), 2)
        self.assertEqual(generator.rank_associates.await_args.kwargs["slots"], 2)
