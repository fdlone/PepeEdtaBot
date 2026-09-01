"""L1 as a route with a slot budget (M3R-230, change l1-hot-route).

The route's slots are the first attempts of the pool-building loop, each
seeded by its own hot n-gram; at ratio 0 nothing is drawn and nothing is read;
with context present the route never runs. Pool composition is captured at
selection, as in the sweep harness (see test_route_slot_budget).
"""
from __future__ import annotations

import random
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.generation_telemetry import CandidateRoute
from app.core.response_generator import GenerationRequest, ResponseGenerator
from tests.test_response_generator import (
    _learning_service,
    _runtime_state,
    _score,
    _traced_generator,
)
from tests.test_route_slot_budget import PoolCompositionTestCase

HOT_POOL = [("пиво", "сегодня"), ("опять", "ты"), ("ну", "такое")]


def _self_initiated() -> GenerationRequest:
    return GenerationRequest(
        chat_id=123,
        context_tokens=[],
        seed=None,
        current_message_normalized="что-то в чате",
    )


def _addressed() -> GenerationRequest:
    return GenerationRequest(
        chat_id=123,
        context_tokens=["reply", "context"],
        seed=None,
        current_message_normalized="эй бот",
    )


class HotRouteTestCase(PoolCompositionTestCase):
    @staticmethod
    def _state(ratio: float) -> MagicMock:
        state = _runtime_state()
        state.hot_ngram_slot_ratio = ratio
        state.hot_ngram_min_count = 2
        state.hot_ngram_recency_share = 0.25
        state.slot_mutation_probability = 0.0
        return state

    @staticmethod
    def _walk_generator() -> AsyncMock:
        """Each attempt yields a distinct text and records its seed tokens."""
        generator = _traced_generator()
        counter = iter(range(100))

        def _walk(*_args, **kwargs):
            seed = kwargs.get("seed_tokens")
            prefix = " ".join(seed) if seed else "обычный"
            return f"{prefix} кандидат номер {next(counter)} тут"

        generator.generate_text = AsyncMock(side_effect=_walk)
        return generator

    async def _run(
        self,
        state: MagicMock,
        generator: AsyncMock,
        request: GenerationRequest,
        *,
        hot: list[tuple[str, ...]] | None = None,
        verbatim: bool = False,
        target: int = 5,
    ) -> tuple[list[object], AsyncMock]:
        learning_service = _learning_service()
        learning_service.get_hot_ngrams = AsyncMock(return_value=list(hot or []))
        learning_service.is_verbatim_copy = AsyncMock(return_value=verbatim)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            await response_generator.generate_with_result(
                request, rng=random.Random(11), candidate_target=target
            )
        return (self.captured[-1] if self.captured else []), learning_service


class TestHotRouteSlots(HotRouteTestCase):
    async def test_two_slots_seed_two_different_ngrams(self) -> None:
        generator = self._walk_generator()
        pool, _ = await self._run(
            self._state(0.4), generator, _self_initiated(), hot=HOT_POOL
        )
        calls = generator.generate_text.await_args_list
        seeds = [tuple(c.kwargs["seed_tokens"]) for c in calls[:2]]
        self.assertEqual(len(set(seeds)), 2)
        self.assertTrue(all(seed in HOT_POOL for seed in seeds))
        self.assertTrue(all(c.kwargs["seed_tokens"] is None for c in calls[2:]))
        self.assertLessEqual(len(pool), 5)
        routes = [candidate.route for candidate in pool]
        self.assertEqual(routes.count(CandidateRoute.HOT), 2)
        self.assertIn(CandidateRoute.VANILLA, routes)
        breakdown = generator.telemetry.route_breakdown()["hot"]
        self.assertEqual(breakdown["attempts"], 1)
        self.assertEqual(breakdown["present"], 1)

    async def test_small_ratio_gives_one_slot(self) -> None:
        generator = self._walk_generator()
        await self._run(self._state(0.2), generator, _self_initiated(), hot=HOT_POOL)
        calls = generator.generate_text.await_args_list
        self.assertIsNotNone(calls[0].kwargs["seed_tokens"])
        self.assertTrue(all(c.kwargs["seed_tokens"] is None for c in calls[1:]))

    async def test_rejected_hot_attempt_gives_its_slot_back(self) -> None:
        generator = self._walk_generator()
        # Every candidate is a verbatim copy and the walk cannot extend it: the
        # hot attempts are rejected, the loop keeps going and the pool is
        # still filled by the walk.
        learning = _learning_service()
        pool, _ = await self._run(
            self._state(0.4), generator, _self_initiated(), hot=HOT_POOL, verbatim=True
        )
        del learning
        rejected = generator.telemetry.route_rejection_reasons()
        self.assertIn("hot", rejected)
        breakdown = generator.telemetry.route_breakdown()["hot"]
        self.assertEqual(breakdown["attempts"], 1)
        self.assertEqual(breakdown["present"], 0)


class TestHotRouteGates(HotRouteTestCase):
    async def test_zero_ratio_reads_and_draws_nothing(self) -> None:
        generator = self._walk_generator()
        _, learning_service = await self._run(
            self._state(0.0), generator, _self_initiated(), hot=HOT_POOL
        )
        learning_service.get_hot_ngrams.assert_not_awaited()
        calls = generator.generate_text.await_args_list
        self.assertTrue(all(c.kwargs["seed_tokens"] is None for c in calls))
        self.assertNotIn("hot", generator.telemetry.route_attempts)

    async def test_context_disables_the_route(self) -> None:
        generator = self._walk_generator()
        _, learning_service = await self._run(
            self._state(0.4), generator, _addressed(), hot=HOT_POOL
        )
        learning_service.get_hot_ngrams.assert_not_awaited()
        calls = generator.generate_text.await_args_list
        self.assertTrue(all(c.kwargs["seed_tokens"] is None for c in calls))
        self.assertNotIn("hot", generator.telemetry.route_attempts)

    async def test_empty_selection_is_attempted_not_present(self) -> None:
        generator = self._walk_generator()
        pool, _ = await self._run(self._state(0.4), generator, _self_initiated(), hot=[])
        breakdown = generator.telemetry.route_breakdown()["hot"]
        self.assertEqual(breakdown["attempts"], 1)
        self.assertEqual(breakdown["present"], 0)
        self.assertEqual(generator.telemetry.hot_ngram_draws, 1)
        self.assertEqual(generator.telemetry.hot_ngram_empty, 1)
        self.assertEqual(len(pool), 5)
        self.assertTrue(all(c.route == CandidateRoute.VANILLA for c in pool))

    async def test_zero_ratio_leaves_the_rng_stream_untouched(self) -> None:
        """The pool at ratio 0 equals the pool of a generator that never had
        the route: same texts, same order (RNG consumption unchanged)."""
        first = self._walk_generator()
        pool_a, _ = await self._run(self._state(0.0), first, _self_initiated(), hot=HOT_POOL)
        second = self._walk_generator()
        pool_b, _ = await self._run(self._state(0.0), second, _self_initiated(), hot=[])
        self.assertEqual(
            [c.text for c in pool_a], [c.text for c in pool_b]
        )


if __name__ == "__main__":
    unittest.main()
