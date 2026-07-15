"""Mechanism tests for the context+chaos knobs (2026-07-15):

- ``context_jump_boost`` multiplies the M4 jump probability only for walks
  that started from a contextual anchor;
- ``order_mix_probability`` diverts a step to the order-2 pool only when it
  is genuinely wider than the order-3 pool;
- ``verbatim_extension_share`` extends passing near-quote candidates and
  keeps the original when the extension fails.
"""
from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.config.registry import RUNTIME_FIELDS
from app.core.markov import MarkovGenerator, tokenize
from app.core.response_generator import GenerationRequest, ResponseGenerator
from app.infrastructure.database import Database
from tests.test_response_generator import _score, _traced_generator


def _runtime_state(**overrides: object) -> SimpleNamespace:
    state = SimpleNamespace(
        **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS}
    )
    state.reply_flavor_strength = 0.0
    state.emoji_append_chance = 0.0
    state.markov_jump_probability = 0.0
    state.recent_short_replies = {}
    state.recent_replies = {}
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _request() -> GenerationRequest:
    return GenerationRequest(
        chat_id=123,
        context_tokens=["reply", "context", "tokens"],
        seed=None,
        current_message_normalized="другое сообщение",
    )


class _LoopSpyGenerator(MarkovGenerator):
    """Captures the jump probability the generation loop actually receives."""

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.seen_jump_probabilities: list[float] = []

    async def _run_generation_loop(self, *args, **kwargs):  # type: ignore[override]
        self.seen_jump_probabilities.append(float(kwargs["jump_probability"]))
        return await super()._run_generation_loop(*args, **kwargs)


class TestContextJumpBoost(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_boost_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        for text in ("первое пробное сообщение чата", "второе пробное сообщение чата"):
            await self.db.save_message_and_update_model(
                chat_id=1, raw_text=text, tokens=tokenize(text)
            )
        self.generator = _LoopSpyGenerator(self.db)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def _captured_jump_probability(
        self, *, contextual: bool, boost: float
    ) -> float:
        contextual_state = SimpleNamespace(
            state=("пробное", "сообщение", "чата"), order=3, match_kind="exact"
        )
        pick = AsyncMock(return_value=contextual_state if contextual else None)
        with patch.object(
            _LoopSpyGenerator, "_pick_contextual_start", pick
        ):
            await self.generator._generate_text_once(
                chat_id=1,
                max_chars=280,
                context_tokens=["пробное", "сообщение", "чата"],
                context_start_bias=4.0,
                jump_probability=0.1,
                context_jump_boost=boost,
                rng=random.Random(3),
            )
        return self.generator.seen_jump_probabilities[-1]

    async def test_boost_applies_only_to_contextual_walks(self) -> None:
        self.assertAlmostEqual(
            await self._captured_jump_probability(contextual=True, boost=5.0), 0.5
        )
        self.assertAlmostEqual(
            await self._captured_jump_probability(contextual=False, boost=5.0), 0.1
        )

    async def test_boost_is_clamped_to_one(self) -> None:
        self.assertAlmostEqual(
            await self._captured_jump_probability(contextual=True, boost=10.0), 1.0
        )


class TestOrderMix(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_mix_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        # State (б, в) carries three order-2 options while every order-3
        # state has exactly one continuation — the branching shape the valve
        # is built for.
        for text in ("а б в г", "х б в д", "у б в е"):
            await self.db.save_message_and_update_model(
                chat_id=1, raw_text=text, tokens=tokenize(text)
            )
        self.generator = MarkovGenerator(self.db)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def _orders_used(self, mix: float) -> set[int]:
        orders: set[int] = set()
        rng = random.Random(5)
        for _ in range(20):
            text, trace = await self.generator.generate_text_with_trace(
                chat_id=1,
                max_chars=280,
                max_tokens=10,
                order_mix_probability=mix,
                rng=rng,
                attempt_budget=1,
            )
            if text:
                orders.add(trace.markov_order_used)
        return orders

    async def test_mix_off_keeps_order3(self) -> None:
        self.assertEqual(await self._orders_used(0.0), {3})

    async def test_mix_on_steps_down_to_order2(self) -> None:
        self.assertIn(2, await self._orders_used(1.0))


class TestNearQuoteExtension(unittest.IsolatedAsyncioTestCase):
    def _service_with_index(self, candidate: str) -> AsyncMock:
        content = [t.casefold() for t in tokenize(candidate)]
        index = frozenset(
            tuple(content[i : i + 4]) for i in range(len(content) - 3)
        )
        service = AsyncMock()
        service.is_verbatim_copy = AsyncMock(return_value=False)
        service.get_verbatim_ngram_index = AsyncMock(return_value=index)
        service.get_context_idf = AsyncMock(return_value={})
        return service

    async def _generate(
        self, side_effect: list[str], share: float
    ) -> str | None:
        state = _runtime_state(verbatim_extension_share=share)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(side_effect=side_effect)
        service = self._service_with_index(side_effect[0])
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            return await response_generator.generate(
                _request(), rng=random.Random(7), candidate_target=1
            )

    async def test_near_quote_is_extended(self) -> None:
        result = await self._generate(
            [
                "почти цитата из пяти слов ровно",
                "свежий хвост из четырёх слов",
            ],
            share=0.8,
        )
        assert result is not None
        self.assertTrue(result.startswith("почти цитата из пяти слов ровно,"))
        self.assertIn("свежий хвост из четырёх слов", result)

    async def test_original_kept_when_extension_fails(self) -> None:
        result = await self._generate(
            ["почти цитата из пяти слов ровно", "", ""],
            share=0.8,
        )
        assert result is not None
        self.assertEqual(result.rstrip("."), "почти цитата из пяти слов ровно")

    async def test_share_zero_disables_the_channel(self) -> None:
        result = await self._generate(
            ["почти цитата из пяти слов ровно"],
            share=0.0,
        )
        assert result is not None
        self.assertEqual(result.rstrip("."), "почти цитата из пяти слов ровно")


if __name__ == "__main__":
    unittest.main()
