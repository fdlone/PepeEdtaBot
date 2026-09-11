"""Phrase route (M3R-210, change phrase-route).

Three layers, each on its own fixture. The assembler is checked on a hand-
built chain so the phrase's neighbours are known; the index read on a
temporary database; the ranking as a pure function. The route itself is
checked at the ResponseGenerator level with mocks, like the assoc route.
"""
from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.generation_telemetry import CandidateRoute
from app.core.markov import MarkovGenerator, tokenize
from app.core.phrase_route import rank_phrases
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


def _contains_in_order(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    n = len(phrase)
    return any(tuple(tokens[i : i + n]) == phrase for i in range(len(tokens) - n + 1))


class ChainTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_phrase_route_{uuid.uuid4().hex}.sqlite")
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


class TestPhraseAssembly(ChainTestCase):
    async def _assemble(self, phrase: tuple[str, ...]) -> list[str] | None:
        return await self.generator.generate_phrase_candidate(
            CHAT,
            phrase,
            max_tokens=12,
            head_share=0.5,
            next_explore=0.0,
            next_power=1.0,
            repetition_penalty_strength=1.0,
            rng=random.Random(3),
        )

    async def test_trigram_stays_contiguous_between_head_and_tail(self) -> None:
        await self._learn("красный дракон летит над городом ночью")
        result = await self._assemble(("дракон", "летит", "над"))
        assert result is not None
        self.assertTrue(_contains_in_order(result, ("дракон", "летит", "над")), result)
        idx = result.index("дракон")
        self.assertEqual(result[idx - 1], "красный")
        self.assertEqual(result[idx + 3], "городом")

    async def test_bigram_grows_on_both_sides(self) -> None:
        await self._learn("красный дракон летит над городом ночью")
        result = await self._assemble(("летит", "над"))
        assert result is not None
        self.assertTrue(_contains_in_order(result, ("летит", "над")), result)
        self.assertGreater(result.index("летит"), 0)
        self.assertLess(result.index("над"), len(result) - 1)

    async def test_phrase_with_nothing_around_it_is_none(self) -> None:
        # The whole chain is the phrase: no head and no tail can grow.
        await self._learn("дракон летит")
        self.assertIsNone(await self._assemble(("дракон", "летит")))

    async def test_seeded_assembly_is_unchanged_by_the_refactor(self) -> None:
        # The seeded path bootstraps a pair then grows around it; the phrase
        # path skips the bootstrap. Same chain, same rng: the seeded result is
        # the phrase result for the pair the bootstrap picked.
        await self._learn("красный дракон летит над городом ночью")
        seeded = await self.generator.generate_seeded_candidate(
            CHAT, "летит", max_tokens=12, head_share=0.5, next_explore=0.0,
            next_power=1.0, repetition_penalty_strength=1.0, rng=random.Random(3),
        )
        self.assertEqual(seeded, await self._assemble(("летит", "над")))


class TestIndexReadByAnchor(ChainTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.repo = self.db.chat_phrase_ngrams

    async def test_read_is_ordered_and_thresholded(self) -> None:
        await self._learn("холодное пиво вечером зашло", times=3)
        await self._learn("тёплое пиво утром", times=2)
        await self._learn("странное пиво было тут")
        await self.repo.rebuild_chat(CHAT)
        rows = await self.repo.get_phrases_containing(CHAT, ["пиво"], min_count=2)
        self.assertTrue(rows)
        self.assertTrue(all(cnt >= 2 for _, cnt in rows))
        self.assertTrue(all("пиво" in phrase for phrase, _ in rows))
        # CLAUDE.md §5: full ORDER BY — count descending, then the key columns.
        keys = [(-cnt, *phrase, *("",) * (3 - len(phrase))) for phrase, cnt in rows]
        self.assertEqual(keys, sorted(keys))
        self.assertNotIn(("странное", "пиво"), [p for p, _ in rows])

    async def test_no_anchors_reads_nothing(self) -> None:
        self.assertEqual(await self.repo.get_phrases_containing(CHAT, [], min_count=1), [])


class TestRankPhrases(unittest.TestCase):
    ROWS = [
        (("холодное", "пиво", "вечером"), 5),
        (("холодное", "пиво"), 5),
        (("пиво", "вечером"), 5),
        (("громкая", "музыка", "играла"), 4),
        (("пиво", "сегодня"), 3),
    ]

    def test_round_robin_and_slices_skipped(self) -> None:
        picked = rank_phrases(
            self.ROWS, anchors=["пиво", "музыка"], message_tokens=["пиво", "музыка"], slots=3
        )
        self.assertEqual(picked[0], ("холодное", "пиво", "вечером"))
        self.assertEqual(picked[1], ("громкая", "музыка", "играла"))
        # The two bigrams are slices of the chosen trigram; the next distinct
        # phrase for the anchor is taken instead.
        self.assertEqual(picked[2], ("пиво", "сегодня"))

    def test_phrase_made_of_message_tokens_is_a_copy(self) -> None:
        picked = rank_phrases(
            self.ROWS, anchors=["пиво"], message_tokens=["пиво", "сегодня"], slots=5
        )
        self.assertNotIn(("пиво", "сегодня"), picked)

    def test_zero_slots_or_no_anchors(self) -> None:
        self.assertEqual(rank_phrases(self.ROWS, anchors=["пиво"], message_tokens=[], slots=0), [])
        self.assertEqual(rank_phrases(self.ROWS, anchors=[], message_tokens=[], slots=2), [])

    def test_deterministic_on_ties(self) -> None:
        rows = [(("а", "пиво"), 2), (("б", "пиво"), 2)]
        first = rank_phrases(rows, anchors=["пиво"], message_tokens=["пиво"], slots=1)
        self.assertEqual(first, [("а", "пиво")])
        self.assertEqual(
            first, rank_phrases(rows, anchors=["пиво"], message_tokens=["пиво"], slots=1)
        )


def _request() -> GenerationRequest:
    return GenerationRequest(
        chat_id=123,
        context_tokens=["пиво", "сегодня"],
        current_message_normalized="пиво музыка",
    )


class PhraseRouteTestCase(PoolCompositionTestCase):
    @staticmethod
    def _state(ratio: float) -> MagicMock:
        state = _runtime_state()
        state.phrase_slot_ratio = ratio
        state.phrase_min_count = 3
        state.markov_seed_min_token_len = 3
        state.markov_seed_head_share = 0.5
        state.slot_mutation_probability = 0.0
        return state

    @staticmethod
    def _generator() -> AsyncMock:
        generator = _traced_generator()
        counter = iter(range(100))
        generator.generate_text = AsyncMock(
            side_effect=lambda *a, **k: f"обычный кандидат номер {next(counter)} тут"
        )
        generator.generate_phrase_candidate = AsyncMock(
            side_effect=lambda _chat, phrase, **k: ["вот", *phrase, "и", "всё"]
        )
        generator.generate_seeded_candidate = AsyncMock(
            side_effect=lambda _chat, anchor, **k: ["вот", anchor, "и", "всё", "такое"]
        )
        return generator

    async def _run(
        self,
        state: MagicMock,
        generator: AsyncMock,
        rows: list[tuple[tuple[str, ...], int]],
        *,
        target: int = 5,
    ) -> tuple[list[object], AsyncMock]:
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_phrases_containing = AsyncMock(return_value=rows)
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
        return (self.captured[-1] if self.captured else []), learning_service


ROWS = [(("холодное", "пиво", "вечером"), 5), (("громкая", "музыка", "играла"), 4)]


class TestPhraseRoute(PhraseRouteTestCase):
    async def test_ratio_zero_reads_nothing(self) -> None:
        generator = self._generator()
        _pool, service = await self._run(self._state(0.0), generator, ROWS)
        service.get_phrases_containing.assert_not_awaited()
        generator.generate_phrase_candidate.assert_not_awaited()
        self.assertEqual(generator.telemetry.phrase_draws, 0)
        self.assertEqual(generator.telemetry.route_breakdown()["phrase"]["attempts"], 0)

    async def test_two_slots_two_phrases_of_different_anchors(self) -> None:
        generator = self._generator()
        pool, service = await self._run(self._state(0.4), generator, ROWS)
        self.assertEqual(service.get_phrases_containing.await_args.kwargs["min_count"], 3)
        routes = [candidate.route for candidate in pool]
        self.assertEqual(routes.count(CandidateRoute.PHRASE), 2)
        self.assertIn(CandidateRoute.VANILLA, routes)
        self.assertLessEqual(len(pool), 5)
        phrases = [
            call.args[1] for call in generator.generate_phrase_candidate.await_args_list
        ]
        self.assertEqual(phrases, [ROWS[0][0], ROWS[1][0]])
        texts = [c.text for c in pool if c.route == CandidateRoute.PHRASE]
        self.assertTrue(any("холодное пиво вечером" in text for text in texts), texts)
        breakdown = generator.telemetry.route_breakdown()["phrase"]
        self.assertEqual((breakdown["attempts"], breakdown["present"]), (1, 1))
        self.assertEqual(
            (generator.telemetry.phrase_draws, generator.telemetry.phrase_empty), (1, 0)
        )

    async def test_empty_index_is_attempted_not_present(self) -> None:
        generator = self._generator()
        pool, _service = await self._run(self._state(0.4), generator, [])
        generator.generate_phrase_candidate.assert_not_awaited()
        breakdown = generator.telemetry.route_breakdown()["phrase"]
        self.assertEqual((breakdown["attempts"], breakdown["present"]), (1, 0))
        self.assertEqual(generator.telemetry.phrase_empty, 1)
        self.assertEqual(len(pool), 5)

    async def test_walk_keeps_a_slot_with_every_route_on(self) -> None:
        # seeded 0.4 (2) + assoc 0.4 (2) + phrase 0.4 at target 5: assoc and
        # phrase are clamped so the walk keeps its slot (design D4).
        state = self._state(0.4)
        state.markov_seeded_candidate_ratio = 0.4
        state.markov_seed_min_score = 0.0
        state.assoc_slot_ratio = 0.4
        generator = self._generator()
        generator.rank_seeds = AsyncMock(
            return_value=[MagicMock(token=f"якорь{i}", score=1.0) for i in range(3)]
        )
        generator.rank_associates = AsyncMock(return_value=["сосед1", "сосед2"])
        pool, _service = await self._run(state, generator, ROWS)
        routes = [candidate.route for candidate in pool]
        self.assertLessEqual(len(pool), 5)
        self.assertIn(CandidateRoute.VANILLA, routes)
        self.assertEqual(routes.count(CandidateRoute.SEEDED), 2)
        self.assertEqual(routes.count(CandidateRoute.ASSOC), 2)
        self.assertEqual(routes.count(CandidateRoute.PHRASE), 0)
