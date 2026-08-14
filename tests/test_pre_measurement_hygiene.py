"""Гигиена измерений PRE: режим ответа и «канал вернул пусто» (M3R-140/141).

Свойство, которое здесь закрепляется, — читаемость нуля. Ноль в числителе при
живом знаменателе значит «спросили и ответил»; отсутствие знаменателя значит
«не спрашивали», и различать их обязана каждая пара счётчиков.
"""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app import log_masking
from app.core.generation_telemetry import GenerationTelemetry
from app.core.markov import MarkovGenerator
from app.core.response_generator import (
    GENERATION_ATTEMPT_BUDGET,
    GENERATION_ATTEMPTS_WITH_CONTEXT,
    GenerationRequest,
    ResponseGenerator,
)
from tests.test_response_generator import (
    _learning_service,
    _runtime_state,
    _score,
    _traced_generator,
)

CHAT = 4321


class TestContextModeCounters(unittest.TestCase):
    """M3R-140: доля режима — вес вердикта, и она обязана быть долей."""

    def test_share_needs_a_generation_first(self) -> None:
        snap = GenerationTelemetry().snapshot()
        self.assertEqual(snap["context_mode_generations"], 0)
        self.assertIsNone(snap["ctx_generation_share"])

    def test_share_counts_requests_not_outcomes(self) -> None:
        t = GenerationTelemetry()
        t.note_context_mode(with_context=True)
        t.note_context_mode(with_context=True)
        t.note_context_mode(with_context=False)
        snap = t.snapshot()
        self.assertEqual(snap["context_mode_generations"], 3)
        self.assertAlmostEqual(snap["ctx_generation_share"], 2 / 3)


class TestEmptyChannelCounters(unittest.TestCase):
    """M3R-141: у каждой пары свой знаменатель."""

    def test_seed_ranking_asked_and_answered(self) -> None:
        t = GenerationTelemetry()
        t.note_seed_ranking(no_corpus=False)
        t.note_seed_ranking(no_corpus=False)
        snap = t.snapshot()
        self.assertEqual(snap["seed_ranking_asked"], 2)
        # Измеренный ноль: канал спрашивали, корпус был.
        self.assertEqual(snap["seed_ranking_no_corpus_rate"], 0.0)

    def test_seed_ranking_starved_by_data(self) -> None:
        t = GenerationTelemetry()
        t.note_seed_ranking(no_corpus=True)
        t.note_seed_ranking(no_corpus=False)
        self.assertAlmostEqual(
            t.snapshot()["seed_ranking_no_corpus_rate"], 1 / 2
        )

    def test_seed_ranking_never_asked_is_not_a_zero(self) -> None:
        snap = GenerationTelemetry().snapshot()
        self.assertEqual(snap["seed_ranking_asked"], 0)
        self.assertIsNone(snap["seed_ranking_no_corpus_rate"])

    def test_hot_ngram_pair(self) -> None:
        t = GenerationTelemetry()
        t.note_hot_ngram_draw(empty=True)
        t.note_hot_ngram_draw(empty=True)
        t.note_hot_ngram_draw(empty=False)
        snap = t.snapshot()
        self.assertEqual(snap["hot_ngram_draws"], 3)
        self.assertAlmostEqual(snap["hot_ngram_empty_rate"], 2 / 3)

    def test_hot_ngram_never_drawn_is_not_a_zero(self) -> None:
        snap = GenerationTelemetry().snapshot()
        self.assertEqual(snap["hot_ngram_draws"], 0)
        self.assertIsNone(snap["hot_ngram_empty_rate"])

    def test_context_drop_rate_is_over_ctx_generations(self) -> None:
        t = GenerationTelemetry()
        for _ in range(4):
            t.note_context_mode(with_context=True)
        t.note_context_mode(with_context=False)  # в знаменатель не идёт
        t.note_context_dropped()
        self.assertAlmostEqual(t.snapshot()["context_dropped_rate"], 1 / 4)

    def test_context_drop_rate_without_ctx_generations(self) -> None:
        t = GenerationTelemetry()
        t.note_context_mode(with_context=False)
        self.assertIsNone(t.snapshot()["context_dropped_rate"])


def _request(*, with_context: bool) -> GenerationRequest:
    return GenerationRequest(
        chat_id=CHAT,
        context_tokens=["контекст", "из", "трёх"] if with_context else [],
        seed=None,
        current_message_normalized="исходное сообщение",
    )


class TestModeIsCountedOncePerGeneration(unittest.IsolatedAsyncioTestCase):
    async def _run(self, *, with_context: bool, text: str) -> GenerationTelemetry:
        generator = _traced_generator()
        generator.telemetry = GenerationTelemetry()
        generator.generate_text = AsyncMock(return_value=text)
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=_runtime_state(),
            scorer=MagicMock(return_value=_score(1.0)),
        )
        with patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        ):
            await response_generator.generate(
                _request(with_context=with_context),
                rng=random.Random(7),
                candidate_target=1,
            )
        return generator.telemetry

    async def test_reply_with_context(self) -> None:
        telemetry = await self._run(
            with_context=True, text="кандидат из четырёх слов"
        )
        self.assertEqual(telemetry.ctx_generations, 1)
        self.assertEqual(telemetry.noctx_generations, 0)

    async def test_reply_without_context(self) -> None:
        telemetry = await self._run(
            with_context=False, text="кандидат из четырёх слов"
        )
        self.assertEqual(telemetry.ctx_generations, 0)
        self.assertEqual(telemetry.noctx_generations, 1)

    async def test_failed_generation_still_counts_its_mode(self) -> None:
        """Ноль кандидатов — тоже генерация: иначе доля считалась бы по
        успешным ответам, а гейт взвешивался бы по смещённой выборке."""
        telemetry = await self._run(with_context=True, text="")
        self.assertEqual(telemetry.ctx_generations, 1)


class TestContextDropIsObservable(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        with_context: bool,
        texts: list[str],
        target: int = GENERATION_ATTEMPT_BUDGET,
    ) -> GenerationTelemetry:
        generator = _traced_generator()
        generator.telemetry = GenerationTelemetry()
        generator.generate_text = AsyncMock(side_effect=texts)
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=_runtime_state(),
            scorer=MagicMock(return_value=_score(1.0)),
        )
        with patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        ):
            await response_generator.generate(
                _request(with_context=with_context),
                rng=random.Random(7),
                candidate_target=target,
            )
        return generator.telemetry

    async def test_drop_counted_once_not_per_attempt(self) -> None:
        # Пустой текст на каждой попытке: пул не наберётся, бюджет попыток с
        # контекстом кончится, оставшиеся пойдут без него.
        telemetry = await self._run(
            with_context=True, texts=[""] * GENERATION_ATTEMPT_BUDGET
        )
        self.assertGreater(
            GENERATION_ATTEMPT_BUDGET, GENERATION_ATTEMPTS_WITH_CONTEXT
        )
        self.assertEqual(telemetry.context_dropped, 1)
        self.assertAlmostEqual(telemetry.snapshot()["context_dropped_rate"], 1.0)

    async def test_no_drop_when_the_pool_fills_early(self) -> None:
        telemetry = await self._run(
            with_context=True, texts=["кандидат из четырёх слов"], target=1
        )
        self.assertEqual(telemetry.context_dropped, 0)
        self.assertEqual(telemetry.snapshot()["context_dropped_rate"], 0.0)

    async def test_noctx_generation_never_reports_a_drop(self) -> None:
        """Без контекста терять нечего — иначе доля считалась бы от чужого
        знаменателя."""
        telemetry = await self._run(
            with_context=False, texts=[""] * GENERATION_ATTEMPT_BUDGET
        )
        self.assertEqual(telemetry.context_dropped, 0)


class TestSeedRankingCountsWhereItStarves(unittest.IsolatedAsyncioTestCase):
    def _generator(self, n_docs: int) -> MarkovGenerator:
        db = AsyncMock()
        db.get_n_docs = AsyncMock(return_value=n_docs)
        db.get_seed_forward = AsyncMock(return_value=[("продолжение", 7)])
        db.get_reverse_branch = AsyncMock(return_value=3)
        db.get_token_df = AsyncMock(return_value=2)
        return MarkovGenerator(db=db)

    async def test_empty_df_is_counted_as_no_corpus(self) -> None:
        generator = self._generator(n_docs=0)
        self.assertEqual(await self._rank(generator), [])
        snap = generator.telemetry.snapshot()
        self.assertEqual(snap["seed_ranking_asked"], 1)
        self.assertEqual(snap["seed_ranking_no_corpus_rate"], 1.0)

    async def test_accumulated_df_is_counted_as_answered(self) -> None:
        generator = self._generator(n_docs=500)
        await self._rank(generator)
        snap = generator.telemetry.snapshot()
        self.assertEqual(snap["seed_ranking_asked"], 1)
        self.assertEqual(snap["seed_ranking_no_corpus_rate"], 0.0)

    async def test_no_scorable_tokens_is_not_an_empty_corpus(self) -> None:
        """Нет пригодных токенов — тоже пустой ответ, но по другой причине;
        смешение этих двух и есть та неразличимость, ради которой счётчик
        заводился."""
        generator = self._generator(n_docs=0)
        self.assertEqual(
            await generator.rank_seeds(
                CHAT,
                ["и", "а"],
                min_support=5.0,
                branch_min=2.0,
                branch_ideal=6.0,
                branch_max=50.0,
                min_token_len=3,
            ),
            [],
        )
        snap = generator.telemetry.snapshot()
        self.assertEqual(snap["seed_ranking_asked"], 1)
        self.assertEqual(snap["seed_ranking_no_corpus_rate"], 0.0)

    @staticmethod
    async def _rank(generator: MarkovGenerator) -> list[object]:
        return list(
            await generator.rank_seeds(
                CHAT,
                ["собака", "гуляла"],
                min_support=5.0,
                branch_min=2.0,
                branch_ideal=6.0,
                branch_max=50.0,
                min_token_len=3,
            )
        )


class TestHotNgramDrawIsCounted(unittest.IsolatedAsyncioTestCase):
    """Пустой `get_hot` — канал, выключенный данными (карта §3.2б)."""

    def setUp(self) -> None:
        # Путь непустой затравки пишет отладочную строку с маскированным
        # chat_id; без инициализации маскирование падает намеренно.
        log_masking.init_masking("test-secret-for-measurement-hygiene")

    def _pipeline(self, hot: list[tuple[str, ...]]) -> tuple[object, MagicMock]:
        from app.services.reply_pipeline import ReplyPipeline

        generator = MagicMock()
        generator.telemetry = GenerationTelemetry()
        learning_service = AsyncMock()
        learning_service.get_hot_ngrams = AsyncMock(return_value=hot)
        state = SimpleNamespace(
            hot_ngram_seed_chance=1.0,
            hot_ngram_min_count=3,
            hot_ngram_recency_share=0.5,
            markov_hot_ngram_meme_ordering=False,
        )
        pipeline = ReplyPipeline.__new__(ReplyPipeline)
        pipeline._runtime_state = state  # type: ignore[attr-defined]
        pipeline._generator = generator  # type: ignore[attr-defined]
        pipeline._learning_service = learning_service  # type: ignore[attr-defined]
        return pipeline, generator

    async def _draw(self, hot: list[tuple[str, ...]]) -> GenerationTelemetry:
        pipeline, generator = self._pipeline(hot)
        msg = SimpleNamespace(chat_id=CHAT)
        obs = SimpleNamespace(address_reply=False)
        await pipeline._hot_ngram_seed(msg, obs)  # type: ignore[attr-defined]
        return generator.telemetry

    async def test_empty_selection_is_counted(self) -> None:
        telemetry = await self._draw([])
        snap = telemetry.snapshot()
        self.assertEqual(snap["hot_ngram_draws"], 1)
        self.assertEqual(snap["hot_ngram_empty_rate"], 1.0)

    async def test_non_empty_selection_is_counted(self) -> None:
        telemetry = await self._draw([("горячая", "фраза")])
        snap = telemetry.snapshot()
        self.assertEqual(snap["hot_ngram_draws"], 1)
        self.assertEqual(snap["hot_ngram_empty_rate"], 0.0)

    async def test_channel_off_never_draws(self) -> None:
        pipeline, generator = self._pipeline([("горячая", "фраза")])
        pipeline._runtime_state.hot_ngram_seed_chance = 0.0  # type: ignore[attr-defined]
        await pipeline._hot_ngram_seed(  # type: ignore[attr-defined]
            SimpleNamespace(chat_id=CHAT), SimpleNamespace(address_reply=False)
        )
        snap = generator.telemetry.snapshot()
        self.assertEqual(snap["hot_ngram_draws"], 0)
        self.assertIsNone(snap["hot_ngram_empty_rate"])


if __name__ == "__main__":
    unittest.main()
