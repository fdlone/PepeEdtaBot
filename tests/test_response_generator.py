from __future__ import annotations

import random
import unittest
from collections import Counter, deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from app.core.candidate_scorer import CandidateScore
from app.core.generation_telemetry import GenerationTelemetry
from app.core.markov import tokenize
from app.core.mood import MoodModifiers
from app.core.response_generator import (
    CANDIDATE_TARGET,
    GENERATION_ATTEMPT_BUDGET,
    GENERATION_ATTEMPTS_WITH_CONTEXT,
    GenerationRequest,
    ResponseGenerator,
)
from app.core.slot_mutation import frequencies_by_ending


def _runtime_state() -> MagicMock:
    state = MagicMock()
    state.randomness_strength = 0.5
    state.max_reply_chars = 280
    state.max_reply_tokens = 45
    state.reply_context_bias = 1.8
    state.reply_context_start_bias = 2.2
    state.repetition_penalty_strength = 1.0
    state.markov_order = 3
    state.enable_backoff = True
    state.normalize_lower = False
    state.fuzzy_context_casefold = False
    state.auto_capitalize_replies = False
    state.recent_short_replies = {}
    state.recent_replies = {}
    state.recent_reply_penalty_strength = 1.0
    state.verbatim_penalty_strength = 0.0
    state.length_mode_weights = (0.25, 0.55, 0.2)
    state.length_context_adaptation = 0.0
    # Argmax selection and no ending transforms: existing tests assert the
    # best-scored candidate text verbatim.
    state.candidate_selection_temperature = 0.0
    state.reply_flavor_strength = 0.0
    # Emoji channel off: these tests assert the selected candidate verbatim and
    # must not consult the learning service's emoji stats.
    state.emoji_append_chance = 0.0
    state.markov_jump_probability = 0.0
    state.context_jump_boost = 1.0
    state.verbatim_extension_share = 0.0
    state.order_mix_probability = 0.0
    # Off by default: a bare MagicMock compares truthy and would silently turn
    # the mutation channel on in every test here.
    state.slot_mutation_probability = 0.0
    state.hot_ngram_min_count = 3
    state.hot_ngram_recency_share = 0.5
    state.intonation_profile_strength = 0.0
    # Phase 2 knobs neutral, for the same reason as slot_mutation_probability
    # above: a bare MagicMock would silently turn both features on.
    state.markov_entropy_temp_gain = 0.0
    state.markov_entropy_pivot = 0.5
    state.markov_entropy_temp_min = 0.5
    state.markov_entropy_temp_max = 12.0
    state.markov_branching_degenerate_max = 0.0
    state.markov_branching_candidate_floor = 2
    # Phase 4 collocation weights neutral, same MagicMock-truthiness reason:
    # non-zero would send the pipeline to get_active_collocations on a mock.
    state.markov_collocation_bonus = 0.0
    state.markov_collocation_break_penalty = 0.0
    state.markov_hot_ngram_meme_ordering = False
    # Phase 5 seeded off by default: non-zero (or a bare MagicMock) would send
    # the pipeline to the seed ranking on a mock.
    state.markov_seeded_candidate_ratio = 0.0
    state.markov_seed_branch_min = 2.0
    state.markov_seed_branch_ideal = 6.0
    state.markov_seed_branch_max = 50.0
    state.markov_seed_min_support = 5.0
    state.markov_seed_min_score = 0.1
    state.markov_seed_min_token_len = 3
    state.markov_seed_head_share = 0.4
    return state


def _learning_service() -> AsyncMock:
    """LearningService mock whose corpus reads return real empty containers.

    A bare AsyncMock hands back a MagicMock for these, which happens to survive
    ``in`` checks but not ``max(idf.values())``. Returning the real empty types
    keeps the scorer on its documented no-corpus path.
    """
    service = AsyncMock()
    service.get_verbatim_ngram_index = AsyncMock(return_value=frozenset())
    service.get_context_idf = AsyncMock(return_value={})
    service.get_word_frequencies_by_ending = AsyncMock(return_value={})
    service.get_hot_ngrams = AsyncMock(return_value=[])
    service.get_intonation_profile = AsyncMock(return_value=None)
    return service


def _traced_generator() -> AsyncMock:
    """MarkovGenerator mock whose generate_text_with_trace delegates to the
    plain generate_text AsyncMock tests configure, wrapping the text in the
    (text, trace) tuple the ResponseGenerator consumes."""
    generator = AsyncMock()
    # Реальная телеметрия, а не мок: у AsyncMock каждый note_* возвращает
    # корутину, которую никто не ждёт, — предупреждение вместо учёта.
    generator.telemetry = GenerationTelemetry()

    async def _delegate(*args: object, **kwargs: object) -> tuple[str, SimpleNamespace]:
        text = await generator.generate_text(*args, **kwargs)
        return text, SimpleNamespace(markov_order_used=3, start_source="global")

    generator.generate_text_with_trace = AsyncMock(side_effect=_delegate)
    return generator


_SHORT_INCOMING = "а что?"
_LONG_INCOMING = " ".join(f"слово{index}" for index in range(20))


def _request() -> GenerationRequest:
    return GenerationRequest(
        chat_id=123,
        context_tokens=["reply", "context", "tokens"],
        seed=["reply", "context"],
        current_message_normalized="same current message",
    )


def _score(value: float) -> CandidateScore:
    return CandidateScore(value, 0.0, 0.0, 0.0, 0.0)


class TestMutatedTokensMatchTheSentText(unittest.IsolatedAsyncioTestCase):
    """Маршрут MUTATED скорит тот же объект, который уходит в чат.

    `detokenize` обрывает список токенов на символьном пределе, а
    `_mutated_variant` возвращал исходный, необрезанный. Скорер читал хвост,
    которого в ответе нет: длина, повторы, тематичность и анти-цитата
    считались по несуществующему тексту, а `is_short_generated_reply` —
    по «длинному» варианту. Это был единственный маршрут с таким
    расхождением: остальные три выводят токены из финального текста.
    """

    async def test_tokens_are_derived_from_the_truncated_text(self) -> None:
        state = _runtime_state()
        state.max_reply_chars = 60
        state.slot_mutation_probability = 1.0
        generator = _traced_generator()
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_learning_service(),
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        long_tokens = ["словечко"] * 30

        # Гейты приёмки к делу не относятся: тест про соответствие токенов
        # тексту, а не про то, пройдёт ли кандидат анти-повтор.
        with (
            patch(
                "app.core.response_generator.mutate_candidate_tokens",
                return_value=long_tokens,
            ),
            patch.object(
                ResponseGenerator,
                "_candidate_reject_reason",
                AsyncMock(return_value=None),
            ),
        ):
            variant = await response_generator._mutated_variant(
                _request(),
                long_tokens,
                frequencies={},
                protected_tokens=frozenset(),
                rng=random.Random(1),
            )

        assert variant is not None
        text, tokens = variant
        self.assertLessEqual(len(text), 60, "текст не обрезан — тест не о том")
        self.assertEqual(
            tokens,
            tokenize(text, normalize_lower=state.normalize_lower),
            "скорятся токены, которых нет в отправленном тексте",
        )
        self.assertLess(len(tokens), len(long_tokens), "обрезки не произошло")


class TestRouteDenominatorsOnEmptyPool(unittest.IsolatedAsyncioTestCase):
    """Генерация, не давшая ни одного кандидата, обязана попасть в знаменатель.

    Пара «маршрут отработал / маршрут положил кандидата в пул» (M3R-103)
    существует ровно затем, чтобы отличать выключенный маршрут от включённого
    и бесплодного. Пока `note_routes` стоял ниже раннего возврата по пустому
    пулу, самый сильный случай «отработал и ничего не дал» не считался вовсе —
    и обе доли, present/attempts и won/present, были смещены вверх, то есть в
    сторону «маршрут полезен». Для гейта промоушена (M3R-220) это худшее из
    направлений ошибки.
    """

    async def test_empty_pool_still_counts_the_attempt(self) -> None:
        state = _runtime_state()
        state.markov_seeded_candidate_ratio = 0.0
        state.slot_mutation_probability = 0.0
        generator = _traced_generator()
        # Пустой текст не проходит приёмку: пул остаётся пустым.
        generator.generate_text = AsyncMock(return_value="")
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_learning_service(),
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate_with_result(
                _request(), rng=random.Random(3), candidate_target=2
            )

        self.assertIsNone(result.text)
        vanilla = generator.telemetry.route_breakdown()["vanilla"]
        self.assertEqual(
            vanilla["attempts"], 1, "провальная генерация выпала из знаменателя"
        )
        self.assertEqual(vanilla["present"], 0)

    async def test_present_rate_is_not_inflated_by_skipped_failures(self) -> None:
        """Одна удачная и одна провальная генерация дают present_rate 1/2."""
        state = _runtime_state()
        state.markov_seeded_candidate_ratio = 0.0
        state.slot_mutation_probability = 0.0
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            # Первая генерация проваливается: пул пуст.
            await response_generator.generate_with_result(
                _request(), rng=random.Random(3), candidate_target=1
            )
            # Вторая отдаёт годного кандидата.
            generator.generate_text = AsyncMock(
                return_value="нормальный кандидат из четырёх слов"
            )
            await response_generator.generate_with_result(
                _request(), rng=random.Random(4), candidate_target=1
            )

        vanilla = generator.telemetry.route_breakdown()["vanilla"]
        self.assertEqual(vanilla["attempts"], 2)
        self.assertEqual(vanilla["present"], 1)
        self.assertAlmostEqual(vanilla["present_rate"], 0.5)


class TestResponseGenerator(unittest.IsolatedAsyncioTestCase):
    async def test_seeded_next_explore_is_mapped_to_probability(self) -> None:
        # randomness_strength is a 0-3 scale; markov.py consumes next_explore
        # as a probability (rng.random() < value). The raw strength (>= 1.0)
        # used to make exploration unconditional in seeded walks.
        state = _runtime_state()
        state.randomness_strength = 2.0
        state.markov_seeded_candidate_ratio = 0.5
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            return_value="обычный кандидат из четырёх слов"
        )
        generator.rank_seeds = AsyncMock(
            return_value=[SimpleNamespace(token="слово", score=1.0)]
        )
        generator.generate_seeded_candidate = AsyncMock(return_value=[])
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        with patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        ):
            # candidate_target=4, а не 1: с O10 маршрут берёт слоты изнутри
            # бюджета пула, и при пуле в одно место бюджет равен нулю —
            # конкурировать не за что, ветка не запускается (route_slot_budget).
            # Здесь проверяется отображение next_explore, а не бюджет.
            await response_generator.generate(
                _request(), rng=random.Random(1), candidate_target=4
            )
        kwargs = generator.generate_seeded_candidate.await_args.kwargs
        self.assertAlmostEqual(
            kwargs["next_explore"], min(0.98, 0.12 + 0.18 * 2.0)
        )

    async def test_acceptance_checks_keep_existing_order(self) -> None:
        # Echo and anti-repeat gates still discard; a verbatim training-sample
        # copy is EXTENDED with a fresh continuation instead of discarded.
        state = _runtime_state()
        state.recent_short_replies = {123: deque(["привет"], maxlen=5)}
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=[
                "same current message",
                "Привет",
                "training sample has four tokens",
                "fresh continuation has four tokens",
            ]
        )
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(side_effect=[True, False])
        scorer = MagicMock(return_value=_score(1.0))
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=scorer,
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate(
                _request(),
                rng=random.Random(7),
                candidate_target=1,
            )

        assert result is not None
        # Splice marker may be a wordy connective or the silent sentence
        # break — only the quote + continuation shape is pinned.
        self.assertTrue(result.startswith("training sample has four tokens"))
        self.assertIn("fresh continuation has four tokens", result)
        self.assertEqual(generator.generate_text.await_count, 4)
        scorer.assert_called_once()
        # Gate ran on the original copy, then again on the extended text.
        verbatim_calls = learning_service.is_verbatim_copy.await_args_list
        self.assertEqual(len(verbatim_calls), 2)
        self.assertEqual(
            verbatim_calls[0], call(123, "training sample has four tokens")
        )
        self.assertEqual(verbatim_calls[1], call(123, result))

    async def test_recognized_unit_guard_moves_the_penalty_not_the_extension(
        self,
    ) -> None:
        """M3R-120 / design D4: гард живёт в скорере и только в нём.

        Триггер verbatim-дописки сравнивается с СЫРОЙ долей — дописка
        существует, чтобы добавить отсебятину к почти-цитате, и признанная
        единица этого не отменяет. Проверяется по вызовам: при включённом
        гарде решение о дописке принято по тому же числу, что и при
        выключенном, а штраф изменился.
        """
        from app.core.candidate_scorer import verbatim_ngram_overlap

        tokens = tokenize("наш кот опять уронил ёлку а потом ушёл спать")
        content = [t.casefold() for t in tokens]
        corpus = {tuple(content[0:4]), tuple(content[1:5])}

        raw = verbatim_ngram_overlap(tokens, corpus)
        guarded = verbatim_ngram_overlap(
            tokens, corpus, exempt_recognized_unit=True
        )
        self.assertNotEqual(raw, guarded)

        state = _runtime_state()
        state.verbatim_recognized_unit = True
        state.verbatim_penalty_strength = 1.5
        generator = _traced_generator()
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_learning_service(),
            runtime_state=state,
        )

        score = response_generator._build_score(
            "наш кот опять уронил ёлку а потом ушёл спать",
            tokens,
            [],
            "medium",
            context_idf={},
            recent_trigrams=set(),
            recent_penalty_strength=0.0,
            corpus_ngrams=corpus,
            verbatim_penalty_strength=1.5,
            chat_id=123,
        )
        state.verbatim_recognized_unit = False
        unguarded = response_generator._build_score(
            "наш кот опять уронил ёлку а потом ушёл спать",
            tokens,
            [],
            "medium",
            context_idf={},
            recent_trigrams=set(),
            recent_penalty_strength=0.0,
            corpus_ngrams=corpus,
            verbatim_penalty_strength=1.5,
            chat_id=123,
        )
        # Штраф считается по доле с исключением единицы...
        from app.core.candidate_scorer import verbatim_quote_severity

        self.assertAlmostEqual(
            score.verbatim_penalty, 1.5 * verbatim_quote_severity(guarded)
        )
        # ...а без гарда — по сырой.
        self.assertAlmostEqual(
            unguarded.verbatim_penalty, 1.5 * verbatim_quote_severity(raw)
        )

    async def test_winner_route_names_the_extension_per_reply(self) -> None:
        # M3R-143: пер-ответный сигнал шва — молчаливую связку текст-скан не
        # видит по построению, маршрут победителя видит всегда.
        state = _runtime_state()
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=[
                "training sample has four tokens",
                "fresh continuation has four tokens",
            ]
        )
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(side_effect=[True, False])
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate_with_result(
                _request(), rng=random.Random(7), candidate_target=1
            )

        assert result.text is not None
        self.assertEqual(result.winner_route, "extension")

    async def test_no_reply_carries_no_winner_route(self) -> None:
        state = _runtime_state()
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_learning_service(),
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate_with_result(
                _request(), rng=random.Random(7), candidate_target=1
            )

        self.assertIsNone(result.text)
        self.assertIsNone(result.winner_route)

    async def test_context_falls_back_after_context_attempt_budget(self) -> None:
        state = _runtime_state()
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=[""] * GENERATION_ATTEMPTS_WITH_CONTEXT + ["accepted"]
        )
        learning_service = _learning_service()
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
        )
        rng = random.Random(11)
        request = _request()

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate(
                request,
                rng=rng,
                candidate_target=1,
            )

        self.assertEqual(result, "accepted")
        calls = generator.generate_text.await_args_list
        self.assertEqual(len(calls), GENERATION_ATTEMPTS_WITH_CONTEXT + 1)
        self.assertTrue(
            all(
                call.kwargs["context_tokens"] == request.context_tokens
                for call in calls[:GENERATION_ATTEMPTS_WITH_CONTEXT]
            )
        )
        self.assertIsNone(calls[GENERATION_ATTEMPTS_WITH_CONTEXT].kwargs["context_tokens"])
        self.assertEqual(calls[0].kwargs["seed_tokens"], request.seed)
        self.assertTrue(
            all(call.kwargs["seed_tokens"] is None for call in calls[1:])
        )
        self.assertTrue(all(call.kwargs["rng"] is rng for call in calls))
        strengths = [call.kwargs["randomness_strength"] for call in calls]
        self.assertEqual(strengths, sorted(strengths))
        self.assertEqual(strengths[0], state.randomness_strength)

    async def test_returns_none_after_single_bounded_budget(self) -> None:
        state = _runtime_state()
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        learning_service = _learning_service()
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate(
                _request(),
                rng=random.Random(13),
            )

        self.assertIsNone(result)
        self.assertEqual(generator.generate_text.await_count, GENERATION_ATTEMPT_BUDGET)
        self.assertTrue(
            all(
                call.kwargs["attempt_budget"] == 1
                for call in generator.generate_text.await_args_list
            )
        )

    async def test_collects_multiple_candidates_and_picks_highest_score(self) -> None:
        state = _runtime_state()
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=["first candidate", "best candidate", "third candidate"]
        )
        learning_service = _learning_service()
        scores = {
            "first candidate": _score(1.0),
            "best candidate": _score(3.0),
            "third candidate": _score(2.0),
        }
        scorer = MagicMock(
            side_effect=lambda text, tokens, context, length_mode: scores[text]
        )
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=scorer,
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate_with_result(
                _request(),
                rng=random.Random(17),
                candidate_target=3,
            )

        self.assertEqual(result.text, "best candidate")
        self.assertEqual(result.candidates_scored, 3)
        self.assertEqual(generator.generate_text.await_count, 3)
        self.assertEqual(scorer.call_count, 3)

    async def test_duplicate_candidates_are_scored_once(self) -> None:
        state = _runtime_state()
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=["same candidate"] * GENERATION_ATTEMPT_BUDGET
        )
        learning_service = _learning_service()
        scorer = MagicMock(return_value=_score(1.0))
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=scorer,
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate_with_result(
                _request(),
                rng=random.Random(19),
            )

        self.assertEqual(result.text, "same candidate")
        self.assertEqual(result.candidates_scored, 1)
        self.assertEqual(generator.generate_text.await_count, GENERATION_ATTEMPT_BUDGET)
        scorer.assert_called_once()

    async def test_equal_scores_use_first_seen_candidate(self) -> None:
        state = _runtime_state()
        generator = _traced_generator()
        generator.generate_text = AsyncMock(side_effect=["first choice", "second choice"])
        learning_service = _learning_service()
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(2.0)),
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate(
                _request(),
                rng=random.Random(23),
                candidate_target=2,
            )

        self.assertEqual(result, "first choice")
        self.assertLessEqual(CANDIDATE_TARGET, GENERATION_ATTEMPT_BUDGET)

    async def test_softmax_selection_stays_within_score_margin(self) -> None:
        from app.core.response_generator import (
            SELECTION_SCORE_MARGIN,
            _ScoredCandidate,
            select_scored_candidate,
        )

        candidates = [
            _ScoredCandidate(text="best", score=_score(3.0)),
            _ScoredCandidate(text="close", score=_score(3.0 - SELECTION_SCORE_MARGIN / 2)),
            _ScoredCandidate(text="weak", score=_score(0.0)),
        ]
        rng = random.Random(31)
        picked = {
            select_scored_candidate(candidates, 0.7, rng).text for _ in range(500)
        }
        self.assertIn("best", picked)
        self.assertIn("close", picked)
        self.assertNotIn("weak", picked)

    async def test_zero_temperature_is_argmax(self) -> None:
        from app.core.response_generator import (
            _ScoredCandidate,
            select_scored_candidate,
        )

        candidates = [
            _ScoredCandidate(text="best", score=_score(3.0)),
            _ScoredCandidate(text="close", score=_score(2.9)),
        ]
        rng = random.Random(37)
        for _ in range(50):
            self.assertEqual(
                select_scored_candidate(candidates, 0.0, rng).text, "best"
            )

    async def test_recent_full_reply_is_rejected_and_retried(self) -> None:
        state = _runtime_state()
        state.recent_replies = {
            123: deque(["дубль полного ответа"], maxlen=20)
        }
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=["Дубль полного ответа.", "свежий ответ на этот раз"]
        )
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        scorer = MagicMock(return_value=_score(1.0))
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=scorer,
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate(
                _request(),
                rng=random.Random(41),
                candidate_target=1,
            )

        self.assertEqual(result, "свежий ответ на этот раз")
        scorer.assert_called_once()

    async def test_recent_trigram_overlap_penalizes_candidate(self) -> None:
        state = _runtime_state()
        state.recent_replies = {
            123: deque(["один два три четыре пять"], maxlen=20)
        }
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=[
                "один два три четыре шесть",
                "совсем другой свежий ответ",
            ]
        )
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        scorer = MagicMock(return_value=_score(1.0))
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=scorer,
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate(
                _request(),
                rng=random.Random(43),
                candidate_target=2,
            )

        # Equal base scores: the trigram-overlap penalty must flip argmax
        # away from the first-seen (overlapping) candidate.
        self.assertEqual(result, "совсем другой свежий ответ")

    async def test_zero_recent_penalty_strength_disables_soft_penalty(self) -> None:
        state = _runtime_state()
        state.recent_reply_penalty_strength = 0.0
        state.recent_replies = {
            123: deque(["один два три четыре пять"], maxlen=20)
        }
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=[
                "один два три четыре шесть",
                "совсем другой свежий ответ",
            ]
        )
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate(
                _request(),
                rng=random.Random(47),
                candidate_target=2,
            )

        # Penalty off: ties resolve to the first-seen candidate again.
        self.assertEqual(result, "один два три четыре шесть")

    async def test_short_length_mode_caps_generator_max_tokens(self) -> None:
        from app.core.response_generator import SHORT_MODE_MAX_TOKENS

        state = _runtime_state()
        state.length_mode_weights = (1.0, 0.0, 0.0)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="короткий ответ")
        scorer = MagicMock(return_value=_score(1.0))
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_learning_service(),
            runtime_state=state,
            scorer=scorer,
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            await response_generator.generate(
                _request(),
                rng=random.Random(53),
                candidate_target=1,
            )

        self.assertEqual(
            generator.generate_text.await_args.kwargs["max_tokens"],
            SHORT_MODE_MAX_TOKENS,
        )
        self.assertEqual(scorer.call_args.args[3], "short")

    async def _length_mode_for_incoming(
        self, message: str, adaptation: float, seed: int
    ) -> str:
        state = _runtime_state()
        # Equal weights: whatever mode comes out was chosen by the tilt alone.
        state.length_mode_weights = (1.0, 1.0, 1.0)
        state.length_context_adaptation = adaptation
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="какой-то ответ")
        scorer = MagicMock(return_value=_score(1.0))
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_learning_service(),
            runtime_state=state,
            scorer=scorer,
        )
        request = replace(_request(), current_message_normalized=message)

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            await response_generator.generate(
                request,
                rng=random.Random(seed),
                candidate_target=1,
            )

        return str(scorer.call_args.args[3])

    async def _length_modes_over_seeds(
        self, message: str, adaptation: float
    ) -> Counter[str]:
        modes: Counter[str] = Counter()
        for seed in range(20):
            modes[await self._length_mode_for_incoming(message, adaptation, seed)] += 1
        return modes

    async def test_short_incoming_message_tilts_the_length_mode_short(self) -> None:
        modes = await self._length_modes_over_seeds(_SHORT_INCOMING, 3.0)
        self.assertGreater(modes["short"], modes["long"])
        self.assertEqual(modes.most_common(1)[0][0], "short")

    async def test_long_incoming_message_tilts_the_length_mode_long(self) -> None:
        modes = await self._length_modes_over_seeds(_LONG_INCOMING, 3.0)
        self.assertGreater(modes["long"], modes["short"])
        self.assertEqual(modes.most_common(1)[0][0], "long")

    async def test_length_adaptation_off_ignores_the_incoming_message(self) -> None:
        # Same seeds, same weights: with the knob off a two-word question and a
        # twenty-word rant must draw the same modes -- the incoming length is
        # not consulted at all.
        self.assertEqual(
            await self._length_modes_over_seeds(_SHORT_INCOMING, 0.0),
            await self._length_modes_over_seeds(_LONG_INCOMING, 0.0),
        )

    async def test_long_length_mode_keeps_max_tokens_and_reaches_scorer(
        self,
    ) -> None:
        state = _runtime_state()
        state.length_mode_weights = (0.0, 0.0, 1.0)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="длинный ответ важен")
        scorer = MagicMock(return_value=_score(1.0))
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_learning_service(),
            runtime_state=state,
            scorer=scorer,
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            await response_generator.generate(
                _request(),
                rng=random.Random(59),
                candidate_target=1,
            )

        self.assertEqual(
            generator.generate_text.await_args.kwargs["max_tokens"],
            state.max_reply_tokens,
        )
        self.assertEqual(scorer.call_args.args[3], "long")

    async def test_normalize_reply_for_repeat_strips_flavor_ending(self) -> None:
        from app.core.response_generator import normalize_reply_for_repeat

        self.assertEqual(
            normalize_reply_for_repeat("Привет как дела..."),
            normalize_reply_for_repeat("привет как дела!"),
        )
        self.assertEqual(
            normalize_reply_for_repeat("Привет как дела."),
            "привет как дела",
        )

    async def test_normalize_reply_for_repeat_strips_appended_emoji(self) -> None:
        from app.core.response_generator import normalize_reply_for_repeat

        # An M3 emoji flavor appended to the sent form must not defeat the exact
        # anti-repeat match against the pre-flavor candidate.
        self.assertEqual(
            normalize_reply_for_repeat("привет как дела 🍺"),
            normalize_reply_for_repeat("привет как дела"),
        )
        self.assertEqual(
            normalize_reply_for_repeat("привет как дела! 🔥"),
            "привет как дела",
        )

    async def test_remember_recent_reply_keeps_rolling_window(self) -> None:
        from app.core.response_generator import (
            RECENT_REPLY_LIMIT,
            remember_recent_reply,
        )

        state = _runtime_state()
        state.recent_replies = {}
        for index in range(RECENT_REPLY_LIMIT + 5):
            remember_recent_reply(state, 123, f"ответ номер {index}")

        recent = state.recent_replies[123]
        self.assertEqual(len(recent), RECENT_REPLY_LIMIT)
        self.assertNotIn("ответ номер 0", recent)
        self.assertIn(f"ответ номер {RECENT_REPLY_LIMIT + 4}", recent)

    async def test_flavor_strength_applies_to_selected_text(self) -> None:
        state = _runtime_state()
        state.reply_flavor_strength = 2.0
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="стабильный ответ.")
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_learning_service(),
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )

        endings: set[str] = set()
        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            for seed in range(40):
                result = await response_generator.generate(
                    _request(),
                    rng=random.Random(seed),
                    candidate_target=1,
                )
                assert result is not None
                self.assertTrue(result.startswith("стабильный ответ"))
                endings.add(result.removeprefix("стабильный ответ"))
        self.assertGreater(len(endings), 1)

    async def test_auto_capitalization_only_changes_final_selected_text(self) -> None:
        candidate = "привет. hello!"
        scorer = MagicMock(return_value=_score(1.0))

        async def generate_with_flag(enabled: bool) -> str | None:
            state = _runtime_state()
            state.auto_capitalize_replies = enabled
            generator = _traced_generator()
            generator.generate_text = AsyncMock(return_value=candidate)
            response_generator = ResponseGenerator(
                generator=generator,
                learning_service=_learning_service(),
                runtime_state=state,
                scorer=scorer,
            )
            return await response_generator.generate(
                _request(),
                rng=random.Random(29),
                candidate_target=1,
            )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            self.assertEqual(await generate_with_flag(False), candidate)
            self.assertEqual(await generate_with_flag(True), "Привет. Hello!")
        self.assertEqual(
            [call.args[0] for call in scorer.call_args_list],
            [candidate, candidate],
        )


class TestEntropySamplingSettings(unittest.TestCase):
    """M2R-100: the rollback path is a /set away, with no restart."""

    def _generator(self) -> ResponseGenerator:
        return ResponseGenerator(
            generator=_traced_generator(),
            learning_service=_learning_service(),
            runtime_state=_runtime_state(),
        )

    def test_settings_follow_the_runtime_state(self) -> None:
        response_generator = self._generator()
        self.assertEqual(response_generator.entropy_sampling.gain, 0.0)

        response_generator.runtime_state.markov_entropy_temp_gain = 0.6
        self.assertEqual(response_generator.entropy_sampling.gain, 0.6)

        # Reverting is the same path in reverse: no restart, no rebuild.
        response_generator.runtime_state.markov_entropy_temp_gain = 0.0
        self.assertEqual(response_generator.entropy_sampling.gain, 0.0)

    def test_clamp_and_pivot_come_from_the_state_too(self) -> None:
        response_generator = self._generator()
        state = response_generator.runtime_state
        state.markov_entropy_pivot = 0.42
        state.markov_entropy_temp_min = 1.0
        state.markov_entropy_temp_max = 8.0
        sampling = response_generator.entropy_sampling
        self.assertEqual(
            (sampling.pivot, sampling.temp_min, sampling.temp_max),
            (0.42, 1.0, 8.0),
        )


class TestBranchingAwareCandidateTarget(unittest.IsolatedAsyncioTestCase):
    """M2R-110: how much choice the walk had decides how many candidates to ask
    for. The unit-level rule lives in tests/test_markov2r_phase2.py; this checks
    that the candidate loop actually obeys it."""

    @staticmethod
    def _generator(branching: float) -> AsyncMock:
        generator = AsyncMock()

        async def _delegate(*args: object, **kwargs: object):
            text = await generator.generate_text(*args, **kwargs)
            return text, SimpleNamespace(
                markov_order_used=3,
                start_source="global",
                mean_branching=branching,
            )

        generator.generate_text_with_trace = AsyncMock(side_effect=_delegate)
        return generator

    async def _run(self, *, branching: float, degenerate_max: float) -> int:
        state = _runtime_state()
        state.markov_branching_degenerate_max = degenerate_max
        state.markov_branching_candidate_floor = 2
        generator = self._generator(branching)
        generator.generate_text = AsyncMock(
            side_effect=[f"кандидат номер {index}" for index in range(10)]
        )
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_learning_service(),
            runtime_state=state,
            scorer=MagicMock(side_effect=lambda *_: _score(1.0)),
        )
        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate_with_result(
                _request(), rng=random.Random(17), candidate_target=5
            )
        return result.candidates_scored

    async def test_degenerate_chain_stops_at_the_floor(self) -> None:
        self.assertEqual(await self._run(branching=1.0, degenerate_max=1.5), 2)

    async def test_wide_chain_reaches_the_full_target(self) -> None:
        self.assertEqual(await self._run(branching=9.0, degenerate_max=1.5), 5)

    async def test_disabled_knob_reaches_the_full_target(self) -> None:
        """Even on a maximally degenerate chain, 0 restores the old behaviour."""
        self.assertEqual(await self._run(branching=1.0, degenerate_max=0.0), 5)

    async def test_early_stop_never_returns_an_empty_reply(self) -> None:
        """A reduced target is a cap on accepted candidates, not on attempts:
        a chain that keeps failing the gates still gets the whole budget."""
        state = _runtime_state()
        state.markov_branching_degenerate_max = 1.5
        state.markov_branching_candidate_floor = 2
        generator = self._generator(1.0)
        # Nine dead attempts, then one usable candidate on the last try.
        generator.generate_text = AsyncMock(
            side_effect=[""] * 9 + ["единственный выживший кандидат"]
        )
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_learning_service(),
            runtime_state=state,
            scorer=MagicMock(side_effect=lambda *_: _score(1.0)),
        )
        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            result = await response_generator.generate_with_result(
                _request(), rng=random.Random(17), candidate_target=5
            )
        self.assertEqual(result.text, "единственный выживший кандидат")


class TestResponseGeneratorSlotMutation(unittest.IsolatedAsyncioTestCase):
    """P2: a slot-mutated copy competes in scoring next to its original."""

    def setUp(self) -> None:
        patcher = patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    _ORIGINAL = "завтра сегодня работа опять"
    _MUTATED = "завтра сегодня суббота опять"

    def _mutating_setup(self) -> tuple[AsyncMock, AsyncMock, MagicMock]:
        state = _runtime_state()
        state.slot_mutation_probability = 1.0
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value=self._ORIGINAL)
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_word_frequencies_by_ending = AsyncMock(
            return_value=frequencies_by_ending({"суббота": 10})
        )
        return state, generator, learning_service

    async def test_mutated_copy_competes_and_can_win(self) -> None:
        state, generator, learning_service = self._mutating_setup()
        scores = {self._ORIGINAL: _score(1.0), self._MUTATED: _score(2.0)}
        scorer = MagicMock(
            side_effect=lambda text, tokens, context, length_mode: scores[text]
        )
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=scorer,
        )

        result = await response_generator.generate_with_result(
            _request(),
            rng=random.Random(61),
            candidate_target=2,
        )

        self.assertEqual(result.text, self._MUTATED)
        self.assertEqual(result.candidates_scored, 2)
        learning_service.get_word_frequencies_by_ending.assert_awaited_once_with(123)
        learning_service.get_hot_ngrams.assert_awaited_once()
        # One real walk produced both candidates.
        self.assertEqual(generator.generate_text.await_count, 1)

    async def test_zero_probability_never_reads_frequencies(self) -> None:
        state = _runtime_state()
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value=self._ORIGINAL)
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )

        await response_generator.generate(
            _request(), rng=random.Random(67), candidate_target=1
        )

        learning_service.get_word_frequencies_by_ending.assert_not_awaited()
        learning_service.get_hot_ngrams.assert_not_awaited()

    async def test_mutated_copy_failing_gates_is_dropped(self) -> None:
        state, generator, learning_service = self._mutating_setup()
        generator.generate_text = AsyncMock(
            side_effect=[self._ORIGINAL] * GENERATION_ATTEMPT_BUDGET
        )
        # The mutated text is a verbatim training-sample copy: it must fall at
        # the same gate the original passed instead of joining the pool.
        learning_service.is_verbatim_copy = AsyncMock(
            side_effect=lambda _chat_id, text: text == self._MUTATED
        )
        scorer = MagicMock(return_value=_score(1.0))
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=scorer,
        )

        result = await response_generator.generate_with_result(
            _request(),
            rng=random.Random(71),
            candidate_target=2,
        )

        self.assertEqual(result.text, self._ORIGINAL)
        self.assertEqual(result.candidates_scored, 1)

    async def test_hot_ngram_words_are_never_mutated(self) -> None:
        state, generator, learning_service = self._mutating_setup()
        # Both mutable words belong to hot n-grams -> no eligible slot.
        learning_service.get_hot_ngrams = AsyncMock(
            return_value=[("сегодня", "работа")]
        )
        scorer = MagicMock(return_value=_score(1.0))
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=scorer,
        )

        result = await response_generator.generate_with_result(
            _request(),
            rng=random.Random(73),
            candidate_target=2,
        )

        self.assertEqual(result.text, self._ORIGINAL)
        self.assertEqual(result.candidates_scored, 1)


class TestResponseGeneratorIntonation(unittest.IsolatedAsyncioTestCase):
    """P4: the chat intonation profile blends into length weights and flavor."""

    def setUp(self) -> None:
        patcher = patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    _PROFILE_WEIGHTS = (0.7, 0.25, 0.05)

    def _profile(self):  # noqa: ANN202
        from app.core.intonation import IntonationProfile

        return IntonationProfile(
            length_weights=self._PROFILE_WEIGHTS,
            ending_none_share=0.8,
            ending_ellipsis_share=0.1,
            ending_exclamation_share=0.05,
        )

    def _generator(self) -> AsyncMock:
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="стабильный ответ длиной")
        return generator

    async def test_full_strength_replaces_length_weights(self) -> None:
        state = _runtime_state()
        state.intonation_profile_strength = 1.0
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_intonation_profile = AsyncMock(
            return_value=self._profile()
        )
        captured: list[tuple[float, float, float]] = []

        def fake_sample(
            weights: tuple[float, float, float],
            rng: object,
            base_weights: object = None,
        ) -> str:
            captured.append(weights)
            return "medium"

        rg = ResponseGenerator(
            generator=self._generator(),
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        with patch(
            "app.core.response_generator.sample_length_mode", side_effect=fake_sample
        ):
            await rg.generate(_request(), rng=random.Random(0), candidate_target=1)

        for got, want in zip(captured[0], self._PROFILE_WEIGHTS):
            self.assertAlmostEqual(got, want)
        learning_service.get_intonation_profile.assert_awaited_once_with(123)

    async def test_zero_strength_never_reads_profile(self) -> None:
        state = _runtime_state()
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        rg = ResponseGenerator(
            generator=self._generator(),
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        await rg.generate(_request(), rng=random.Random(3), candidate_target=1)
        learning_service.get_intonation_profile.assert_not_awaited()

    async def test_unprofiled_chat_keeps_base_weights(self) -> None:
        # Below the message floor the profile is None: base weights apply.
        state = _runtime_state()
        state.intonation_profile_strength = 1.0
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        captured: list[tuple[float, float, float]] = []

        def fake_sample(
            weights: tuple[float, float, float],
            rng: object,
            base_weights: object = None,
        ) -> str:
            captured.append(weights)
            return "medium"

        rg = ResponseGenerator(
            generator=self._generator(),
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        with patch(
            "app.core.response_generator.sample_length_mode", side_effect=fake_sample
        ):
            await rg.generate(_request(), rng=random.Random(5), candidate_target=1)

        self.assertEqual(captured[0], state.length_mode_weights)

    async def test_profile_reaches_reply_flavor(self) -> None:
        state = _runtime_state()
        state.intonation_profile_strength = 1.0
        state.reply_flavor_strength = 1.0
        profile = self._profile()
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_intonation_profile = AsyncMock(return_value=profile)
        captured: dict[str, object] = {}

        def fake_flavor(text: str, rng: object, strength: float, **kwargs: object) -> str:
            captured.update(kwargs)
            return text

        rg = ResponseGenerator(
            generator=self._generator(),
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        with patch(
            "app.core.response_generator.apply_reply_flavor", side_effect=fake_flavor
        ):
            await rg.generate(_request(), rng=random.Random(7), candidate_target=1)

        self.assertIs(captured["ending_profile"], profile)
        self.assertEqual(captured["profile_strength"], 1.0)


class TestResponseGeneratorMoodModulation(unittest.IsolatedAsyncioTestCase):
    """M1: mood modifiers adjust randomness, length weights and flavor strength."""

    def setUp(self) -> None:
        patcher = patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _generator(self) -> AsyncMock:
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="fresh reply has four tokens")
        return generator

    def _learning(self) -> AsyncMock:
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        return learning_service

    async def _first_randomness(self, modifiers: MoodModifiers | None) -> float:
        generator = self._generator()
        rg = ResponseGenerator(
            generator=generator,
            learning_service=self._learning(),
            runtime_state=_runtime_state(),
            mood_modifiers=modifiers,
        )
        await rg.generate(_request(), rng=random.Random(0))
        return generator.generate_text.await_args_list[0].kwargs["randomness_strength"]

    async def test_randomness_delta_raises_first_attempt_strength(self) -> None:
        neutral = await self._first_randomness(None)
        heated = await self._first_randomness(
            MoodModifiers(1.0, 0.5, (1.0, 1.0, 1.0), 1.0)
        )
        self.assertGreater(heated, neutral)

    async def test_negative_delta_clamped_at_zero(self) -> None:
        # base randomness 0.5, delta -0.9 -> clamped to 0.0, never negative.
        strength = await self._first_randomness(
            MoodModifiers(1.0, -0.9, (1.0, 1.0, 1.0), 1.0)
        )
        self.assertGreaterEqual(strength, 0.0)

    async def test_length_weights_are_scaled_by_modifiers(self) -> None:
        captured: list[tuple[float, float, float]] = []

        def fake_sample(
            weights: tuple[float, float, float],
            rng: object,
            base_weights: object = None,
        ) -> str:
            captured.append(weights)
            return "medium"

        rg = ResponseGenerator(
            generator=self._generator(),
            learning_service=self._learning(),
            runtime_state=_runtime_state(),  # base weights (0.25, 0.55, 0.2)
            mood_modifiers=MoodModifiers(1.0, 0.0, (2.0, 1.0, 0.5), 1.0),
        )
        with patch(
            "app.core.response_generator.sample_length_mode", side_effect=fake_sample
        ):
            await rg.generate(_request(), rng=random.Random(0))
        self.assertEqual(captured[0], (0.25 * 2.0, 0.55 * 1.0, 0.2 * 0.5))

    async def test_flavor_strength_is_scaled_by_modifiers(self) -> None:
        captured: list[float] = []

        def fake_flavor(
            text: str, rng: object, strength: float, **kwargs: object
        ) -> str:
            captured.append(strength)
            return text

        state = _runtime_state()
        state.reply_flavor_strength = 1.0
        rg = ResponseGenerator(
            generator=self._generator(),
            learning_service=self._learning(),
            runtime_state=state,
            mood_modifiers=MoodModifiers(1.0, 0.0, (1.0, 1.0, 1.0), 1.5),
        )
        with patch(
            "app.core.response_generator.apply_reply_flavor", side_effect=fake_flavor
        ):
            await rg.generate(_request(), rng=random.Random(0))
        self.assertAlmostEqual(captured[0], 1.5)
