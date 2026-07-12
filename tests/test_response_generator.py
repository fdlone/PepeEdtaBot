from __future__ import annotations

import random
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from app.core.candidate_scorer import CandidateScore
from app.core.mood import MoodModifiers
from app.core.response_generator import (
    CANDIDATE_TARGET,
    GENERATION_ATTEMPT_BUDGET,
    GENERATION_ATTEMPTS_WITH_CONTEXT,
    GenerationRequest,
    ResponseGenerator,
)


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
    state.fuzzy_context_prefix = False
    state.auto_capitalize_replies = False
    state.recent_short_replies = {}
    state.recent_replies = {}
    state.recent_reply_penalty_strength = 1.0
    state.verbatim_penalty_strength = 0.0
    state.reply_context_emit_start = True
    state.length_mode_weights = (0.25, 0.55, 0.2)
    # Argmax selection and no ending transforms: existing tests assert the
    # best-scored candidate text verbatim.
    state.candidate_selection_temperature = 0.0
    state.reply_flavor_strength = 0.0
    # Emoji channel off: these tests assert the selected candidate verbatim and
    # must not consult the learning service's emoji stats.
    state.emoji_append_chance = 0.0
    state.markov_jump_probability = 0.0
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
    return service


def _traced_generator() -> AsyncMock:
    """MarkovGenerator mock whose generate_text_with_trace delegates to the
    plain generate_text AsyncMock tests configure, wrapping the text in the
    (text, trace) tuple the ResponseGenerator consumes."""
    generator = AsyncMock()

    async def _delegate(*args: object, **kwargs: object) -> tuple[str, SimpleNamespace]:
        text = await generator.generate_text(*args, **kwargs)
        return text, SimpleNamespace(markov_order_used=3, start_source="global")

    generator.generate_text_with_trace = AsyncMock(side_effect=_delegate)
    return generator


def _request() -> GenerationRequest:
    return GenerationRequest(
        chat_id=123,
        context_tokens=["reply", "context", "tokens"],
        seed=["reply", "context"],
        current_message_normalized="same current message",
    )


def _score(value: float) -> CandidateScore:
    return CandidateScore(value, 0.0, 0.0, 0.0, 0.0)


class TestResponseGenerator(unittest.IsolatedAsyncioTestCase):
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
        self.assertTrue(result.startswith("training sample has four tokens,"))
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

    async def test_coherence_penalty_applies_regardless_of_start_source(self) -> None:
        # A backed-off (order-2) walk is less coherent whatever anchored it:
        # the penalty must not be discounted for context-anchored candidates.
        # No overlap with the request's context tokens: the IDF fallback bonus
        # must stay at zero on both sides so only the coherence gap decides.
        ctx_text = "anchored walk backed off"
        global_text = "fluent full order walk"

        state = _runtime_state()
        state.candidate_selection_temperature = 0.0  # argmax
        generator = AsyncMock()
        generator.generate_text_with_trace = AsyncMock(
            side_effect=[
                (ctx_text, SimpleNamespace(markov_order_used=2, start_source="context")),
                (
                    global_text,
                    SimpleNamespace(markov_order_used=3, start_source="global"),
                ),
            ]
        )
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        scores = {ctx_text: _score(1.05), global_text: _score(1.0)}
        scorer = MagicMock(side_effect=lambda text, tokens, context, mode: scores[text])
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=scorer,
        )

        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            selected = await response_generator.generate(
                _request(), rng=random.Random(1), candidate_target=2
            )

        # ctx 1.05 - 0.10 (order-2 coherence) = 0.95 < global 1.0 -> global wins.
        self.assertEqual(selected, global_text)

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

        def fake_sample(weights: tuple[float, float, float], rng: object) -> str:
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

        def fake_flavor(text: str, rng: object, strength: float) -> str:
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
