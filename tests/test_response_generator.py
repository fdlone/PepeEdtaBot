from __future__ import annotations

import random
import unittest
from collections import deque
from unittest.mock import AsyncMock, MagicMock, call, patch

from app.core.candidate_scorer import CandidateScore
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
    state.backoff_min_order = 1
    state.normalize_lower = False
    state.fuzzy_context_casefold = False
    state.auto_capitalize_replies = False
    state.recent_short_replies = {}
    return state


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
        state = _runtime_state()
        state.recent_short_replies = {123: deque(["привет"], maxlen=5)}
        generator = AsyncMock()
        generator.generate_text = AsyncMock(
            side_effect=[
                "same current message",
                "Привет",
                "training sample has four tokens",
                "fresh response has four tokens",
            ]
        )
        learning_service = AsyncMock()
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

        self.assertEqual(result, "fresh response has four tokens")
        self.assertEqual(generator.generate_text.await_count, 4)
        scorer.assert_called_once()
        self.assertEqual(
            learning_service.is_verbatim_copy.await_args_list,
            [
                call(123, "training sample has four tokens"),
                call(123, "fresh response has four tokens"),
            ],
        )

    async def test_context_falls_back_after_context_attempt_budget(self) -> None:
        state = _runtime_state()
        generator = AsyncMock()
        generator.generate_text = AsyncMock(
            side_effect=[""] * GENERATION_ATTEMPTS_WITH_CONTEXT + ["accepted"]
        )
        learning_service = AsyncMock()
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
        generator = AsyncMock()
        generator.generate_text = AsyncMock(return_value="")
        learning_service = AsyncMock()
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
        generator = AsyncMock()
        generator.generate_text = AsyncMock(
            side_effect=["first candidate", "best candidate", "third candidate"]
        )
        learning_service = AsyncMock()
        scores = {
            "first candidate": _score(1.0),
            "best candidate": _score(3.0),
            "third candidate": _score(2.0),
        }
        scorer = MagicMock(side_effect=lambda text, tokens, context: scores[text])
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
        generator = AsyncMock()
        generator.generate_text = AsyncMock(
            side_effect=["same candidate"] * GENERATION_ATTEMPT_BUDGET
        )
        learning_service = AsyncMock()
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
        generator = AsyncMock()
        generator.generate_text = AsyncMock(side_effect=["first choice", "second choice"])
        learning_service = AsyncMock()
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

    async def test_auto_capitalization_only_changes_final_selected_text(self) -> None:
        candidate = "привет. hello!"
        scorer = MagicMock(return_value=_score(1.0))

        async def generate_with_flag(enabled: bool) -> str | None:
            state = _runtime_state()
            state.auto_capitalize_replies = enabled
            generator = AsyncMock()
            generator.generate_text = AsyncMock(return_value=candidate)
            response_generator = ResponseGenerator(
                generator=generator,
                learning_service=AsyncMock(),
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
