"""Phase 4 association measures (M2R-300, TZ §10.1).

The property this phase rests on is that association ranks differently from
frequency. Measured on the prod copy, a raw-frequency-shaped ranking puts
one-character tokens on top while the pairs a human would call memes sit at
normalized PMI 0.91–0.94 — so "a frequent pair of frequent tokens must not
outrank a rare pair that always co-occurs" is the test that matters, not the
formula's algebra.
"""

from __future__ import annotations

import math
import random
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.collocations import (
    MIN_SUPPORTABLE_JOINT_COUNT,
    collocation_effect,
    lift,
    normalized_pmi,
    recency_factor,
    score_pair,
    support_factor,
)
from app.core.markov import MarkovGenerator, tokenize
from app.core.response_generator import GenerationRequest, ResponseGenerator
from tests.test_response_generator import (
    _learning_service,
    _runtime_state,
    _score,
    _traced_generator,
)

TOTAL = 10_000.0


class TestAssociationRanksAboveFrequency(unittest.TestCase):
    def test_frequent_pair_of_frequent_tokens_scores_below_a_devoted_pair(self) -> None:
        # "и" + "не": both everywhere, together no more than chance predicts.
        filler = score_pair(
            joint=100, left=2000, right=500, total=TOTAL,
            min_support=10, recent_share=1.0,
        )
        # A real meme: uncommon words that essentially only occur together.
        meme = score_pair(
            joint=30, left=32, right=31, total=TOTAL,
            min_support=10, recent_share=1.0,
        )
        self.assertGreater(meme.meme_score, filler.meme_score)
        self.assertGreater(meme.npmi, 0.8)

    def test_independent_pair_has_lift_about_one(self) -> None:
        # joint exactly as chance predicts: left*right/total
        value = lift(joint=100, left=2000, right=500, total=TOTAL)
        self.assertAlmostEqual(value, 1.0, places=6)

    def test_npmi_is_bounded(self) -> None:
        for joint, left, right in ((1, 1, 1), (30, 32, 31), (100, 2000, 500), (5, 9000, 9000)):
            value = normalized_pmi(joint, left, right, TOTAL)
            self.assertGreaterEqual(value, -1.0 - 1e-9)
            self.assertLessEqual(value, 1.0 + 1e-9)

    def test_llr_grows_with_frequency_which_is_why_it_is_not_the_ranking_key(
        self,
    ) -> None:
        """Documents the trap rather than the feature.

        Both pairs are equally exclusive (they always co-occur), but one is ten
        times more frequent. LLR prefers the frequent one; normalized PMI does
        not. Ranking by LLR would rebuild the frequency ranking this phase
        exists to replace.
        """
        rare = score_pair(joint=5, left=5, right=5, total=TOTAL,
                          min_support=5, recent_share=1.0)
        common = score_pair(joint=50, left=50, right=50, total=TOTAL,
                            min_support=5, recent_share=1.0)
        self.assertGreater(common.llr, rare.llr)
        self.assertAlmostEqual(common.npmi, rare.npmi, delta=0.15)


class TestFactors(unittest.TestCase):
    def test_support_factor_saturates(self) -> None:
        self.assertEqual(support_factor(10, 10), 1.0)
        self.assertEqual(support_factor(1000, 10), 1.0)
        self.assertAlmostEqual(support_factor(5, 10), 0.5)

    def test_support_factor_without_a_target_is_neutral(self) -> None:
        self.assertEqual(support_factor(3, 0), 1.0)

    def test_recency_factor_clamps(self) -> None:
        self.assertEqual(recency_factor(-0.5), 0.0)
        self.assertEqual(recency_factor(1.7), 1.0)
        self.assertAlmostEqual(recency_factor(0.4), 0.4)

    def test_stale_meme_falls_out_on_its_own(self) -> None:
        fresh = score_pair(joint=30, left=32, right=31, total=TOTAL,
                           min_support=10, recent_share=1.0)
        stale = score_pair(joint=30, left=32, right=31, total=TOTAL,
                           min_support=10, recent_share=0.05)
        self.assertLess(stale.meme_score, fresh.meme_score * 0.2)


class TestDegenerateInputs(unittest.TestCase):
    def test_scores_are_finite_for_any_legal_counts(self) -> None:
        cases = [
            (1, 1, 1, 1),
            (0, 5, 5, 100),
            (5, 5, 5, 5),
            (1, 10_000, 10_000, 10_000),
        ]
        for joint, left, right, total in cases:
            result = score_pair(
                joint=joint, left=left, right=right, total=total,
                min_support=2, recent_share=1.0,
            )
            for value in result:
                self.assertTrue(math.isfinite(value), (joint, left, right, total))

    def test_zero_joint_scores_zero(self) -> None:
        result = score_pair(joint=0, left=5, right=5, total=100,
                            min_support=2, recent_share=1.0)
        self.assertEqual(result.meme_score, 0.0)

    def test_negative_association_scores_zero_not_negative(self) -> None:
        """Two tokens that avoid each other are a different finding."""
        result = score_pair(joint=1, left=5000, right=5000, total=TOTAL,
                            min_support=1, recent_share=1.0)
        self.assertLess(result.npmi, 0.0)
        self.assertEqual(result.meme_score, 0.0)

    def test_hard_floor_rejects_single_occurrence_pairs(self) -> None:
        self.assertGreaterEqual(MIN_SUPPORTABLE_JOINT_COUNT, 2)


class TestCollocationEffect(unittest.TestCase):
    """The scoring side (M2R-320) — and the guard that is its whole point."""

    ACTIVE = frozenset({("кек", "лол")})

    def test_intact_reproduction_earns_the_bonus(self) -> None:
        effect = collocation_effect(
            ["сегодня", "кек", "лол", "опять"], self.ACTIVE, lambda _s, _t: True
        )
        self.assertEqual(effect.bonus_hits, 1)
        self.assertEqual(effect.penalty_hits, 0)

    def test_break_is_penalized_when_the_chain_offered_the_right_token(self) -> None:
        effect = collocation_effect(
            ["сегодня", "кек", "потом"], self.ACTIVE, lambda _s, _t: True
        )
        self.assertEqual(effect.penalty_hits, 1)
        self.assertEqual(effect.withheld, 0)

    def test_no_penalty_when_the_right_token_was_never_available(self) -> None:
        """The guard: otherwise we punish the candidate for the corpus."""
        effect = collocation_effect(
            ["сегодня", "кек", "потом"], self.ACTIVE, lambda _s, _t: False
        )
        self.assertEqual(effect.penalty_hits, 0)
        self.assertEqual(effect.withheld, 1)

    def test_no_active_collocations_is_a_no_op(self) -> None:
        effect = collocation_effect(
            ["кек", "лол"], frozenset(), lambda _s, _t: True
        )
        self.assertEqual(effect, (0, 0, 0))

    def test_neutral_weights_produce_no_adjustment(self) -> None:
        effect = collocation_effect(
            ["кек", "лол", "кек", "потом"], self.ACTIVE, lambda _s, _t: True
        )
        self.assertGreater(effect.bonus_hits, 0)
        self.assertEqual(effect.delta(0.0, 0.0), 0.0)

    def test_delta_uses_both_weights(self) -> None:
        effect = collocation_effect(
            ["кек", "лол", "тут", "кек", "иначе"], self.ACTIVE, lambda _s, _t: True
        )
        self.assertEqual(effect.bonus_hits, 1)
        self.assertEqual(effect.penalty_hits, 1)
        self.assertAlmostEqual(effect.delta(0.5, 0.2), 0.3)

    def test_availability_is_asked_with_the_preceding_state(self) -> None:
        """The scorer must ask about the real state, not a bare token."""
        seen: list[tuple[tuple[str, ...], str]] = []

        def probe(state: tuple[str, ...], token: str) -> bool:
            seen.append((state, token))
            return False

        collocation_effect(["вчера", "кек", "потом"], self.ACTIVE, probe)
        self.assertEqual(seen, [(("вчера", "кек"), "лол")])


class TestAvailabilityFromCachedPools(unittest.TestCase):
    """D4: availability answers come from pools walks already cached — no SQL."""

    def setUp(self) -> None:
        self.generator = MarkovGenerator(db=MagicMock())
        # TransitionRow: (token, cnt, s_value, s_updated_at).
        self.generator._cache2[(1, "вчера", "кек")] = [("лол", 2, 2.0, 0)]
        self.generator._cache3[(1, "мы", "вчера", "кек")] = [("лол", 1, 1.0, 0)]

    def test_cached_order2_pool_answers(self) -> None:
        self.assertTrue(
            self.generator.transition_was_available(1, ("вчера", "кек"), "лол")
        )
        self.assertFalse(
            self.generator.transition_was_available(1, ("вчера", "кек"), "нет")
        )

    def test_order3_pool_answers_when_order2_is_not_cached(self) -> None:
        self.assertTrue(
            self.generator.transition_was_available(
                2, ("мы", "вчера", "кек"), "лол"
            )
            is False
        )
        self.assertTrue(
            self.generator.transition_was_available(
                1, ("мы", "вчера", "кек"), "лол"
            )
        )

    def test_unknown_pool_means_unavailable_not_a_query(self) -> None:
        """A cache miss withholds the penalty; it never falls through to SQL."""
        self.assertFalse(
            self.generator.transition_was_available(1, ("не", "кэшировано"), "лол")
        )
        self.generator.db.get_transitions.assert_not_called()
        self.generator.db.get_transitions3.assert_not_called()


def _collocation_pipeline(
    *,
    candidates: list[str],
    bonus: float,
    penalty: float,
    active: frozenset[tuple[str, str]],
    available: bool,
) -> tuple[ResponseGenerator, AsyncMock, MagicMock]:
    """ResponseGenerator wired for M2R-320 tests: equal base scores, argmax."""
    state = _runtime_state()
    state.markov_collocation_bonus = bonus
    state.markov_collocation_break_penalty = penalty
    generator = _traced_generator()
    generator.generate_text = AsyncMock(side_effect=candidates)
    generator.transition_was_available = MagicMock(return_value=available)
    generator.telemetry = MagicMock()
    service = _learning_service()
    service.is_verbatim_copy = AsyncMock(return_value=False)
    service.get_active_collocations = AsyncMock(return_value=active)
    response_generator = ResponseGenerator(
        generator=generator,
        learning_service=service,
        runtime_state=state,
        scorer=MagicMock(return_value=_score(1.0)),
    )
    return response_generator, service, generator.telemetry


def _request() -> GenerationRequest:
    return GenerationRequest(
        chat_id=123,
        context_tokens=[],
        seed=None,
        current_message_normalized="совсем другое сообщение",
    )


async def _generate(pipeline: ResponseGenerator, target: int) -> str | None:
    with patch(
        "app.core.response_generator.mask_chat_id", return_value="chat"
    ):
        return await pipeline.generate(
            _request(), rng=random.Random(7), candidate_target=target
        )


class TestPipelineCollocationScoring(unittest.IsolatedAsyncioTestCase):
    """M2R-320 in the reply pipeline: the effect reaches candidate selection."""

    ACTIVE = frozenset({("кек", "лол")})

    async def test_bonus_moves_selection(self) -> None:
        pipeline, _, telemetry = _collocation_pipeline(
            candidates=[
                "сегодня без единой пары тут",
                "сегодня кек лол опять тут",
            ],
            bonus=0.5,
            penalty=0.0,
            active=self.ACTIVE,
            available=True,
        )
        text = await _generate(pipeline, 2)
        self.assertEqual(text, "сегодня кек лол опять тут")
        counted = sum(
            kwargs["bonus_hits"]
            for _, kwargs in telemetry.note_collocations.call_args_list
        )
        self.assertEqual(counted, 1)

    async def test_break_penalty_moves_selection_when_available(self) -> None:
        pipeline, _, telemetry = _collocation_pipeline(
            candidates=[
                "сегодня кек потом опять тут",
                "сегодня без единой пары тут",
            ],
            bonus=0.0,
            penalty=0.7,
            active=self.ACTIVE,
            available=True,
        )
        text = await _generate(pipeline, 2)
        self.assertEqual(text, "сегодня без единой пары тут")
        counted = sum(
            kwargs["penalty_hits"]
            for _, kwargs in telemetry.note_collocations.call_args_list
        )
        self.assertEqual(counted, 1)

    async def test_withheld_break_does_not_move_selection(self) -> None:
        """The guard through the whole pipeline: unavailable ⇒ no penalty."""
        pipeline, _, telemetry = _collocation_pipeline(
            candidates=[
                "сегодня кек потом опять тут",
                "сегодня без единой пары тут",
            ],
            bonus=0.0,
            penalty=0.7,
            active=self.ACTIVE,
            available=False,
        )
        text = await _generate(pipeline, 2)
        # Equal totals, argmax keeps the first accepted candidate.
        self.assertEqual(text, "сегодня кек потом опять тут")
        counted = sum(
            kwargs["withheld"]
            for _, kwargs in telemetry.note_collocations.call_args_list
        )
        self.assertEqual(counted, 1)

    async def test_registry_is_read_once_per_reply(self) -> None:
        pipeline, service, _ = _collocation_pipeline(
            candidates=[
                "сегодня кек лол опять тут",
                "сегодня без единой пары тут",
            ],
            bonus=0.5,
            penalty=0.7,
            active=self.ACTIVE,
            available=True,
        )
        await _generate(pipeline, 2)
        self.assertEqual(service.get_active_collocations.await_count, 1)

    async def test_neutral_weights_never_touch_the_registry(self) -> None:
        """Neutral defaults add no query — the generation_hash contract."""
        pipeline, service, telemetry = _collocation_pipeline(
            candidates=["сегодня кек лол опять тут"],
            bonus=0.0,
            penalty=0.0,
            active=self.ACTIVE,
            available=True,
        )
        text = await _generate(pipeline, 1)
        self.assertIsNotNone(text)
        service.get_active_collocations.assert_not_awaited()
        telemetry.note_collocations.assert_not_called()


class TestTokenizationUntouched(unittest.IsolatedAsyncioTestCase):
    """ADR-016: collocations influence scoring only, tokenization is shared.

    The learn path and the candidate scorer both call ``app.core.markov
    .tokenize`` with the chat's ``normalize_lower``. With collocations active,
    the tokens the scorer sees must be exactly that tokenization — two separate
    tokens, never a glued «кек лол» unit.
    """

    async def test_scorer_sees_plain_tokenization_with_collocations_active(
        self,
    ) -> None:
        candidate = "сегодня кек лол опять тут"
        seen_tokens: list[list[str]] = []

        def spy_scorer(
            _text: str, tokens: list[str], _context: list[str], _mode: str
        ):
            seen_tokens.append(tokens)
            return _score(1.0)

        pipeline, _, _ = _collocation_pipeline(
            candidates=[candidate],
            bonus=0.5,
            penalty=0.7,
            active=frozenset({("кек", "лол")}),
            available=True,
        )
        pipeline.scorer = spy_scorer
        await _generate(pipeline, 1)
        self.assertEqual(seen_tokens, [tokenize(candidate, normalize_lower=False)])
        self.assertTrue(
            all(" " not in token for token in seen_tokens[0]),
            "glued collocation token leaked into scoring",
        )


class TestMemePassTelemetry(unittest.IsolatedAsyncioTestCase):
    """Task 4.3: each analyzer pass lands in process telemetry, not only logs."""

    async def test_maintenance_pass_records_duration_and_scored_pairs(
        self,
    ) -> None:
        from app.infrastructure.database import MaintenanceOutcome
        from app.services.learning_service import LearningService
        from app.services.meme_analyzer import AnalysisResult

        db = MagicMock()
        db.decay_flavor_stats_if_due = AsyncMock(
            return_value=MaintenanceOutcome.DONE
        )
        db.list_chat_ids = AsyncMock(return_value=[1])
        generator = MagicMock(spec=MarkovGenerator)
        generator.telemetry = MagicMock()
        service = LearningService(db, generator)
        with patch(
            "app.services.learning_service.analyze_chat_memes",
            AsyncMock(
                return_value=AnalysisResult(
                    scored_pairs=500, stored_pairs=100, duration_ms=41.0
                )
            ),
        ), patch(
            "app.services.learning_service.mask_chat_id", return_value="chat"
        ):
            await service.run_due_maintenance()
        generator.telemetry.note_meme_pass.assert_called_once_with(
            scored_pairs=500, duration_ms=41.0
        )


if __name__ == "__main__":
    unittest.main()
