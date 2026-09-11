"""Selection-window knobs and the diversity bonus (M3R-100, selection-knobs)."""
from __future__ import annotations

import random
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.config.registry import RUNTIME_FIELDS
from app.core.candidate_scorer import (
    CONTEXT_RELEVANCE_CAP,
    CONTEXT_RELEVANCE_WEIGHT,
    CandidateScore,
    idf_context_relevance,
)
from app.core.markov import tokenize
from app.core.response_generator import (
    SELECTION_SCORE_MARGIN,
    _ScoredCandidate,
    apply_diversity_bonus,
    select_scored_candidate,
)
from app.core.trajectory import EDGE_OVERLAP_SIMILAR
from tests.test_response_generator import (
    GenerationRequest,
    ResponseGenerator,
    _learning_service,
    _runtime_state,
    _score,
    _traced_generator,
)
from tests.test_route_slot_budget import PoolCompositionTestCase
from tools.eval.config import load_thresholds


def _cand(text: str, total: float) -> _ScoredCandidate:
    return _ScoredCandidate(text=text, score=CandidateScore(total, 0.0, 0.0, 0.0))


def _default(name: str) -> float:
    spec = next(spec for spec in RUNTIME_FIELDS if spec.name == name)
    return float(spec.parse(spec.default))


class TestKnobDefaults(unittest.TestCase):
    def test_registry_defaults_equal_the_constants(self) -> None:
        self.assertEqual(_default("selection_score_margin"), SELECTION_SCORE_MARGIN)
        self.assertEqual(_default("context_relevance_weight"), CONTEXT_RELEVANCE_WEIGHT)
        self.assertEqual(_default("context_relevance_cap"), CONTEXT_RELEVANCE_CAP)
        # Promoted 2026-09-11 (O20): 0.2 in ctx next to the phrase route; O19:
        # 0.2 in noctx. Both below the margin — a bonus above the margin
        # narrows the window (selection-grid verdict, d40).
        self.assertEqual(_default("selection_diversity_bonus"), 0.2)
        self.assertEqual(_default("selection_diversity_bonus_noctx"), 0.2)
        self.assertLess(_default("selection_diversity_bonus"), SELECTION_SCORE_MARGIN)
        self.assertLess(_default("selection_diversity_bonus_noctx"), SELECTION_SCORE_MARGIN)

    def test_similarity_threshold_equals_the_gate(self) -> None:
        gate = load_thresholds()["structural_escape"]["edge_overlap_similar"]
        self.assertEqual(EDGE_OVERLAP_SIMILAR, float(gate))


class TestRelevanceKnobs(unittest.TestCase):
    def test_weight_scales_and_cap_clips(self) -> None:
        context = tokenize("пиво сегодня вечером")
        tokens = tokenize("пиво вечером будет холодное")
        idf = {"пив": 2.0, "сегодн": 1.0, "вечер": 1.0}
        base = idf_context_relevance(tokens, context, idf)
        doubled = idf_context_relevance(tokens, context, idf, weight=3.2, cap=4.0)
        self.assertGreater(base, 0.0)
        self.assertAlmostEqual(doubled, base * 2, places=6)
        clipped = idf_context_relevance(tokens, context, idf, weight=3.2, cap=0.5)
        self.assertEqual(clipped, 0.5)


class TestMargin(unittest.TestCase):
    def test_wider_margin_lets_the_runner_up_win(self) -> None:
        pool = [_cand("best", 1.0), _cand("runner", 0.5)]
        narrow = {
            select_scored_candidate(pool, 0.7, random.Random(i), margin=0.3).text
            for i in range(100)
        }
        wide = {
            select_scored_candidate(pool, 0.7, random.Random(i), margin=0.8).text
            for i in range(100)
        }
        self.assertEqual(narrow, {"best"})
        self.assertIn("runner", wide)


class TestDiversityBonus(unittest.TestCase):
    def test_zero_returns_the_same_list(self) -> None:
        pool = [_cand("а б в", 1.0), _cand("г д е", 0.5)]
        self.assertIs(apply_diversity_bonus(pool, 0.0), pool)

    def test_distinct_candidate_is_lifted_and_best_is_not(self) -> None:
        pool = [_cand("пиво сегодня будет", 1.0), _cand("завтра дождь пойдёт", 0.5)]
        lifted = apply_diversity_bonus(pool, 0.2)
        self.assertEqual(lifted[0].score.total, 1.0)
        self.assertAlmostEqual(lifted[1].score.total, 0.7)
        self.assertAlmostEqual(lifted[1].score.diversity_bonus, 0.2)

    def test_walk_cut_short_is_not_lifted(self) -> None:
        pool = [_cand("пиво сегодня будет холодное", 1.0), _cand("пиво сегодня будет", 0.6)]
        lifted = apply_diversity_bonus(pool, 0.2)
        self.assertEqual(lifted[1].score.total, 0.6)

    def test_partial_overlap_scales_the_bonus(self) -> None:
        # Shares one edge of two: overlap 0.5 -> at the threshold, no lift;
        # shares one of three: overlap 1/3 -> lift of bonus * 2/3.
        pool = [_cand("а б в г", 1.0), _cand("а б х у", 0.5)]
        lifted = apply_diversity_bonus(pool, 0.3)
        self.assertAlmostEqual(lifted[1].score.diversity_bonus, 0.3 * (1 - 1 / 3))

    def test_single_candidate_untouched(self) -> None:
        pool = [_cand("один", 1.0)]
        self.assertIs(apply_diversity_bonus(pool, 0.5), pool)


if __name__ == "__main__":
    unittest.main()


class TestBonusByMode(PoolCompositionTestCase):
    """O19 (split-diversity-bonus-by-mode): one knob per context mode."""

    _TEXTS = (
        "холодное пиво вечером зашло отлично",
        "громкая музыка играла долго вчера",
        "красный дракон летит над городом",
        "старый кот ловит рыбу ловко",
        "новая сборка сломала прогон снова",
    )

    async def _pool(self, *, context: list[str]) -> list[object]:
        state = _runtime_state()
        state.selection_diversity_bonus = 0.0
        state.selection_diversity_bonus_noctx = 0.2
        texts = iter(self._TEXTS)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(side_effect=lambda *a, **k: next(texts))
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        request = GenerationRequest(
            chat_id=123, context_tokens=context, seed=None,
            current_message_normalized="пиво сегодня",
        )
        with patch("app.core.response_generator.mask_chat_id", return_value="chat"):
            await response_generator.generate_with_result(
                request, rng=random.Random(5), candidate_target=5
            )
        return self.captured[-1]

    async def test_reply_without_context_uses_the_noctx_knob(self) -> None:
        pool = await self._pool(context=[])
        self.assertTrue(any(c.score.diversity_bonus > 0 for c in pool), pool)

    async def test_reply_with_context_uses_the_context_knob(self) -> None:
        pool = await self._pool(context=["пиво", "сегодня"])
        self.assertTrue(all(c.score.diversity_bonus == 0 for c in pool), pool)
