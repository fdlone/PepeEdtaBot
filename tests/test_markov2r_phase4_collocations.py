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
import unittest

from app.core.collocations import (
    MIN_SUPPORTABLE_JOINT_COUNT,
    lift,
    normalized_pmi,
    recency_factor,
    score_pair,
    support_factor,
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


if __name__ == "__main__":
    unittest.main()
