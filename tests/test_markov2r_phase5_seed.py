"""Seed scoring for lexical anchoring (M2R-410, TZ §9.4).

The property this rests on: a distinctive, well-used token beats a junk unique
token, because seed choice is IDF × support × branching-band, not max-IDF.
"""

from __future__ import annotations

import math
import unittest

from app.core.seed import (
    branching_quality,
    idf,
    is_scorable,
    score_seeds,
)

# A middling branching value inside the default band used across the cases.
BAND = dict(min_support=5.0, branch_min=2.0, branch_ideal=6.0, branch_max=30.0,
            min_token_len=3)


def _uniform_branch(tokens: list[str], value: int) -> dict[str, int]:
    return {token: value for token in tokens}


class TestSeedRanksAboveRawIdf(unittest.TestCase):
    def test_junk_unique_token_loses_to_a_supported_distinctive_one(self) -> None:
        tokens = ["foobar123", "бобёр"]
        scored = score_seeds(
            tokens,
            df_of={"foobar123": 1, "бобёр": 4},
            support_of={"foobar123": 1, "бобёр": 40},
            forward_branch_of=_uniform_branch(tokens, 5),
            reverse_branch_of=_uniform_branch(tokens, 5),
            n_docs=1000,
            **BAND,
        )
        ranking = [s.token for s in scored]
        self.assertEqual(ranking[0], "бобёр")
        junk = next(s for s in scored if s.token == "foobar123")
        meme = next(s for s in scored if s.token == "бобёр")
        # foobar123 has the HIGHER raw idf but loses on support.
        self.assertGreater(junk.normalized_idf, 0.0)
        self.assertGreater(meme.score, junk.score)

    def test_branching_outside_the_band_drives_score_down(self) -> None:
        tokens = ["якорь", "разброс"]
        scored = score_seeds(
            tokens,
            df_of={"якорь": 5, "разброс": 5},
            support_of={"якорь": 40, "разброс": 40},
            # identical everything except branching: one inside the band, one
            # far above the ideal.
            forward_branch_of={"якорь": 6, "разброс": 500},
            reverse_branch_of={"якорь": 6, "разброс": 500},
            n_docs=1000,
            **BAND,
        )
        good = next(s for s in scored if s.token == "якорь")
        wide = next(s for s in scored if s.token == "разброс")
        self.assertEqual(good.branching, 1.0)
        self.assertEqual(wide.branching, 0.0)
        self.assertGreater(good.score, wide.score)

    def test_weakest_direction_governs(self) -> None:
        """A seed usable forward but dead in reverse is unusable."""
        scored = score_seeds(
            ["односторонний"],
            df_of={"односторонний": 5},
            support_of={"односторонний": 40},
            forward_branch_of={"односторонний": 6},
            reverse_branch_of={"односторонний": 1},  # below branch_min
            n_docs=1000,
            **BAND,
        )
        self.assertEqual(scored[0].branching, 0.0)
        self.assertEqual(scored[0].score, 0.0)


class TestFilters(unittest.TestCase):
    def test_stopwords_and_short_tokens_are_not_scored(self) -> None:
        self.assertFalse(is_scorable("the", min_token_len=3))
        self.assertFalse(is_scorable("ok", min_token_len=3))
        self.assertTrue(is_scorable("бобёр", min_token_len=3))

    def test_message_of_only_unscorable_tokens_yields_nothing(self) -> None:
        scored = score_seeds(
            ["the", "on", "ok"],
            df_of={}, support_of={}, forward_branch_of={},
            reverse_branch_of={}, n_docs=1000, **BAND,
        )
        self.assertEqual(scored, [])


class TestDegenerate(unittest.TestCase):
    def test_empty_corpus_yields_no_seeds(self) -> None:
        self.assertEqual(idf(0, 0), 0.0)
        scored = score_seeds(
            ["бобёр"],
            df_of={"бобёр": 0}, support_of={"бобёр": 3},
            forward_branch_of={"бобёр": 5}, reverse_branch_of={"бобёр": 5},
            n_docs=0, **BAND,
        )
        self.assertEqual(scored, [])

    def test_scores_are_finite_for_any_legal_counts(self) -> None:
        cases = [
            (0, 0, 1),   # df, support, branch — single-token corpus edge
            (1, 1, 1),
            (5, 40, 6),
            (1, 1, 10_000),
        ]
        for df_v, sup, br in cases:
            scored = score_seeds(
                ["t"],
                df_of={"t": df_v}, support_of={"t": sup},
                forward_branch_of={"t": br}, reverse_branch_of={"t": br},
                n_docs=10_000, min_support=5.0,
                branch_min=2.0, branch_ideal=6.0, branch_max=30.0,
                min_token_len=1,
            )
            for s in scored:
                self.assertTrue(math.isfinite(s.score), (df_v, sup, br))

    def test_branching_quality_handles_misordered_bounds(self) -> None:
        # ideal below minimum, and minimum >= maximum: must not raise.
        self.assertEqual(branching_quality(5, minimum=10, ideal=1, maximum=3), 0.0)
        self.assertEqual(branching_quality(2, minimum=2, ideal=8, maximum=6), 1.0)


if __name__ == "__main__":
    unittest.main()
