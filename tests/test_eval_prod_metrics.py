"""Unit tests for the pure metric helpers in tools/eval_prod.py."""
from __future__ import annotations

import unittest

from tools.eval_prod import (
    VERBATIM_MAX_N,
    VERBATIM_MIN_N,
    build_verbatim_index,
    longest_verbatim_run,
    novel_ngram_share,
    reply_has_seam,
)


class TestReplyHasSeam(unittest.TestCase):
    """M3R-143: честный числитель connective_reply_rate."""

    def test_silent_extension_seam_counts(self) -> None:
        # Молчаливая связка «.» невидима текст-скану — решает маршрут.
        self.assertTrue(
            reply_has_seam(["цитата", ".", "хвост"], "extension", 0)
        )

    def test_walk_jump_counts(self) -> None:
        self.assertTrue(reply_has_seam(["обычный", "ответ"], "vanilla", 1))

    def test_wordy_connective_counts_via_text_scan(self) -> None:
        self.assertTrue(
            reply_has_seam(["пиво", ",", "кстати", "холодное"], "vanilla", 0)
        )

    def test_plain_reply_without_seams_does_not_count(self) -> None:
        self.assertFalse(reply_has_seam(["обычный", "ответ"], "vanilla", 0))
        self.assertFalse(reply_has_seam(["обычный", "ответ"], None, 0))


class TestNovelNgramShare(unittest.TestCase):
    def setUp(self) -> None:
        # One training message: every 4-gram of it is "known corpus".
        self.index = build_verbatim_index(
            ["наш кот опять уронил ёлку на пол"]
        )

    def test_pure_quote_scores_zero(self) -> None:
        reply = ["наш", "кот", "опять", "уронил", "ёлку"]
        self.assertEqual(novel_ngram_share(reply, self.index), 0.0)

    def test_fully_novel_reply_scores_one(self) -> None:
        reply = ["собака", "съела", "весь", "новогодний", "оливье"]
        self.assertEqual(novel_ngram_share(reply, self.index), 1.0)

    def test_one_changed_word_breaks_surrounding_windows(self) -> None:
        # «ёлку» -> «тумбу»: windows covering the swapped word become novel.
        reply = ["наш", "кот", "опять", "уронил", "тумбу", "на", "пол"]
        share = novel_ngram_share(reply, self.index)
        assert share is not None
        # 4 windows total, только «наш кот опять уронил» остаётся корпусной.
        self.assertAlmostEqual(share, 0.75)

    def test_short_reply_is_not_evaluable(self) -> None:
        self.assertIsNone(novel_ngram_share(["кот", "уронил", "ёлку"], self.index))

    def test_agrees_with_longest_run_on_the_extremes(self) -> None:
        quote = ["наш", "кот", "опять", "уронил", "ёлку", "на", "пол"]
        self.assertEqual(novel_ngram_share(quote, self.index), 0.0)
        self.assertEqual(
            longest_verbatim_run(quote, self.index), min(VERBATIM_MAX_N, len(quote))
        )
        novel = ["пять", "разных", "слов", "без", "корпусных", "окон", "тут"]
        self.assertEqual(novel_ngram_share(novel, self.index), 1.0)
        self.assertLess(longest_verbatim_run(novel, self.index), VERBATIM_MIN_N)


if __name__ == "__main__":
    unittest.main()
