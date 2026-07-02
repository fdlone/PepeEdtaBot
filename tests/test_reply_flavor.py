from __future__ import annotations

import random
import unittest

from app.core.reply_flavor import (
    DOUBLE_TERMINAL_PROBABILITY,
    DROP_FINAL_PERIOD_PROBABILITY,
    ELLIPSIS_PROBABILITY,
    EXCLAMATION_PROBABILITY,
    apply_reply_flavor,
)


class TestApplyReplyFlavor(unittest.TestCase):
    def test_zero_strength_is_noop(self) -> None:
        for text in ("привет.", "как дела?", "ну да!", "без точки"):
            self.assertEqual(
                apply_reply_flavor(text, random.Random(1), strength=0.0), text
            )

    def test_empty_text_is_noop(self) -> None:
        self.assertEqual(apply_reply_flavor("", random.Random(1)), "")

    def test_only_ending_punctuation_changes(self) -> None:
        rng = random.Random(99)
        for _ in range(500):
            flavored = apply_reply_flavor("слова остаются на месте.", rng)
            self.assertTrue(flavored.startswith("слова остаются на месте"))
            self.assertIn(
                flavored.removeprefix("слова остаются на месте"),
                {"", ".", "...", "!"},
            )

    def test_period_transform_distribution(self) -> None:
        rng = random.Random(7)
        outcomes = {"": 0, ".": 0, "...": 0, "!": 0}
        rolls = 5000
        for _ in range(rolls):
            flavored = apply_reply_flavor("проверка формы.", rng)
            outcomes[flavored.removeprefix("проверка формы")] += 1
        self.assertAlmostEqual(
            outcomes[""] / rolls, DROP_FINAL_PERIOD_PROBABILITY, delta=0.03
        )
        self.assertAlmostEqual(
            outcomes["..."] / rolls, ELLIPSIS_PROBABILITY, delta=0.02
        )
        self.assertAlmostEqual(
            outcomes["!"] / rolls, EXCLAMATION_PROBABILITY, delta=0.02
        )
        self.assertGreater(outcomes["."], 0)

    def test_question_and_exclamation_can_double(self) -> None:
        rng = random.Random(13)
        doubled = 0
        rolls = 5000
        for _ in range(rolls):
            if apply_reply_flavor("серьёзно?", rng).endswith("??"):
                doubled += 1
        self.assertAlmostEqual(
            doubled / rolls, DOUBLE_TERMINAL_PROBABILITY, delta=0.02
        )

    def test_existing_ellipsis_and_doubles_are_untouched(self) -> None:
        rng = random.Random(3)
        for text in ("ну...", "да??", "нет!!"):
            for _ in range(100):
                self.assertEqual(apply_reply_flavor(text, rng), text)

    def test_single_period_text_is_not_emptied(self) -> None:
        rng = random.Random(5)
        for _ in range(200):
            self.assertEqual(apply_reply_flavor(".", rng), ".")


if __name__ == "__main__":
    unittest.main()
