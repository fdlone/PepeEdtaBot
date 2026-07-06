from __future__ import annotations

import random
import unittest

from app.core.reply_flavor import (
    DOUBLE_TERMINAL_PROBABILITY,
    DROP_FINAL_PERIOD_PROBABILITY,
    ELLIPSIS_PROBABILITY,
    EXCLAMATION_PROBABILITY,
    FALSE_START_FILLERS,
    RARE_EVENT_KINDS,
    VERDICT_WORDS,
    apply_rare_event,
    apply_reply_flavor,
    roll_rare_event,
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


class RareEventsTest(unittest.TestCase):
    def test_roll_disabled_at_zero_chances(self) -> None:
        rng = random.Random(1)
        for _ in range(200):
            self.assertIsNone(
                roll_rare_event(rng, event_chance=0.0, false_start_chance=0.0)
            )

    def test_roll_certain_event_returns_event_kind(self) -> None:
        rng = random.Random(1)
        kinds = {
            roll_rare_event(rng, event_chance=1.0, false_start_chance=0.0)
            for _ in range(60)
        }
        self.assertEqual(kinds, set(RARE_EVENT_KINDS))

    def test_roll_certain_false_start(self) -> None:
        rng = random.Random(1)
        self.assertEqual(
            roll_rare_event(rng, event_chance=0.0, false_start_chance=1.0),
            "false_start",
        )

    def test_verdict_is_single_word_from_pool(self) -> None:
        out = apply_rare_event("verdict", "какой-то длинный ответ", random.Random(1))
        self.assertEqual(len(out), 1)
        self.assertIn(out[0], VERDICT_WORDS)

    def test_caps_uppercases_reply(self) -> None:
        out = apply_rare_event("caps", "ну это интересно", random.Random(1))
        self.assertEqual(out, ["НУ ЭТО ИНТЕРЕСНО"])

    def test_double_splits_at_sentence_boundary(self) -> None:
        out = apply_rare_event(
            "double", "Первая мысль. Вторая мысль подлиннее", random.Random(1)
        )
        self.assertEqual(out, ["Первая мысль.", "Вторая мысль подлиннее"])

    def test_double_without_boundary_is_noop(self) -> None:
        out = apply_rare_event("double", "одно предложение без точки", random.Random(1))
        self.assertEqual(out, ["одно предложение без точки"])

    def test_false_start_prepends_filler(self) -> None:
        out = apply_rare_event("false_start", "настоящий ответ", random.Random(1))
        self.assertEqual(len(out), 2)
        self.assertIn(out[0], FALSE_START_FILLERS)
        self.assertEqual(out[1], "настоящий ответ")

    def test_unknown_kind_is_noop(self) -> None:
        out = apply_rare_event("nope", "текст", random.Random(1))
        self.assertEqual(out, ["текст"])


if __name__ == "__main__":
    unittest.main()
