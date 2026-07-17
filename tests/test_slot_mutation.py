from __future__ import annotations

import random
import unittest

from app.core.slot_mutation import (
    eligible_slot_indexes,
    is_mutable_word,
    mutate_candidate_tokens,
    pick_replacement,
)

NO_PROTECTED: frozenset[str] = frozenset()
NO_CONTEXT: frozenset[str] = frozenset()


class TestIsMutableWord(unittest.TestCase):
    def test_accepts_long_content_word(self) -> None:
        self.assertTrue(is_mutable_word("работа"))

    def test_rejects_short_word(self) -> None:
        self.assertFalse(is_mutable_word("дом"))

    def test_rejects_stopword(self) -> None:
        self.assertFalse(is_mutable_word("что"))

    def test_rejects_non_alphabetic(self) -> None:
        self.assertFalse(is_mutable_word("2026"))
        self.assertFalse(is_mutable_word("привет1"))


class TestEligibleSlotIndexes(unittest.TestCase):
    def test_excludes_first_and_last_tokens(self) -> None:
        tokens = ["пятница", "сегодня", "работа", "хорошая", "погода"]
        slots = eligible_slot_indexes(
            tokens, protected_tokens=NO_PROTECTED, context_tokens=NO_CONTEXT
        )
        self.assertNotIn(0, slots)
        self.assertNotIn(len(tokens) - 1, slots)
        self.assertEqual(slots, [1, 2, 3])

    def test_excludes_context_overlap(self) -> None:
        tokens = ["завтра", "сегодня", "работа", "опять"]
        slots = eligible_slot_indexes(
            tokens,
            protected_tokens=NO_PROTECTED,
            context_tokens=frozenset({"работа"}),
        )
        self.assertEqual(slots, [1])

    def test_excludes_hot_ngram_words(self) -> None:
        tokens = ["завтра", "сегодня", "работа", "опять"]
        slots = eligible_slot_indexes(
            tokens,
            protected_tokens=frozenset({"сегодня"}),
            context_tokens=NO_CONTEXT,
        )
        self.assertEqual(slots, [2])

    def test_excludes_punctuation_and_stopwords(self) -> None:
        tokens = ["ну", "что", ",", "погода", "?"]
        slots = eligible_slot_indexes(
            tokens, protected_tokens=NO_PROTECTED, context_tokens=NO_CONTEXT
        )
        self.assertEqual(slots, [3])


class TestPickReplacement(unittest.TestCase):
    def test_requires_matching_ending(self) -> None:
        replacement = pick_replacement(
            "погода",
            {"свобода": 10, "пятница": 10},
            excluded_tokens=frozenset(),
            rng=random.Random(1),
        )
        self.assertEqual(replacement, "свобода")

    def test_requires_comparable_length(self) -> None:
        replacement = pick_replacement(
            "дела",
            {"переговорочка": 10},
            excluded_tokens=frozenset(),
            rng=random.Random(1),
        )
        self.assertIsNone(replacement)

    def test_rejects_rare_words(self) -> None:
        replacement = pick_replacement(
            "работа",
            {"погода": 2},
            excluded_tokens=frozenset(),
            rng=random.Random(1),
        )
        self.assertIsNone(replacement)

    def test_rejects_excluded_and_identical(self) -> None:
        replacement = pick_replacement(
            "работа",
            {"работа": 10, "погода": 10},
            excluded_tokens=frozenset({"погода"}),
            rng=random.Random(1),
        )
        self.assertIsNone(replacement)

    def test_prefers_frequent_words(self) -> None:
        rng = random.Random(42)
        picks = [
            pick_replacement(
                "погода",
                {"свобода": 90, "борода": 3},
                excluded_tokens=frozenset(),
                rng=rng,
            )
            for _ in range(50)
        ]
        self.assertGreater(picks.count("свобода"), picks.count("борода"))


class TestMutateCandidateTokens(unittest.TestCase):
    def test_mutates_exactly_one_slot(self) -> None:
        tokens = ["завтра", "сегодня", "работа", "опять"]
        mutated = mutate_candidate_tokens(
            tokens,
            frequencies={"суббота": 10},
            protected_tokens=NO_PROTECTED,
            context_tokens=(),
            rng=random.Random(7),
        )
        self.assertIsNotNone(mutated)
        assert mutated is not None
        diffs = [i for i, (a, b) in enumerate(zip(tokens, mutated)) if a != b]
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0], 2)
        self.assertEqual(mutated[2], "суббота")

    def test_original_tokens_untouched(self) -> None:
        tokens = ["завтра", "сегодня", "работа", "опять"]
        snapshot = list(tokens)
        mutate_candidate_tokens(
            tokens,
            frequencies={"суббота": 10},
            protected_tokens=NO_PROTECTED,
            context_tokens=(),
            rng=random.Random(7),
        )
        self.assertEqual(tokens, snapshot)

    def test_none_when_no_eligible_slot(self) -> None:
        mutated = mutate_candidate_tokens(
            ["ну", "да", "ок"],
            frequencies={"погода": 10},
            protected_tokens=NO_PROTECTED,
            context_tokens=(),
            rng=random.Random(7),
        )
        self.assertIsNone(mutated)

    def test_none_when_no_replacement_fits(self) -> None:
        mutated = mutate_candidate_tokens(
            ["завтра", "работа", "опять"],
            frequencies={"стол": 10},
            protected_tokens=NO_PROTECTED,
            context_tokens=(),
            rng=random.Random(7),
        )
        self.assertIsNone(mutated)

    def test_replacement_never_echoes_context(self) -> None:
        mutated = mutate_candidate_tokens(
            ["завтра", "работа", "опять"],
            frequencies={"суббота": 10},
            protected_tokens=NO_PROTECTED,
            context_tokens=("суббота",),
            rng=random.Random(7),
        )
        self.assertIsNone(mutated)


if __name__ == "__main__":
    unittest.main()
