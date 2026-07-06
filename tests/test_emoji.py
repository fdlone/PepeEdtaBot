from __future__ import annotations

import random
import unittest

from app.core.emoji import (
    append_emoji_flavor,
    count_emojis,
    extract_emojis,
    sample_emoji,
    strip_trailing_emojis,
)


class TestExtractEmojis(unittest.TestCase):
    def test_extracts_in_order_with_repeats(self) -> None:
        self.assertEqual(extract_emojis("привет 😂 огонь 🔥🔥"), ["😂", "🔥", "🔥"])

    def test_plain_text_has_none(self) -> None:
        self.assertEqual(extract_emojis("просто текст без эмодзи"), [])

    def test_empty_string(self) -> None:
        self.assertEqual(extract_emojis(""), [])

    def test_various_blocks(self) -> None:
        # emoticon, pictograph, transport, symbol, dingbat
        text = "😀 🍺 🚗 ⚽ ✂"
        self.assertEqual(len(extract_emojis(text)), 5)

    def test_skin_tone_modifier_not_split_off(self) -> None:
        # A skin-tone modifier must not be extracted as a bare glyph; it stays
        # attached to its base pictograph as one sequence.
        self.assertEqual(extract_emojis("👍🏽"), ["👍🏽"])
        self.assertNotIn("🏿", extract_emojis("привет 👋🏿 всем"))
        self.assertEqual(extract_emojis("привет 👋🏿 всем"), ["👋🏿"])

    def test_zwj_sequences_extracted_whole(self) -> None:
        # ZWJ-composed emojis (rainbow flag, pirate flag, family) must be one
        # sequence each — never split into base fragments.
        self.assertEqual(extract_emojis("флаг 🏳️‍🌈 тут"), ["🏳️‍🌈"])
        self.assertEqual(extract_emojis("пират 🏴‍☠️"), ["🏴‍☠️"])
        self.assertEqual(extract_emojis("семья 👨‍👩‍👧"), ["👨‍👩‍👧"])
        self.assertNotIn("🌈", extract_emojis("🏳️‍🌈"))

    def test_zwj_sequence_counts_as_one(self) -> None:
        counts = count_emojis("🏳️‍🌈 и ещё 🏳️‍🌈")
        self.assertEqual(counts["🏳️‍🌈"], 2)
        self.assertNotIn("🏳️", counts)

    def test_regional_indicators_folded_into_flag(self) -> None:
        self.assertEqual(extract_emojis("флаг 🇷🇺 тут"), ["🇷🇺"])
        self.assertEqual(extract_emojis("🇷🇺🇺🇸"), ["🇷🇺", "🇺🇸"])

    def test_lone_regional_indicator_dropped(self) -> None:
        self.assertEqual(extract_emojis("🇷 один"), [])

    def test_strip_trailing_emojis(self) -> None:
        self.assertEqual(strip_trailing_emojis("привет как дела 🍺"), "привет как дела")
        self.assertEqual(strip_trailing_emojis("ну что там! 🔥🔥"), "ну что там")
        self.assertEqual(strip_trailing_emojis("без эмодзи"), "без эмодзи")

    def test_strip_trailing_zwj_sequence(self) -> None:
        # An appended ZWJ emoji must be stripped whole — no dangling fragments.
        self.assertEqual(strip_trailing_emojis("ответ бота 🏳️‍🌈"), "ответ бота")
        self.assertEqual(strip_trailing_emojis("ответ 🏴‍☠️!"), "ответ")

    def test_strip_trailing_emojis_keeps_bare_punctuation(self) -> None:
        # A punctuation-only tail is not an emoji flavor and must survive.
        self.assertEqual(strip_trailing_emojis("привет..."), "привет...")
        self.assertEqual(strip_trailing_emojis("ну что там!"), "ну что там!")
        # Punctuation *after* the emoji still goes with it.
        self.assertEqual(strip_trailing_emojis("привет 🍺!"), "привет")

    def test_count_folds_frequency(self) -> None:
        counts = count_emojis("😂 текст 😂🔥")
        self.assertEqual(counts["😂"], 2)
        self.assertEqual(counts["🔥"], 1)


class TestSampleEmoji(unittest.TestCase):
    def test_none_when_empty(self) -> None:
        self.assertIsNone(sample_emoji({}, random.Random(0)))

    def test_none_when_all_non_positive(self) -> None:
        self.assertIsNone(sample_emoji({"😂": 0, "🔥": -3}, random.Random(0)))

    def test_single_emoji_always_returned(self) -> None:
        self.assertEqual(sample_emoji({"🍺": 5}, random.Random(1)), "🍺")

    def test_dominant_emoji_wins_more_often(self) -> None:
        rng = random.Random(42)
        picks = [sample_emoji({"😂": 100, "🔥": 1}, rng) for _ in range(200)]
        self.assertGreater(picks.count("😂"), picks.count("🔥"))

    def test_flattening_lifts_rare_emoji_vs_raw(self) -> None:
        # power < 1 must give the rare emoji a better shot than raw proportional.
        rng = random.Random(7)
        flat = [sample_emoji({"😂": 100, "🔥": 1}, rng, power=0.3) for _ in range(400)]
        rng = random.Random(7)
        raw = [sample_emoji({"😂": 100, "🔥": 1}, rng, power=1.0) for _ in range(400)]
        self.assertGreaterEqual(flat.count("🔥"), raw.count("🔥"))


class TestAppendEmojiFlavor(unittest.TestCase):
    def test_appends_when_roll_passes(self) -> None:
        out = append_emoji_flavor("норм текст", {"🍺": 5}, random.Random(3), chance=1.0)
        self.assertEqual(out, "норм текст 🍺")

    def test_no_append_when_chance_zero(self) -> None:
        out = append_emoji_flavor("норм", {"🍺": 5}, random.Random(3), chance=0.0)
        self.assertEqual(out, "норм")

    def test_no_append_without_stats(self) -> None:
        self.assertEqual(
            append_emoji_flavor("норм", {}, random.Random(3), chance=1.0), "норм"
        )

    def test_suppressed_after_question(self) -> None:
        out = append_emoji_flavor("как дела?", {"🍺": 5}, random.Random(3), chance=1.0)
        self.assertEqual(out, "как дела?")

    def test_heated_boosts_chance(self) -> None:
        # A chance that never fires calm can fire heated (boost > 1). Find a roll
        # value between chance and chance*boost via a stubbed rng.
        class _Rng(random.Random):
            def random(self) -> float:  # type: ignore[override]
                return 0.4

        calm = append_emoji_flavor(
            "текст", {"🍺": 5}, _Rng(), chance=0.3, heated=False
        )
        heated = append_emoji_flavor(
            "текст", {"🍺": 5}, _Rng(), chance=0.3, heated=True, heated_boost=2.0
        )
        self.assertEqual(calm, "текст")  # 0.4 >= 0.3 → no append
        self.assertEqual(heated, "текст 🍺")  # 0.4 < 0.6 → append

    def test_empty_text_unchanged(self) -> None:
        self.assertEqual(
            append_emoji_flavor("", {"🍺": 5}, random.Random(3), chance=1.0), ""
        )


if __name__ == "__main__":
    unittest.main()
