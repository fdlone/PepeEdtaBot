from __future__ import annotations

import unittest

from app.core.intonation import (
    MIN_LENGTH_MODE_SHARE,
    IntonationProfile,
    blend_length_weights,
    build_intonation_profile,
)

SHORT = "ну да"
MEDIUM = "сегодня отличная погода для прогулки по парку"
LONG = " ".join(f"слово{index}" for index in range(16))


class TestBuildIntonationProfile(unittest.TestCase):
    def test_none_below_message_floor(self) -> None:
        self.assertIsNone(
            build_intonation_profile([MEDIUM] * 10, min_messages=11)
        )

    def test_length_shares_reflect_distribution(self) -> None:
        messages = [SHORT] * 60 + [MEDIUM] * 30 + [LONG] * 10
        profile = build_intonation_profile(messages, min_messages=10)
        assert profile is not None
        short, medium, long_ = profile.length_weights
        self.assertAlmostEqual(short + medium + long_, 1.0, places=6)
        self.assertGreater(short, medium)
        self.assertGreater(medium, long_)

    def test_missing_mode_gets_floor_not_zero(self) -> None:
        profile = build_intonation_profile([SHORT] * 50, min_messages=10)
        assert profile is not None
        self.assertGreaterEqual(
            profile.length_weights[2], MIN_LENGTH_MODE_SHARE / 2
        )
        self.assertGreater(profile.length_weights[2], 0.0)

    def test_ending_shares_counted(self) -> None:
        messages = (
            ["как дела"] * 5  # no terminal punctuation
            + ["как дела..."] * 3
            + ["как дела!"] * 1
            + ["как дела."] * 1
        )
        profile = build_intonation_profile(messages, min_messages=5)
        assert profile is not None
        self.assertAlmostEqual(profile.ending_none_share, 0.5)
        self.assertAlmostEqual(profile.ending_ellipsis_share, 0.3)
        self.assertAlmostEqual(profile.ending_exclamation_share, 0.1)

    def test_question_and_period_do_not_count_as_none(self) -> None:
        profile = build_intonation_profile(
            ["как дела?"] * 6 + ["как дела."] * 6, min_messages=5
        )
        assert profile is not None
        self.assertEqual(profile.ending_none_share, 0.0)

    def test_blank_and_punctuation_only_messages_skipped(self) -> None:
        self.assertIsNone(
            build_intonation_profile(["", "   ", "..."] * 20, min_messages=1)
        )


class TestBlendLengthWeights(unittest.TestCase):
    _PROFILE = IntonationProfile(
        length_weights=(0.7, 0.25, 0.05),
        ending_none_share=0.0,
        ending_ellipsis_share=0.0,
        ending_exclamation_share=0.0,
    )

    def test_zero_strength_returns_base_untouched(self) -> None:
        base = (0.25, 0.55, 0.2)
        self.assertEqual(
            blend_length_weights(base, self._PROFILE.length_weights, 0.0),
            base,
        )

    def test_full_strength_returns_profile(self) -> None:
        blended = blend_length_weights(
            (0.25, 0.55, 0.2), self._PROFILE.length_weights, 1.0
        )
        for got, want in zip(blended, self._PROFILE.length_weights):
            self.assertAlmostEqual(got, want)

    def test_half_strength_lands_between(self) -> None:
        blended = blend_length_weights(
            (0.25, 0.55, 0.2), self._PROFILE.length_weights, 0.5
        )
        self.assertGreater(blended[0], 0.25)
        self.assertLess(blended[0], 0.7)
        self.assertAlmostEqual(sum(blended), 1.0, places=6)

    def test_strength_above_one_is_clamped(self) -> None:
        self.assertEqual(
            blend_length_weights(
                (0.25, 0.55, 0.2), self._PROFILE.length_weights, 5.0
            ),
            blend_length_weights(
                (0.25, 0.55, 0.2), self._PROFILE.length_weights, 1.0
            ),
        )


if __name__ == "__main__":
    unittest.main()
