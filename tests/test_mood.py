from __future__ import annotations

import unittest

from app.core.mood import (
    CALM,
    HEATED,
    LIVELY,
    NEUTRAL_MODIFIERS,
    SLEEPY,
    ChatMoodState,
    MoodConfig,
    classify_mood,
    initial_mood_state,
    message_intensity,
    modifiers_for_mood,
    update_mood_state,
)

CONFIG = MoodConfig(
    ewma_alpha=0.4,
    lively_rate_per_min=12.0,
    sleepy_rate_per_min=2.0,
    heated_intensity=0.4,
    max_rate_per_min=120.0,
)


class TestMessageIntensity(unittest.TestCase):
    def test_plain_text_is_calm(self) -> None:
        self.assertEqual(message_intensity("привет как дела"), 0.0)

    def test_exclamations_raise_intensity(self) -> None:
        self.assertGreater(message_intensity("да!!!"), 0.0)

    def test_all_caps_raises_intensity(self) -> None:
        self.assertGreater(message_intensity("ЧТО ЗА БРЕД"), 0.4)

    def test_bounded_to_unit_interval(self) -> None:
        self.assertLessEqual(message_intensity("ЧТО?!?!?!?!!!"), 1.0)

    def test_empty_and_punct_only(self) -> None:
        self.assertEqual(message_intensity(""), 0.0)
        self.assertGreater(message_intensity("?!"), 0.0)


class TestClassifyMood(unittest.TestCase):
    def test_intensity_wins_first(self) -> None:
        self.assertEqual(
            classify_mood(rate_ewma=100.0, intensity_ewma=0.9, config=CONFIG), HEATED
        )

    def test_low_rate_is_sleepy(self) -> None:
        self.assertEqual(
            classify_mood(rate_ewma=1.0, intensity_ewma=0.0, config=CONFIG), SLEEPY
        )

    def test_high_rate_is_lively(self) -> None:
        self.assertEqual(
            classify_mood(rate_ewma=30.0, intensity_ewma=0.0, config=CONFIG), LIVELY
        )

    def test_midrange_is_calm(self) -> None:
        self.assertEqual(
            classify_mood(rate_ewma=6.0, intensity_ewma=0.0, config=CONFIG), CALM
        )


class TestUpdateMoodState(unittest.TestCase):
    def test_first_message_seeds_calm(self) -> None:
        state = update_mood_state(
            None, now=100.0, text="привет", mentioned=False, config=CONFIG
        )
        self.assertEqual(state.mood, CALM)
        self.assertEqual(state.last_ts, 100.0)
        self.assertEqual(state.intensity_ewma, 0.0)

    def test_rapid_calm_traffic_becomes_lively(self) -> None:
        # ~30 msg/min (every 2 s) of neutral text drives the rate EWMA up.
        state: ChatMoodState | None = None
        now = 0.0
        for _ in range(40):
            state = update_mood_state(
                state, now=now, text="ну что там по игре", mentioned=False, config=CONFIG
            )
            now += 2.0
        assert state is not None
        self.assertEqual(state.mood, LIVELY)

    def test_shouting_becomes_heated(self) -> None:
        state: ChatMoodState | None = None
        now = 0.0
        for _ in range(20):
            state = update_mood_state(
                state, now=now, text="ДА КАК ТЫ НЕ ПОНЯЛ!!!", mentioned=True, config=CONFIG
            )
            now += 5.0
        assert state is not None
        self.assertEqual(state.mood, HEATED)

    def test_long_silence_becomes_sleepy(self) -> None:
        # Start lively, then let very long gaps drag the rate EWMA down.
        state: ChatMoodState | None = None
        now = 0.0
        for _ in range(10):
            state = update_mood_state(
                state, now=now, text="движ", mentioned=False, config=CONFIG
            )
            now += 2.0
        for _ in range(15):
            now += 600.0  # 10-minute gaps
            state = update_mood_state(
                state, now=now, text="ау", mentioned=False, config=CONFIG
            )
        assert state is not None
        self.assertEqual(state.mood, SLEEPY)

    def test_simultaneous_messages_do_not_diverge(self) -> None:
        # dt == 0 must clamp to max_rate_per_min, not divide by zero.
        state = initial_mood_state(0.0, config=CONFIG)
        state = update_mood_state(
            state, now=0.0, text="спам", mentioned=False, config=CONFIG
        )
        self.assertLessEqual(state.rate_ewma, CONFIG.max_rate_per_min)

    def test_mention_ewma_tracks_addressing(self) -> None:
        state: ChatMoodState | None = None
        now = 0.0
        for _ in range(10):
            state = update_mood_state(
                state, now=now, text="эй бот", mentioned=True, config=CONFIG
            )
            now += 3.0
        assert state is not None
        self.assertGreater(state.mention_ewma, 0.5)


class TestMoodModifiers(unittest.TestCase):
    def test_calm_is_neutral(self) -> None:
        self.assertEqual(modifiers_for_mood(CALM, 1.0), NEUTRAL_MODIFIERS)

    def test_unknown_mood_is_neutral(self) -> None:
        self.assertEqual(modifiers_for_mood("???", 1.0), NEUTRAL_MODIFIERS)

    def test_strength_zero_collapses_to_neutral(self) -> None:
        for mood in (LIVELY, HEATED, SLEEPY):
            self.assertEqual(modifiers_for_mood(mood, 0.0), NEUTRAL_MODIFIERS)

    def test_lively_boosts_reply_probability(self) -> None:
        self.assertGreater(modifiers_for_mood(LIVELY, 1.0).reply_probability_mult, 1.0)

    def test_sleepy_suppresses_reply_probability(self) -> None:
        self.assertLess(modifiers_for_mood(SLEEPY, 1.0).reply_probability_mult, 1.0)

    def test_heated_adds_randomness_and_flavor(self) -> None:
        heated = modifiers_for_mood(HEATED, 1.0)
        self.assertGreater(heated.randomness_delta, 0.0)
        self.assertGreater(heated.flavor_strength_mult, 1.0)

    def test_strength_scales_deviation_linearly(self) -> None:
        full = modifiers_for_mood(LIVELY, 1.0).reply_probability_mult
        half = modifiers_for_mood(LIVELY, 0.5).reply_probability_mult
        # Half strength sits halfway between neutral (1.0) and full deviation.
        self.assertAlmostEqual(half, 1.0 + 0.5 * (full - 1.0))

    def test_sleepy_shifts_length_weights_shorter(self) -> None:
        weights = modifiers_for_mood(SLEEPY, 1.0).length_weight_mult
        self.assertGreater(weights[0], weights[2])  # short favoured over long


if __name__ == "__main__":
    unittest.main()
