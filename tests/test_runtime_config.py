from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.config.runtime_config import (
    UNKNOWN_RUNTIME_KEY_MESSAGE,
    InvalidRuntimeSettingValueError,
    UnknownRuntimeSettingError,
    apply_runtime_setting,
    parse_bool,
)


def make_state() -> SimpleNamespace:
    return SimpleNamespace(
        reply_probability=0.08,
        min_cooldown_sec=45,
        min_tokens_for_model=200,
        max_reply_chars=280,
        max_reply_tokens=45,
        normalize_lower=False,
        auto_capitalize_replies=False,
        typing_min_ms=350,
        typing_max_ms=1100,
        typing_per_char_ms=12,
        randomness_strength=2.0,
        candidate_selection_temperature=0.7,
        reply_flavor_strength=1.0,
        emoji_append_chance=0.15,
        repetition_penalty_strength=1.0,
        recent_reply_penalty_strength=1.0,
        verbatim_penalty_strength=1.0,
        length_mode_weights=(0.25, 0.55, 0.2),
        markov_order=3,
        enable_backoff=True,
        markov_jump_probability=0.04,
        hot_ngram_seed_chance=0.05,
        hot_ngram_min_count=3,
        hot_ngram_recency_share=0.5,
        rare_event_chance=0.005,
        false_start_chance=0.03,
        rare_event_daily_cap=3,
        user_quirk_chance=0.1,
        user_quirk_min_interactions=25,
        use_reply_context=True,
        fuzzy_context_casefold=False,
        fuzzy_context_stem=False,
        reply_context_max_tokens=12,
        reply_context_last_tokens=3,
        reply_context_bias=1.8,
        reply_context_start_bias=2.2,
        reply_context_only_for_replies=True,
        reply_context_include_current_message=True,
        pivo_recent_pool_window=5,
        pivo_temporal_flavor_chance=0.5,
        mood_enabled=True,
        mood_modulation_strength=1.0,
        mood_ewma_alpha=0.3,
        mood_lively_rate_per_min=12.0,
        mood_sleepy_rate_per_min=2.0,
        mood_heated_intensity=0.4,
        mood_max_rate_per_min=120.0,
        reply_director_enabled=True,
        reply_probability_min=0.02,
        reply_probability_max=0.30,
        reply_burst_boost_sec=180,
        reply_burst_boost_mult=2.0,
        reply_burst_suppress_sec=600,
        reply_burst_suppress_mult=0.5,
        reply_max_per_hour=20,
    )


class TestRuntimeConfig(unittest.TestCase):
    def test_parse_bool_accepts_supported_values(self) -> None:
        self.assertTrue(parse_bool("yes"))
        self.assertTrue(parse_bool("ON"))
        self.assertFalse(parse_bool("0"))
        self.assertFalse(parse_bool("false"))
        self.assertIsNone(parse_bool("maybe"))

    def test_apply_runtime_setting_updates_float_key(self) -> None:
        state = make_state()
        apply_runtime_setting(state, "reply_probability", "0.25")

        self.assertEqual(state.reply_probability, 0.25)

    def test_apply_runtime_setting_normalizes_key_name(self) -> None:
        state = make_state()
        apply_runtime_setting(state, "  NORMALIZE_LOWER  ", "true")

        self.assertTrue(state.normalize_lower)

    def test_apply_runtime_setting_updates_auto_capitalize_replies(self) -> None:
        state = make_state()
        apply_runtime_setting(state, "auto_capitalize_replies", "true")

        self.assertTrue(state.auto_capitalize_replies)

    def test_apply_runtime_setting_updates_fuzzy_context_casefold(self) -> None:
        state = make_state()
        apply_runtime_setting(state, "fuzzy_context_casefold", "true")

        self.assertTrue(state.fuzzy_context_casefold)

    def test_apply_runtime_setting_updates_fuzzy_context_stem(self) -> None:
        state = make_state()
        apply_runtime_setting(state, "fuzzy_context_stem", "true")

        self.assertTrue(state.fuzzy_context_stem)

    def test_apply_runtime_setting_updates_user_quirk_knobs(self) -> None:
        state = make_state()
        apply_runtime_setting(state, "user_quirk_chance", "0.5")
        apply_runtime_setting(state, "user_quirk_min_interactions", "10")

        self.assertEqual(state.user_quirk_chance, 0.5)
        self.assertEqual(state.user_quirk_min_interactions, 10)

    def test_apply_runtime_setting_rejects_bad_user_quirk_values(self) -> None:
        state = make_state()

        with self.assertRaises(InvalidRuntimeSettingValueError):
            apply_runtime_setting(state, "user_quirk_chance", "1.5")
        with self.assertRaises(InvalidRuntimeSettingValueError):
            apply_runtime_setting(state, "user_quirk_min_interactions", "0")

        self.assertEqual(state.user_quirk_chance, 0.1)
        self.assertEqual(state.user_quirk_min_interactions, 25)

    def test_apply_runtime_setting_rejects_probability_out_of_range(self) -> None:
        state = make_state()

        with self.assertRaises(InvalidRuntimeSettingValueError):
            apply_runtime_setting(state, "reply_probability", "1.5")

        self.assertEqual(state.reply_probability, 0.08)

    def test_apply_runtime_setting_updates_max_reply_tokens(self) -> None:
        state = make_state()
        apply_runtime_setting(state, "max_reply_tokens", "24")

        self.assertEqual(state.max_reply_tokens, 24)

    def test_apply_runtime_setting_rejects_max_reply_tokens_out_of_range(self) -> None:
        state = make_state()

        with self.assertRaises(InvalidRuntimeSettingValueError):
            apply_runtime_setting(state, "max_reply_tokens", "0")

        self.assertEqual(state.max_reply_tokens, 45)

    def test_apply_runtime_setting_rejects_typing_min_above_max(self) -> None:
        state = make_state()

        with self.assertRaises(InvalidRuntimeSettingValueError):
            apply_runtime_setting(state, "typing_min_ms", "1200")

        self.assertEqual(state.typing_min_ms, 350)

    def test_apply_runtime_setting_rejects_reply_context_last_tokens_above_max(
        self,
    ) -> None:
        state = make_state()
        state.reply_context_max_tokens = 2

        with self.assertRaises(InvalidRuntimeSettingValueError):
            apply_runtime_setting(state, "reply_context_last_tokens", "3")

        self.assertEqual(state.reply_context_last_tokens, 3)

    def test_apply_runtime_setting_rejects_unknown_key(self) -> None:
        state = make_state()

        with self.assertRaises(UnknownRuntimeSettingError):
            apply_runtime_setting(state, "unknown_key", "1")

    def test_unknown_key_message_lists_known_keys(self) -> None:
        self.assertIn("reply_probability", UNKNOWN_RUNTIME_KEY_MESSAGE)
        self.assertIn(
            "reply_context_include_current_message", UNKNOWN_RUNTIME_KEY_MESSAGE
        )
        self.assertIn("auto_capitalize_replies", UNKNOWN_RUNTIME_KEY_MESSAGE)


if __name__ == "__main__":
    unittest.main()
