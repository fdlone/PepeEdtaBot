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
        randomness_strength=2.0,
        repetition_penalty_strength=1.0,
        markov_order=3,
        enable_backoff=True,
        backoff_min_order=1,
        use_reply_context=True,
        reply_context_max_tokens=12,
        reply_context_last_tokens=3,
        reply_context_bias=1.8,
        reply_context_start_bias=2.2,
        reply_context_only_for_replies=True,
        reply_context_include_current_message=True,
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

    def test_apply_runtime_setting_rejects_backoff_min_order_equal_to_markov_order(
        self,
    ) -> None:
        state = make_state()
        state.markov_order = 2

        with self.assertRaises(InvalidRuntimeSettingValueError):
            apply_runtime_setting(state, "backoff_min_order", "2")

        self.assertEqual(state.backoff_min_order, 1)

    def test_apply_runtime_setting_rejects_markov_order_when_backoff_conflicts(
        self,
    ) -> None:
        state = make_state()
        state.backoff_min_order = 2

        with self.assertRaises(InvalidRuntimeSettingValueError):
            apply_runtime_setting(state, "markov_order", "2")

        self.assertEqual(state.markov_order, 3)

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
