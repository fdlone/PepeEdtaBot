from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.config.registry import (
    RUNTIME_FIELDS,
    UNKNOWN_RUNTIME_KEY_MESSAGE,
    InvalidRuntimeSettingValueError,
    UnknownRuntimeSettingError,
    apply_runtime_setting,
)


def make_state() -> SimpleNamespace:
    """Состояние с дефолтами реестра, кроме намеренных отклонений ниже.

    Собирается из ``RUNTIME_FIELDS``, а не перечислением полей руками:
    рукописный список расходится с реестром молча, и новая ручка ломает
    тесты, к которым отношения не имеет (O6). В прежнем списке уже
    недоставало 29 ручки — то есть расхождение было не гипотезой.

    Отклонения ниже перенесены из прежнего списка дословно, чтобы
    поведение существующих тестов не изменилось.
    """
    return SimpleNamespace(
        **{
            **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS},
            "normalize_lower": False,
            "fuzzy_context_casefold": False,
            "context_jump_boost": 1.0,
            "markov_jump_probability": 0.04,
            "order_mix_probability": 0.0,
            "slot_mutation_probability": 0.0,
            "verbatim_penalty_strength": 1.0,
            "verbatim_extension_share": 0.0,
            "recent_reply_penalty_strength": 1.0,
            "length_context_adaptation": 0.0,
            "hot_ngram_seed_chance": 0.05,
            "rare_event_chance": 0.005,
            "false_start_chance": 0.03,
        },
    )


class TestRuntimeConfig(unittest.TestCase):
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

        self.assertEqual(state.user_quirk_chance, 0.3)
        self.assertEqual(state.user_quirk_min_interactions, 10)

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
