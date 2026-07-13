from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.presentation.bot_messages import (
    TELEGRAM_COMMANDS,
    format_clear_confirmation_message,
    format_config_message,
    format_help_message,
    format_set_help_message,
    format_stats_message,
)


def make_state() -> SimpleNamespace:
    return SimpleNamespace(
        reply_probability=0.08,
        min_cooldown_sec=45,
        min_tokens_for_model=200,
        max_reply_chars=280,
        max_reply_tokens=45,
        normalize_lower=False,
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
    )


class TestBotMessages(unittest.TestCase):
    def test_help_message_groups_commands(self) -> None:
        text = format_help_message()

        self.assertIn("Основное:", text)
        self.assertIn("Настройки:", text)
        self.assertIn("Админское:", text)
        self.assertIn("/set help", text)
        self.assertIn("/pivo", text)
        self.assertIn("/pivo_on", text)
        self.assertIn("/pivo_off", text)
        self.assertIn("/pivo_privacy", text)
        self.assertNotIn("/seed", text)

    def test_help_dialogue_block_shows_registry_ranges(self) -> None:
        text = format_help_message()

        self.assertIn("Диалог (новое, PR50-58):", text)
        # Range hints are generated from the registry, not hard-coded.
        self.assertIn("/set mood_enabled - настроение чата (true/false)", text)
        self.assertIn("/set emoji_append_chance - эмодзи в ответах (0..1)", text)
        self.assertIn("/set reply_max_per_hour - лимит ответов в час (0..1000)", text)
        self.assertIn(
            "/set mention_cooldown_sec - пауза на упоминания, сек (0..3600)", text
        )

    def test_telegram_commands_are_registered(self) -> None:
        command_names = {command for command, _ in TELEGRAM_COMMANDS}

        self.assertIn("help", command_names)
        self.assertIn("pivo", command_names)
        self.assertIn("pivo_on", command_names)
        self.assertIn("pivo_off", command_names)
        self.assertIn("pivo_privacy", command_names)
        self.assertIn("set", command_names)
        self.assertIn("clear", command_names)
        self.assertNotIn("seed", command_names)

    def test_stats_message_is_compact_and_readable(self) -> None:
        text = format_stats_message({"messages": 10, "volume": 250})

        self.assertEqual("объём модели: 250", text)
        self.assertNotIn("сообщений", text)
        self.assertNotIn("готовность", text)
        self.assertNotIn("250/200", text)
        self.assertNotIn("transitions", text)

    def test_config_message_defaults_to_short_view(self) -> None:
        text = format_config_message(make_state())

        self.assertIn("шанс ответа: 0.08", text)
        self.assertIn("длина ответа: 45 токенов", text)
        self.assertNotIn("reply_context_start_bias", text)

    def test_config_message_full_includes_advanced_values(self) -> None:
        text = format_config_message(make_state(), full=True)

        self.assertIn("Дополнительно:", text)
        self.assertIn("reply_context_start_bias=2.2", text)

    def test_set_help_message_shows_common_keys(self) -> None:
        text = format_set_help_message()

        self.assertIn("/set reply_probability 0.08", text)
        self.assertIn("/set max_reply_tokens 45", text)
        self.assertIn("/config full", text)

    def test_clear_confirmation_message_requires_confirm(self) -> None:
        self.assertIn("/clear confirm", format_clear_confirmation_message())


if __name__ == "__main__":
    unittest.main()
