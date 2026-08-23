from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.config.registry import RUNTIME_FIELDS
from app.core.generation_telemetry import GenerationTelemetry, UserQuirkGate
from app.presentation.bot_messages import (
    TELEGRAM_COMMANDS,
    format_clear_confirmation_message,
    format_config_message,
    format_help_message,
    format_set_help_message,
    format_stats_message,
)


def make_state() -> SimpleNamespace:
    """Состояние с дефолтами реестра, кроме намеренных отклонений ниже.

    Собирается из ``RUNTIME_FIELDS``, а не перечислением полей руками:
    рукописный список расходится с реестром молча, и новая ручка ломает
    тесты, к которым отношения не имеет (O6). В прежнем списке уже
    недоставало 21 ручки — то есть расхождение было не гипотезой.

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
            "markov_entropy_pivot": 0.5,
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
        self.assertIn("markov_entropy_temp_gain=0.0", text)

    def test_config_message_full_includes_collocation_knobs(self) -> None:
        """Task 7.4: the Phase 4 scoring knobs are readable in /config full."""
        text = format_config_message(make_state(), full=True)

        self.assertIn("markov_collocation_bonus=0.0", text)
        self.assertIn("markov_collocation_break_penalty=0.0", text)
        self.assertIn("markov_hot_ngram_meme_ordering=False", text)

    def test_stats_message_shows_collocation_registry_by_status(self) -> None:
        """Task 7.4: registry size per status; an empty registry adds nothing."""
        text = format_stats_message(
            {"volume": 250}, collocations={"active": 7, "retired": 2}
        )
        self.assertIn("коллокации: active=7, retired=2", text)

        empty = format_stats_message({"volume": 250}, collocations={})
        self.assertNotIn("коллокации", empty)

    def test_stats_message_shows_collocation_effect_counters(self) -> None:
        """Telemetry spec: applied and withheld counts are the effect."""
        telemetry = {
            "generations": 4,
            "collocation_bonus_hits": 3,
            "collocation_penalty_hits": 1,
            "collocation_withheld": 2,
        }
        text = format_stats_message({"volume": 250}, telemetry=telemetry)
        self.assertIn("бонусов 3, штрафов 1, удержано 2", text)

    def test_stats_message_shows_meme_pass_cost(self) -> None:
        """Task 4.3: pass duration and scored pairs, even with no generations."""
        telemetry = {
            "generations": 0,
            "meme_passes": 2,
            "meme_scored_pairs": 500,
            "meme_mean_pass_ms": 41.2,
        }
        text = format_stats_message({"volume": 250}, telemetry=telemetry)
        self.assertIn("мем-анализ: 2 проходов, пар оценено 500", text)
        self.assertIn("41 мс", text)

    def test_stats_message_shows_the_quirk_funnel(self) -> None:
        """Каждый гейт причуд — со своим знаменателем, воронкой сверху вниз."""
        telemetry = GenerationTelemetry()
        telemetry.note_user_quirk_outcome(UserQuirkGate.ADDRESSED)
        telemetry.note_user_quirk_outcome(UserQuirkGate.ROLL)
        telemetry.note_user_quirk_outcome(None)

        text = format_stats_message({"volume": 250}, telemetry=telemetry.snapshot())

        self.assertIn("причуды: сработало 1 из 3 ответов", text)
        self.assertIn(
            "причуды по гейтам: адресность 1/3, ручка 0/2, сутки 0/2, "
            "розыгрыш 1/2, порог 0/1",
            text,
        )

    def test_quirk_funnel_is_printed_even_with_the_knob_at_zero(self) -> None:
        """Исчезающая строка неотличима от выключенного канала — печатаем нули."""
        text = format_stats_message(
            {"volume": 250}, telemetry=GenerationTelemetry().snapshot()
        )

        self.assertIn("причуды: сработало 0 из 0 ответов", text)
        self.assertIn("адресность 0/0", text)
        self.assertIn("порог 0/0", text)

    def test_quirk_funnel_line_carries_numbers_only(self) -> None:
        """Приватность: в воронке нет имён, идентификаторов и хэшей.

        Проверяется формой строки, а не отсутствием конкретной подстроки:
        счётчики агрегатны по процессу, и любое имя собственное в них — дефект.
        """
        telemetry = GenerationTelemetry()
        telemetry.note_user_quirk_outcome(UserQuirkGate.THRESHOLD)

        text = format_stats_message({"volume": 250}, telemetry=telemetry.snapshot())
        funnel = [
            line for line in text.splitlines() if line.startswith("причуды")
        ]

        self.assertEqual(len(funnel), 2)
        for line in funnel:
            self.assertRegex(line, r"^[А-Яа-яёЁ ]+: [А-Яа-яёЁ0-9 /,]+$")

    def test_set_help_message_shows_common_keys(self) -> None:
        text = format_set_help_message()

        self.assertIn("/set reply_probability 0.08", text)
        self.assertIn("/set max_reply_tokens 45", text)
        self.assertIn("/config full", text)

    def test_clear_confirmation_message_requires_confirm(self) -> None:
        self.assertIn("/clear confirm", format_clear_confirmation_message())


if __name__ == "__main__":
    unittest.main()
