from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.config.registry import RUNTIME_FIELDS, runtime_field_names
from app.core.generation_telemetry import GenerationTelemetry, UserQuirkGate
from app.presentation.bot_messages import (
    TELEGRAM_COMMANDS,
    format_clear_confirmation_message,
    format_config_message,
    format_help_message,
    format_set_help_message,
    format_stats_message,
    split_for_telegram,
)


@contextmanager
def build_stamp(content: str | None) -> Iterator[None]:
    """Подменить штамп сборки: ``None`` — файла нет, строка — таково содержимое.

    Через реальный файл, а не мок чтения: проверяется в том числе то, что
    штамп читается с диска и обрезается по краям, а мок ``read_text``
    обошёл бы ровно эту часть.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "BUILD_AT"
        if content is not None:
            path.write_text(content, encoding="utf-8")
        with mock.patch(
            "app.presentation.bot_messages._BUILD_STAMP_PATH", path
        ):
            yield


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
        # С переводом строки на конце: именно так его пишет `date -Is >`.
        with build_stamp("2026-08-24T12:33:01+00:00\n"):
            text = format_stats_message({"messages": 10, "volume": 250})

        self.assertEqual(
            "сборка: 2026-08-24T12:33:01+00:00\nобъём модели: 250", text
        )
        self.assertNotIn("сообщений", text)
        self.assertNotIn("готовность", text)
        self.assertNotIn("250/200", text)
        self.assertNotIn("transitions", text)

    def test_stats_message_always_names_the_build(self) -> None:
        """Без штампа строка обязана остаться и сказать `unknown`.

        Условная печать вернула бы неоднозначность, ради которой строка и
        появилась: отсутствие строки счётчика ниже читается как «нечего
        показать» только при известной сборке. Пустой файл проверяется отдельно
        от отсутствующего: `date` с перенаправлением создаёт файл раньше, чем
        пишет в него, поэтому пустой штамп — достижимое состояние, а не
        гипотеза.
        """
        for label, content in (("нет файла", None), ("пустой файл", "")):
            with self.subTest(label), build_stamp(content):
                first = format_stats_message({"volume": 1}).splitlines()[0]
                self.assertEqual("сборка: unknown", first)

    def test_config_message_defaults_to_short_view(self) -> None:
        text = format_config_message(make_state())

        self.assertIn("длина ответа: 45 токенов", text)
        self.assertNotIn("reply_context_start_bias", text)

    def test_config_shows_the_band_while_the_director_is_on(self) -> None:
        """При включённом директоре печатается полоса, а не плоская ручка.

        `reply_probability` при `reply_director_enabled=true` (дефолт реестра)
        не читается вовсе: шанс ведёт полоса [min..max] по моментуму беседы.
        Прежняя форма теста пинила `шанс ответа: 0.08` — то есть закрепляла
        как контракт показ ручки, которая ни на что не влияет, и именно она
        стоит первой строкой, куда человек смотрит раньше всего.
        """
        state = make_state()
        state.reply_director_enabled = True

        text = format_config_message(state)

        self.assertIn(
            f"шанс ответа: {state.reply_probability_min}..{state.reply_probability_max}",
            text,
        )
        self.assertIn("директор", text)

    def test_config_shows_the_flat_knob_when_the_director_is_off(self) -> None:
        """Выключенный директор возвращает прежнюю форму — ручка снова живая."""
        state = make_state()
        state.reply_director_enabled = False

        text = format_config_message(state)

        self.assertIn(f"шанс ответа: {state.reply_probability}", text)
        self.assertNotIn("директор", text)

    def test_tempo_lines_print_without_any_generation(self) -> None:
        """Темп и обращения видны, даже когда бот с рестарта не отвечал.

        Блок телеметрии стоит за гейтом «были ли генерации», и это верно для
        счётчиков генерации. Темп чата и обращения наблюдаются на входящих
        сообщениях, а чат, где бот молчит, — ровно тот случай, ради которого
        счётчик заведён (O14/O15). Спрятать их за тем же гейтом значило бы
        гасить измерение там, где оно информативнее всего.
        """
        telemetry: dict[str, float | int | None] = {
            "generations": 0,
            "tempo_observations": 4,
            "tempo_share_штиль": 0.5,
            "tempo_share_кипит": 0.25,
            "mentions_observed": 2,
            "mention_answer_share": 0.5,
            "mention_answers_peak_hour": 1,
        }

        text = format_stats_message({"messages": 10, "volume": 250}, telemetry)

        self.assertIn("темп чата", text)
        self.assertIn("обращений: 2", text)
        self.assertNotIn("генераций с рестарта", text)

    def test_tempo_lines_absent_when_nothing_observed(self) -> None:
        """Пустой знаменатель — строки нет вовсе, а не нуля."""
        text = format_stats_message({"messages": 10, "volume": 250}, {})

        self.assertNotIn("темп чата", text)
        self.assertNotIn("обращений:", text)
        self.assertNotIn("берст-ритм", text)

    def test_config_full_covers_every_runtime_field(self) -> None:
        """«Полный список» проверяем, а не обещаем.

        Спека chat-scoped-settings требовала «полный набор действующих
        значений» с самого начала, но блок вёлся руками и показывал 52 ключа
        из 97: невидимыми были весь директор ответа, настроение, причуды,
        слот-мутации и `pivo_mention_by_id` — ручка отката O2. Расхождение
        прожило незамеченным именно потому, что требование держалось словом.
        Здесь оно держится реестром.
        """
        text = format_config_message(make_state(), full=True)

        missing = [
            name for name in runtime_field_names() if f"{name}=" not in text
        ]
        self.assertEqual(missing, [], f"не попали в /config full: {missing}")

    def test_config_full_is_split_before_the_telegram_limit(self) -> None:
        """Вывод режется по строкам и не теряет ни одной настройки."""
        text = format_config_message(make_state(), full=True)
        parts = split_for_telegram(text, limit=400)

        self.assertGreater(len(parts), 1, "тест не проверяет разбиение")
        for part in parts:
            self.assertLessEqual(len(part), 400)
        rejoined = "\n".join(parts)
        missing = [
            name for name in runtime_field_names() if f"{name}=" not in rejoined
        ]
        self.assertEqual(missing, [], f"потеряны при разбиении: {missing}")

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

    def test_stats_message_shows_assoc_draw_counters(self) -> None:
        """assoc-route-pilot: the empty-draw pair is printed; None = not asked."""
        telemetry = {"generations": 3, "assoc_draws": 4, "assoc_empty_rate": 0.25}
        text = format_stats_message({"volume": 250}, telemetry=telemetry)
        self.assertIn("ассоциаты: пусто в 25% из 4 розыгрышей", text)
        silent = format_stats_message(
            {"volume": 250}, telemetry={"generations": 3, "assoc_empty_rate": None}
        )
        self.assertNotIn("ассоциаты", silent)

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

    def test_stats_message_shows_rejection_classes(self) -> None:
        """Классы отказов (M3R-021) — с отдельным разрядом неразмеченных."""
        telemetry = GenerationTelemetry()
        telemetry.note_route_rejected("vanilla", "context_heavy")
        telemetry.note_route_rejected("seeded", "no_starts")

        text = format_stats_message(
            {"volume": 250},
            rejection_classes=telemetry.rejections_by_class(),
        )

        self.assertIn("отклонения по классам: F2_context_copy 1", text)
        self.assertIn("не размечено 1", text)

    def test_stats_message_omits_rejection_classes_without_data(self) -> None:
        text = format_stats_message(
            {"volume": 250},
            rejection_classes=GenerationTelemetry().rejections_by_class(),
        )

        self.assertNotIn("отклонения по классам", text)

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
