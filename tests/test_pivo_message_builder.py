from __future__ import annotations

import random
import re
import unittest
from collections.abc import Iterable
from datetime import datetime
from unittest.mock import patch

from app.domain.pivo_templates import (
    PIVO_DEFAULT_BODY_PARTS,
    PIVO_DEFAULT_BOTTOM_PARTS,
    PIVO_DEFAULT_TARGET_INTROS,
    PIVO_DEFAULT_TOP_PARTS,
    PIVO_TARGET_BODY_PARTS,
    PIVO_TARGET_BOTTOM_PARTS,
    PIVO_TARGET_INTROS,
    PIVO_TARGET_TOP_PARTS,
)
from app.services.pivo_message_builder import (
    PivoMessageContext,
    PivoMessageGenerator,
    _format_time_value,
    build_pivo_message_context,
)


def build_pivo_message(
    mentions: str,
    *,
    planned_time: str | None = None,
    target: str | None = None,
    has_explicit_mentions: bool = False,
    rng: random.Random | None = None,
) -> str:
    """Сборка сообщения одной строкой — сокращение для тестов.

    В рантайме этой формы нет: сервис собирает контекст и вызывает генератор
    отдельно, потому что ему нужны ещё и вытянутые индексы шаблонов для
    анти-повтора. Обёртка жила в продовом модуле, хотя вызывалась только
    отсюда.
    """
    context = build_pivo_message_context(
        mentions,
        planned_time=planned_time,
        target=target,
        has_explicit_mentions=has_explicit_mentions,
        rng=rng,
    )
    return PivoMessageGenerator().build(context, rng=rng).text

RANDOM_CHOICE_PATH = "app.services.pivo_message_builder.random.choice"


class TestFallbackMentionsConstant(unittest.TestCase):
    """Строка запасного обращения должна быть одна на весь проект.

    Она объявлялась в двух модулях, и сборщик сообщения сравнивал значение со
    СВОЕЙ копией: расхождение констант тихо сломало бы подавление строки с
    упоминаниями — бот звал бы «Господа дегенераты» и отдельной строкой ещё раз
    их же.
    """

    def test_notification_line_is_suppressed_for_the_fallback(self) -> None:
        from app.domain.pivo import PIVO_FALLBACK_MENTIONS

        context = build_pivo_message_context(
            PIVO_FALLBACK_MENTIONS,
            planned_time=None,
            target=None,
            has_explicit_mentions=False,
        )

        self.assertEqual(context.notification_line, "")

    def test_notification_line_is_built_for_real_mentions(self) -> None:
        context = build_pivo_message_context(
            "@friend",
            planned_time=None,
            target=None,
            has_explicit_mentions=False,
        )

        self.assertIn("@friend", context.notification_line)

FORBIDDEN_TARGET_MODE_TERMS = (
    "СИГейм",
    "Codenames",
    "рисовалка",
    "Gartic",
    "выбрать игру",
    "играем",
    "игры на выбор",
    "ведущий",
    "с ведущим",
    "правила",
)


def _all_default_templates() -> Iterable[str]:
    yield from PIVO_DEFAULT_TARGET_INTROS
    yield from PIVO_DEFAULT_TOP_PARTS
    yield from PIVO_DEFAULT_BODY_PARTS
    yield from PIVO_DEFAULT_BOTTOM_PARTS


def _all_target_templates() -> Iterable[str]:
    yield from PIVO_TARGET_INTROS
    yield from PIVO_TARGET_TOP_PARTS
    yield from PIVO_TARGET_BODY_PARTS
    yield from PIVO_TARGET_BOTTOM_PARTS


class TestPivoMessageBuilder(unittest.TestCase):
    def test_default_mode_uses_default_pools(self) -> None:
        choices = iter(
            [
                PIVO_DEFAULT_TARGET_INTROS[0],
                "Пинги для тех, кто сам подписался на этот цирк: {mentions}.",
                PIVO_DEFAULT_TOP_PARTS[0],
                PIVO_DEFAULT_BODY_PARTS[0],
                PIVO_DEFAULT_BOTTOM_PARTS[0],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text = build_pivo_message("@friend")

        # Subscribers (no explicit mentions) are pinged by the notification line
        # only; the top line keeps the template's own vocative and stays clean.
        self.assertIn("Заслуженные дегенераты диванного фронта, общий сбор", text)
        self.assertIn("долго выбирать игру", text)
        self.assertIn("Пинги", text)
        self.assertIn("@friend", text)

    def test_explicit_target_mode_uses_target_pools(self) -> None:
        choices = iter(
            [
                PIVO_TARGET_INTROS[0],
                PIVO_TARGET_TOP_PARTS[0],
                PIVO_TARGET_BODY_PARTS[0],
                PIVO_TARGET_BOTTOM_PARTS[0],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text = build_pivo_message("Господа дегенераты", target="watch movie")

        self.assertIn("Болотные интеллектуалы", text)
        self.assertIn("watch movie", text)
        self.assertIn("спорить друг с другом", text)
        self.assertNotIn("Пинги", text)

    def test_only_time_is_embedded_naturally(self) -> None:
        choices = iter(
            [
                PIVO_DEFAULT_TARGET_INTROS[0],
                PIVO_DEFAULT_TOP_PARTS[0],
                PIVO_DEFAULT_BODY_PARTS[1],
                PIVO_DEFAULT_BOTTOM_PARTS[0],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text = build_pivo_message("Господа дегенераты", planned_time="20:00")

        self.assertIn("общий сбор в 20:00 в Discord", text)
        self.assertIn("выбрать игру 40 минут", text)

    def test_time_and_target_are_both_embedded(self) -> None:
        choices = iter(
            [
                PIVO_TARGET_INTROS[0],
                PIVO_TARGET_TOP_PARTS[2],
                PIVO_TARGET_BODY_PARTS[1],
                PIVO_TARGET_BOTTOM_PARTS[2],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text = build_pivo_message(
                "Господа дегенераты",
                planned_time="tomorrow evening",
                target="watch movie",
        )

        self.assertIn("собираемся завтра вечером в Discord", text)
        self.assertIn("по плану у нас watch movie", text)
        self.assertIn("Пиво приветствуется", text)

    def test_standalone_tomorrow_time_is_embedded_naturally(self) -> None:
        choices = iter(
            [
                PIVO_TARGET_INTROS[0],
                PIVO_TARGET_TOP_PARTS[2],
                PIVO_TARGET_BODY_PARTS[1],
                PIVO_TARGET_BOTTOM_PARTS[2],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text = build_pivo_message(
                "Господа дегенераты",
                planned_time="завтра",
                target="сосать бибу",
            )

        self.assertIn("собираемся завтра в Discord", text)
        self.assertIn("по плану у нас сосать бибу", text)
        self.assertNotIn("сегодня у нас завтра", text)

    def test_explicit_mentions_are_rendered_inline_without_subscriber_ping_line(self) -> None:
        choices = iter(
            [
                PIVO_DEFAULT_TARGET_INTROS[0],
                PIVO_DEFAULT_TOP_PARTS[2],
                PIVO_DEFAULT_BODY_PARTS[1],
                PIVO_DEFAULT_BOTTOM_PARTS[1],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text = build_pivo_message(
                "@one @two",
                planned_time="20:00",
                has_explicit_mentions=True,
            )

        self.assertIn("@one @two", text)
        self.assertNotIn("Пинги", text)
        self.assertIn("в 20:00", text)

    def test_context_values_are_escaped(self) -> None:
        choices = iter(
            [
                PIVO_TARGET_INTROS[0],
                PIVO_TARGET_TOP_PARTS[1],
                PIVO_TARGET_BODY_PARTS[2],
                PIVO_TARGET_BOTTOM_PARTS[0],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text = build_pivo_message(
                "Господа дегенераты",
                planned_time="<20:00>",
                target="movie & chat",
            )

        self.assertIn("в &lt;20:00&gt;", text)
        self.assertIn("movie &amp; chat", text)

    def test_all_template_pools_render_every_argument_combination(self) -> None:
        combinations = (
            {},
            {"planned_time": "20:00"},
            {"target": "board games"},
            {"planned_time": "tomorrow evening", "target": "board games"},
        )
        raw_placeholders = (
            "{mentions",
            "{time_",
            "{target_",
            "mentions_inline",
            "time_phrase",
            "target_phrase",
            "target_bullet",
        )

        for kwargs in combinations:
            for _ in range(20):
                with self.subTest(kwargs=kwargs):
                    text = build_pivo_message("@friend", **kwargs)

                    self.assertIn("Discord", text)
                    for placeholder in raw_placeholders:
                        self.assertNotIn(placeholder, text)

    def test_explicit_target_template_pools_have_no_default_game_terms(self) -> None:
        for template in _all_target_templates():
            for term in FORBIDDEN_TARGET_MODE_TERMS:
                with self.subTest(term=term, template=template):
                    self.assertNotIn(term, template)

    def test_default_mode_may_include_game_specific_terms(self) -> None:
        default_text = "\n".join(_all_default_templates())

        self.assertIn("Codenames", default_text)
        self.assertIn("ведущ", default_text)
        self.assertIn("правила", default_text)

    def test_explicit_target_output_does_not_add_forbidden_terms(self) -> None:
        target = "фильм"
        for _ in range(50):
            text = build_pivo_message("@friend", planned_time="20:00", target=target)
            text_without_target = text.replace(target, "")
            for term in FORBIDDEN_TARGET_MODE_TERMS:
                with self.subTest(term=term, text=text):
                    self.assertNotIn(term, text_without_target)

    def test_user_target_may_contain_default_game_term(self) -> None:
        choices = iter(
            [
                PIVO_TARGET_INTROS[0],
                PIVO_TARGET_TOP_PARTS[0],
                PIVO_TARGET_BODY_PARTS[2],
                PIVO_TARGET_BOTTOM_PARTS[0],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text = build_pivo_message(
                "Господа дегенераты",
                planned_time="20:00",
                target="Codenames",
            )

        self.assertIn("Codenames", text)
        text_without_target = text.replace("Codenames", "")
        self.assertNotIn("СИГейм", text_without_target)
        self.assertNotIn("рисовалка", text_without_target)

    def test_injected_rng_is_used_and_deterministic(self) -> None:
        import random

        # An injected, seeded RNG must drive selection (not the module random)
        # and produce reproducible output across calls (audit Q9).
        first = build_pivo_message("@friend", rng=random.Random(1234))
        second = build_pivo_message("@friend", rng=random.Random(1234))
        self.assertEqual(first, second)

    def test_injected_rng_bypasses_module_random(self) -> None:
        import random

        # If the injected RNG is honored, patching the module-level
        # random.choice must have no effect on the output.
        with patch(RANDOM_CHOICE_PATH, side_effect=AssertionError("module random used")):
            text = build_pivo_message("@friend", rng=random.Random(7))
        self.assertTrue(text)


class TestPivoTemplateGrammarInvariants(unittest.TestCase):
    """Slot-placement rules that keep the rendered Russian grammatical.

    The rules are documented in app/domain/pivo_templates.py; each one below
    corresponds to a class of broken output the old templates could produce.
    """

    def test_mentions_slot_carries_its_own_space(self) -> None:
        # The value is either "" or " @a @b", so a space written in front of the
        # slot would leave "Так, конченые , собираемся" when mentions are absent.
        for template in (*PIVO_DEFAULT_TOP_PARTS, *PIVO_TARGET_TOP_PARTS):
            with self.subTest(template=template):
                self.assertIn("{mentions_inline}", template)
                self.assertNotIn(" {mentions_inline}", template)
                self.assertFalse(template.startswith("{mentions_inline}"))

    def test_soft_time_slot_is_sentence_final(self) -> None:
        # Without a planned time the value expands to a comma-carrying clause
        # ("ближе к вечеру, как только ..."), which only works before the dot.
        for template in (*PIVO_DEFAULT_TOP_PARTS, *PIVO_TARGET_TOP_PARTS):
            if "{time_phrase_soft}" in template:
                with self.subTest(template=template):
                    self.assertTrue(template.endswith("{time_phrase_soft}."))

    def test_target_slot_is_never_case_governed(self) -> None:
        # The target is raw user text of unknown grammatical form ("фильмы" vs
        # "посмотреть фильм"), so it may only stand in a free position: a line
        # start, or right after a dash, a colon or "у нас".
        free_markers = ("— ", ": ", "у нас ")
        for template in (*PIVO_TARGET_INTROS, *PIVO_TARGET_BODY_PARTS):
            for match in re.finditer(re.escape("{target_phrase}"), template):
                prefix = template[: match.start()]
                with self.subTest(template=template):
                    self.assertTrue(
                        prefix == ""
                        or prefix.endswith("\n")
                        or prefix.endswith(free_markers),
                        f"target phrase is governed by {prefix[-24:]!r}",
                    )

    def test_target_bullet_stands_alone_in_its_list_item(self) -> None:
        for template in (*PIVO_DEFAULT_BODY_PARTS, *PIVO_TARGET_BODY_PARTS):
            for match in re.finditer(re.escape("{target_bullet}"), template):
                with self.subTest(template=template):
                    self.assertTrue(template[: match.start()].endswith("\n"))
                    self.assertTrue(template[match.end() :].startswith(";"))


class TestPivoRenderedTextIsWellFormed(unittest.TestCase):
    def test_no_punctuation_artifacts_in_any_combination(self) -> None:
        combinations = (
            {},
            {"planned_time": "20:00"},
            {"target": "посмотреть фильм"},
            {"target": "фильмы", "planned_time": "завтра вечером"},
        )
        artifacts = ("  ", " ,", " .", " :", " ;", ",,", "..", "!.", "?.")
        for kwargs in combinations:
            for has_explicit_mentions in (False, True):
                for _ in range(40):
                    text = build_pivo_message(
                        "@one @two",
                        has_explicit_mentions=has_explicit_mentions,
                        **kwargs,
                    )
                    for artifact in artifacts:
                        with self.subTest(kwargs=kwargs, artifact=artifact, text=text):
                            self.assertNotIn(artifact, text)

    def test_absent_mentions_never_duplicate_the_template_vocative(self) -> None:
        # Regression: the empty slot used to be filled with a noun phrase, which
        # doubled the vocative a template already had ("Так, конченые конченый
        # состав чата") and ignored the case the sentence governed ("Дорогой
        # конченый коллектив подозрительные личности" — nominative for genitive).
        for _ in range(200):
            text = build_pivo_message("Господа дегенераты")
            with self.subTest(text=text):
                self.assertIsNone(re.search(r"конченые\s+конченый", text))
                self.assertNotIn("подозрительные личности", text)
                self.assertNotIn("местные дегенераты", text)
                self.assertNotIn("морально уставшие участники", text)

    def test_target_keeps_its_own_trailing_punctuation_out_of_the_template(self) -> None:
        for _ in range(40):
            text = build_pivo_message(
                "@one", target="го в дотку!!!", has_explicit_mentions=True
            )
            with self.subTest(text=text):
                self.assertIn("го в дотку", text)
                self.assertNotIn("!", text)


class TestPivoAntiRepeat(unittest.TestCase):
    """S2: recently used pool indices are excluded from the next pick."""

    def _context(self) -> PivoMessageContext:
        return build_pivo_message_context(
            "@friend",
            planned_time=None,
            target=None,
            has_explicit_mentions=True,
        )

    def test_build_returns_picked_indices(self) -> None:
        import random

        result = PivoMessageGenerator().build(self._context(), rng=random.Random(1))
        self.assertEqual(
            set(result.picks), {"default_top", "default_body", "default_bottom"}
        )
        top_idx = result.picks["default_top"]
        # The literal prefix of the chosen top template (before any placeholder)
        # must appear in the rendered message.
        literal_prefix = PIVO_DEFAULT_TOP_PARTS[top_idx].split("{")[0].strip()
        self.assertIn(literal_prefix, result.text)

    def test_avoided_indices_are_never_picked(self) -> None:
        import random

        avoid_top = tuple(i for i in range(len(PIVO_DEFAULT_TOP_PARTS)) if i != 5)
        recent = {"default_top": avoid_top}
        # Only index 5 remains available for the top pool across many draws.
        for seed in range(30):
            result = PivoMessageGenerator().build(
                self._context(), rng=random.Random(seed), recent_indices=recent
            )
            self.assertEqual(result.picks["default_top"], 5)

    def test_all_excluded_falls_back_to_full_pool(self) -> None:
        import random

        all_top = tuple(range(len(PIVO_DEFAULT_TOP_PARTS)))
        result = PivoMessageGenerator().build(
            self._context(),
            rng=random.Random(3),
            recent_indices={"default_top": all_top},
        )
        # No candidate survives exclusion -> the full pool is used, no crash.
        self.assertIn(result.picks["default_top"], set(all_top))

    def test_empty_recent_matches_legacy_seeded_output(self) -> None:
        import random

        # Replicate build_pivo_message internals (one rng shared across context
        # building and generation); adding an empty recent_indices must not
        # change the byte-for-byte seeded output.
        rng = random.Random(99)
        context = build_pivo_message_context(
            "@friend",
            planned_time=None,
            target=None,
            has_explicit_mentions=False,
            rng=rng,
        )
        result = PivoMessageGenerator().build(context, rng=rng, recent_indices={})
        legacy = build_pivo_message("@friend", rng=random.Random(99))
        self.assertEqual(result.text, legacy)


class TestPivoTemporalModifiers(unittest.TestCase):
    """S4: time-aware closing lines replace the neutral bottom when applicable."""

    def _context(self) -> PivoMessageContext:
        return build_pivo_message_context(
            "@friend",
            planned_time=None,
            target=None,
            has_explicit_mentions=True,
        )

    def test_friday_line_used_when_roll_passes(self) -> None:
        import random

        from app.domain.pivo_templates import (
            PIVO_FRIDAY_BOTTOM_PARTS,
            PIVO_SUMMER_BOTTOM_PARTS,
        )

        friday = datetime(2026, 7, 3, 18, 0)  # a Friday evening in summer
        combined = set(PIVO_FRIDAY_BOTTOM_PARTS) | set(PIVO_SUMMER_BOTTOM_PARTS)
        seen = set()
        for seed in range(40):
            result = PivoMessageGenerator().build(
                self._context(),
                rng=random.Random(seed),
                now=friday,
                temporal_flavor_chance=1.0,
            )
            # Temporal bottom is not tracked by anti-repeat.
            self.assertNotIn("default_bottom", result.picks)
            for line in combined:
                if line.strip() in result.text:
                    seen.add(line)
        # Both buckets contribute to the same draw.
        self.assertTrue(seen & set(PIVO_FRIDAY_BOTTOM_PARTS))
        self.assertTrue(seen & set(PIVO_SUMMER_BOTTOM_PARTS))

    def test_late_night_and_monday_lines_available(self) -> None:
        import random

        from app.domain.pivo_templates import (
            PIVO_LATE_NIGHT_BOTTOM_PARTS,
            PIVO_MONDAY_BOTTOM_PARTS,
        )

        # Monday at 02:00 -> both late-night and Monday buckets apply.
        monday_night = datetime(2026, 7, 6, 2, 0)
        combined = set(PIVO_LATE_NIGHT_BOTTOM_PARTS) | set(PIVO_MONDAY_BOTTOM_PARTS)
        seen = set()
        for seed in range(40):
            result = PivoMessageGenerator().build(
                self._context(),
                rng=random.Random(seed),
                now=monday_night,
                temporal_flavor_chance=1.0,
            )
            for line in combined:
                if line.strip() in result.text:
                    seen.add(line)
        self.assertTrue(seen)

    def test_zero_chance_keeps_neutral_bottom(self) -> None:
        import random

        friday = datetime(2026, 7, 3, 18, 0)
        result = PivoMessageGenerator().build(
            self._context(),
            rng=random.Random(0),
            now=friday,
            temporal_flavor_chance=0.0,
        )
        self.assertIn("default_bottom", result.picks)

    def test_season_bucket_applies_on_plain_weekday(self) -> None:
        import random

        from app.domain.pivo_templates import PIVO_SUMMER_BOTTOM_PARTS

        # Wednesday afternoon: no late-night / Friday / Monday bucket, but the
        # seasonal pool always applies, so the flavor roll still has candidates.
        wednesday = datetime(2026, 7, 1, 15, 0)
        result = PivoMessageGenerator().build(
            self._context(),
            rng=random.Random(0),
            now=wednesday,
            temporal_flavor_chance=1.0,
        )
        self.assertNotIn("default_bottom", result.picks)
        self.assertTrue(
            any(line.strip() in result.text for line in PIVO_SUMMER_BOTTOM_PARTS)
        )

    def test_each_month_maps_to_its_season_pool(self) -> None:
        from app.domain.pivo_templates import (
            PIVO_AUTUMN_BOTTOM_PARTS,
            PIVO_SPRING_BOTTOM_PARTS,
            PIVO_SUMMER_BOTTOM_PARTS,
            PIVO_WINTER_BOTTOM_PARTS,
        )
        from app.services.pivo_message_builder import _temporal_bottoms

        by_month = {
            1: PIVO_WINTER_BOTTOM_PARTS, 2: PIVO_WINTER_BOTTOM_PARTS,
            3: PIVO_SPRING_BOTTOM_PARTS, 4: PIVO_SPRING_BOTTOM_PARTS,
            5: PIVO_SPRING_BOTTOM_PARTS, 6: PIVO_SUMMER_BOTTOM_PARTS,
            7: PIVO_SUMMER_BOTTOM_PARTS, 8: PIVO_SUMMER_BOTTOM_PARTS,
            9: PIVO_AUTUMN_BOTTOM_PARTS, 10: PIVO_AUTUMN_BOTTOM_PARTS,
            11: PIVO_AUTUMN_BOTTOM_PARTS, 12: PIVO_WINTER_BOTTOM_PARTS,
        }
        for month, pool in by_month.items():
            # Tue 2026-09-01 etc.: pick a mid-month Tuesday noon to isolate the
            # season bucket from day/hour ones.
            bottoms = _temporal_bottoms(datetime(2026, month, 15, 12, 0))
            for line in pool:
                self.assertIn(line, bottoms, f"month {month}")


class TestPivoSubSlots(unittest.TestCase):
    """Recursive sub-pool slots: {chaos_bullet} -> "спор {dispute_topic}" -> text."""

    def test_sub_slots_fully_resolve_in_built_message(self) -> None:
        import random

        from app.domain.pivo_templates import PIVO_SUB_POOLS

        context = build_pivo_message_context(
            "@friend",
            planned_time=None,
            target=None,
            has_explicit_mentions=True,
        )
        for seed in range(60):
            result = PivoMessageGenerator().build(context, rng=random.Random(seed))
            for name in PIVO_SUB_POOLS:
                self.assertNotIn("{" + name + "}", result.text, f"seed {seed}")
            self.assertNotIn("{chaos_bullet}", result.text)

    def test_expander_resolves_nested_slots(self) -> None:
        import random

        from app.services.pivo_message_builder import _expand_sub_slots

        expanded = _expand_sub_slots("итог: {chaos_bullet};", random.Random(1))
        self.assertNotIn("{chaos_bullet}", expanded)
        self.assertNotIn("{dispute_topic}", expanded)
        self.assertTrue(expanded.startswith("итог: "))

    def test_expander_leaves_context_slots_alone(self) -> None:
        import random

        from app.services.pivo_message_builder import _expand_sub_slots

        text = "{target_bullet}; {time_phrase_soft}"
        self.assertEqual(_expand_sub_slots(text, random.Random(1)), text)

    def test_reference_cycle_is_dropped_not_looped(self) -> None:
        import random
        from unittest.mock import patch as mock_patch

        from app.domain import pivo_templates
        from app.services.pivo_message_builder import _expand_sub_slots

        with mock_patch.dict(
            pivo_templates.PIVO_SUB_POOLS, {"loop_slot": ("{loop_slot}",)}
        ):
            expanded = _expand_sub_slots("до {loop_slot} после", random.Random(1))
        self.assertEqual(expanded, "до  после")

    def test_user_braces_stay_literal(self) -> None:
        import random

        # A user typing a sub-pool slot name must not trigger expansion: user
        # values are substituted after the expander has already run.
        text = build_pivo_message(
            "@friend",
            target="выпить {dispute_topic} пива",
            has_explicit_mentions=True,
            rng=random.Random(3),
        )
        self.assertIn("{dispute_topic}", text)


if __name__ == "__main__":
    unittest.main()


class TestEveryParsedTimeFormIsLocalized(unittest.TestCase):
    """Каждая форма, которую принимает парсер, имеет локализацию.

    «/pivo завтра 19:00» давало «собираемся завтра 19:00» без предлога, а
    «/pivo tomorrow 19:00» — правильное «завтра в 19:00»: в таблице префиксов
    были голые английские формы и не было голых русских. Русский ввод в
    русском чате обслуживался хуже английского. Найдено E-06 (раунд 1),
    подтверждено независимым замером W-1 (раунд 3).
    """

    def test_bare_day_plus_clock_gets_the_preposition(self) -> None:
        for value, expected in (
            ("завтра 19:00", "завтра в 19:00"),
            ("сегодня 19:00", "сегодня в 19:00"),
            ("tomorrow 19:00", "завтра в 19:00"),
            ("today 19:00", "сегодня в 19:00"),
        ):
            with self.subTest(value=value):
                self.assertEqual(_format_time_value(value), expected)

    def test_forms_that_already_worked_are_untouched(self) -> None:
        """Длинные формы проверяются раньше — предлог не дублируется."""
        for value in ("сегодня в 19:00", "завтра в 21:00", "завтра вечером", "сегодня"):
            with self.subTest(value=value):
                self.assertEqual(_format_time_value(value), value)
