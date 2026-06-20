from __future__ import annotations

import unittest
from collections.abc import Iterable
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
from app.services.pivo_message_builder import build_pivo_message

RANDOM_CHOICE_PATH = "app.services.pivo_message_builder.random.choice"

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
                "местные дегенераты",
                "Пинги для тех, кто сам подписался на этот цирк: {mentions}.",
                PIVO_DEFAULT_TOP_PARTS[0],
                PIVO_DEFAULT_BODY_PARTS[0],
                PIVO_DEFAULT_BOTTOM_PARTS[0],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text = build_pivo_message("@friend")

        self.assertIn("местные дегенераты", text)
        self.assertIn("долго выбирать игру", text)
        self.assertIn("Пинги", text)
        self.assertIn("@friend", text)

    def test_explicit_target_mode_uses_target_pools(self) -> None:
        choices = iter(
            [
                PIVO_TARGET_INTROS[0],
                "местные дегенераты",
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
                "конченый состав чата",
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
                "подозрительные личности",
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
                "подозрительные личности",
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
                "местные дегенераты",
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
            "target_context",
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
                "местные дегенераты",
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


if __name__ == "__main__":
    unittest.main()
