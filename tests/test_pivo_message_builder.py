from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.pivo_message_builder import build_pivo_message
from pivo_templates import PIVO_TEMPLATES


class TestPivoMessageBuilder(unittest.TestCase):
    def test_no_arguments_uses_template_unchanged(self) -> None:
        with patch("random.choice", return_value=PIVO_TEMPLATES[0]):
            text = build_pivo_message("@friend")

        self.assertIn("@friend", text)
        self.assertIn("Сегодня объявляется общий сбор в Discord.", text)
        self.assertIn("Возможные дисциплины: СИГейм, Codenames, рисовалка", text)

    def test_mentions_only_uses_template_unchanged(self) -> None:
        with patch("random.choice", return_value=PIVO_TEMPLATES[1]):
            text = build_pivo_message("@friend")

        self.assertIn("@friend", text)
        self.assertIn("потом всё равно играем в СИГейм", text)
        self.assertNotIn("Повестка вечера:", text)

    def test_time_is_embedded_into_existing_template(self) -> None:
        with patch("random.choice", return_value=PIVO_TEMPLATES[0]):
            text = build_pivo_message("@friend", planned_time="20:00")

        self.assertIn("Сегодня в 20:00 объявляется общий сбор в Discord.", text)
        self.assertNotIn("Когда:", text)

    def test_target_replaces_activity_context(self) -> None:
        with patch("random.choice", return_value=PIVO_TEMPLATES[0]):
            text = build_pivo_message("@friend", target="watch movie")

        self.assertIn("Повестка вечера: watch movie.", text)
        self.assertNotIn("СИГейм", text)
        self.assertNotIn("Codenames", text)
        self.assertNotIn("рисовалка", text)
        self.assertNotIn("Повод:", text)

    def test_target_and_time_are_both_embedded(self) -> None:
        with patch("random.choice", return_value=PIVO_TEMPLATES[0]):
            text = build_pivo_message(
                "@friend",
                planned_time="tomorrow evening",
                target="watch movie",
            )

        self.assertIn("Завтра вечером объявляется общий сбор в Discord.", text)
        self.assertIn("Повестка вечера: watch movie.", text)
        self.assertNotIn("Когда:", text)
        self.assertNotIn("Повод:", text)

    def test_context_values_are_escaped(self) -> None:
        with patch("random.choice", return_value=PIVO_TEMPLATES[0]):
            text = build_pivo_message(
                "@friend",
                planned_time="<20:00>",
                target="movie & chat",
            )

        self.assertIn("&lt;20:00&gt; объявляется общий сбор", text)
        self.assertIn("Повестка вечера: movie &amp; chat.", text)


if __name__ == "__main__":
    unittest.main()
