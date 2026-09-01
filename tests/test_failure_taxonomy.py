"""Словарь отказов ответа (M3R-021)."""

from __future__ import annotations

import unittest

from app.core import markov, response_generator
from app.core.failure_taxonomy import (
    AUTOMATIC_CLASSES,
    REASON_TO_CLASS,
    UNMAPPED,
    FailureClass,
    classify_reason,
)
from app.core.generation_telemetry import GenerationTelemetry

# Причины, у которых класса нет намеренно (design D4): отказ произошёл до
# появления текста, это свойство данных чата, а не ответа. Набор перечислен
# здесь, а не выведен, чтобы новая причина ядра валила тест ниже, а не
# растворялась в «не размечено».
DELIBERATELY_UNMAPPED = {
    markov.REJECTION_NO_STARTS,
    markov.REJECTION_NO_START_TRANSITION,
}


class FailureTaxonomyTest(unittest.TestCase):
    def test_mapped_reasons_resolve_to_their_class(self) -> None:
        self.assertEqual(
            classify_reason(markov.REJECTION_SHORT_CONTEXT_COPY),
            FailureClass.CONTEXT_COPY,
        )
        self.assertEqual(
            classify_reason(markov.REJECTION_CONTEXT_HEAVY),
            FailureClass.CONTEXT_COPY,
        )
        self.assertEqual(
            classify_reason(markov.REJECTION_LOW_DIVERSITY),
            FailureClass.STRUCTURAL_REPEAT,
        )
        self.assertEqual(
            classify_reason(markov.REJECTION_RESULT_TOO_SHORT),
            FailureClass.MALFORMED,
        )
        self.assertEqual(
            classify_reason(response_generator.STALE_REPLY_REASON),
            FailureClass.STALE,
        )
        self.assertEqual(
            classify_reason(response_generator.VERBATIM_COPY_REASON),
            FailureClass.STALE,
        )

    def test_unknown_reason_has_no_class(self) -> None:
        self.assertIsNone(classify_reason("причина, которой нет"))

    def test_no_material_reasons_are_deliberately_unmapped(self) -> None:
        """`no_starts` / `no_start_transition` — отказ до текста, не отказ ответа."""
        for reason in DELIBERATELY_UNMAPPED:
            with self.subTest(reason=reason):
                self.assertIsNone(classify_reason(reason))

    def test_every_core_rejection_reason_is_accounted_for(self) -> None:
        """Новая причина ядра обязана быть либо размечена, либо явно исключена.

        Пин против тихой потери разметки: без него добавленная причина просто
        начала бы копиться в «не размечено», и это выглядело бы как рабочее
        состояние.
        """
        core_reasons = {
            value
            for name, value in vars(markov).items()
            if name.startswith("REJECTION_") and isinstance(value, str)
        }
        unaccounted = core_reasons - set(REASON_TO_CLASS) - DELIBERATELY_UNMAPPED
        self.assertEqual(unaccounted, set())

    def test_automatic_classes_are_a_strict_subset(self) -> None:
        """Покрытие автоматикой частичное — и это заявленное свойство."""
        self.assertTrue(AUTOMATIC_CLASSES < set(FailureClass))
        self.assertNotIn(FailureClass.IRRELEVANT, AUTOMATIC_CLASSES)


class RejectionsByClassTest(unittest.TestCase):
    def test_classes_plus_unmapped_equal_reason_totals(self) -> None:
        telemetry = GenerationTelemetry()
        telemetry.note_route_rejected("vanilla", markov.REJECTION_LOW_DIVERSITY)
        telemetry.note_route_rejected("vanilla", markov.REJECTION_CONTEXT_HEAVY)
        telemetry.note_route_rejected("seeded", markov.REJECTION_SHORT_CONTEXT_COPY)
        telemetry.note_route_rejected("seeded", markov.REJECTION_NO_STARTS)
        telemetry.note_route_rejected("seeded", markov.REJECTION_NO_STARTS)

        by_class = telemetry.rejections_by_class()

        self.assertEqual(by_class[FailureClass.CONTEXT_COPY.value], 2)
        self.assertEqual(by_class[FailureClass.STRUCTURAL_REPEAT.value], 1)
        self.assertEqual(by_class[UNMAPPED], 2)
        by_reason = telemetry.route_rejection_reasons()
        self.assertEqual(
            sum(by_class.values()),
            sum(count for reasons in by_reason.values() for count in reasons.values()),
        )

    def test_no_rejections_gives_empty_breakdown(self) -> None:
        self.assertEqual(GenerationTelemetry().rejections_by_class(), {})


if __name__ == "__main__":
    unittest.main()
