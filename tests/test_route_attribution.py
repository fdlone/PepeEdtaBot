"""Per-route attribution of the candidate pool (M3R-103).

Before this, a candidate's origin was invisible: the trace showed seeded
candidates only in the selection block, without provenance, so telling a seeded
candidate from an organic one through ``GEN_TRACE_LOG`` was impossible and the
external review had to wrap the method to find out. A per-route breakdown is
what makes "quality went up by X%" attributable to a mechanism instead of to
the ensemble as a whole.
"""

from __future__ import annotations

import unittest

from app.core.generation_telemetry import CandidateRoute, GenerationTelemetry


class TestRouteEnumeration(unittest.TestCase):
    def test_only_live_producers_are_members(self) -> None:
        """Маршруты без производителя не заводятся — вечный нуль читается
        как «механизм работает и не выигрывает»."""
        self.assertEqual(
            {str(route) for route in CandidateRoute},
            {"vanilla", "seeded", "mutated", "extension", "hot", "assoc"},
        )

    def test_breakdown_covers_every_member(self) -> None:
        """Агрегация идёт по перечислению, поэтому новый маршрут появляется
        в разбивке без правки агрегатора."""
        breakdown = GenerationTelemetry().route_breakdown()
        self.assertEqual(set(breakdown), {str(route) for route in CandidateRoute})


class TestTwoDenominatorsPerRoute(unittest.TestCase):
    """«Не появился в пуле» и «появился и проиграл» — разные находки."""

    def test_share_and_win_rate_are_separate(self) -> None:
        tel = GenerationTelemetry()
        # Три генерации: seeded присутствовал в двух, выиграл в одной.
        tel.note_routes(
            attempted={CandidateRoute.VANILLA, CandidateRoute.SEEDED},
            present={CandidateRoute.VANILLA, CandidateRoute.SEEDED},
            winner=CandidateRoute.SEEDED,
        )
        tel.note_routes(
            attempted={CandidateRoute.VANILLA, CandidateRoute.SEEDED},
            present={CandidateRoute.VANILLA, CandidateRoute.SEEDED},
            winner=CandidateRoute.VANILLA,
        )
        tel.note_routes(
            attempted={CandidateRoute.VANILLA, CandidateRoute.SEEDED},
            present={CandidateRoute.VANILLA},
            winner=CandidateRoute.VANILLA,
        )

        seeded = tel.route_breakdown()["seeded"]
        self.assertEqual(seeded["attempts"], 3)
        self.assertEqual(seeded["present"], 2)
        self.assertAlmostEqual(seeded["present_rate"], 2 / 3)
        self.assertAlmostEqual(seeded["win_rate_given_present"], 1 / 2)

    def test_attempted_but_never_producing_differs_from_disabled(self) -> None:
        tel = GenerationTelemetry()
        tel.note_routes(
            attempted={CandidateRoute.VANILLA, CandidateRoute.SEEDED},
            present={CandidateRoute.VANILLA},
            winner=CandidateRoute.VANILLA,
        )
        breakdown = tel.route_breakdown()
        # Включён и пуст: попытки есть, присутствия нет.
        self.assertEqual(breakdown["seeded"]["attempts"], 1)
        self.assertEqual(breakdown["seeded"]["present"], 0)
        self.assertEqual(breakdown["seeded"]["present_rate"], 0.0)
        # Выключен: попыток нет вовсе — и это другое число, а не тот же ноль.
        self.assertEqual(breakdown["mutated"]["attempts"], 0)
        self.assertIsNone(breakdown["mutated"]["present_rate"])

    def test_no_selection_leaves_win_counters_untouched(self) -> None:
        tel = GenerationTelemetry()
        tel.note_routes(
            attempted={CandidateRoute.VANILLA},
            present=set(),
            winner=None,
        )
        self.assertEqual(tel.route_breakdown()["vanilla"]["won"], 0)


class TestRejectionReasonsPerRoute(unittest.TestCase):
    def test_reasons_are_counted_by_route(self) -> None:
        tel = GenerationTelemetry()
        tel.note_route_rejected(CandidateRoute.SEEDED, "low_diversity")
        tel.note_route_rejected(CandidateRoute.SEEDED, "low_diversity")
        tel.note_route_rejected(CandidateRoute.SEEDED, "context_heavy")
        tel.note_route_rejected(CandidateRoute.VANILLA, "context_heavy")

        self.assertEqual(
            tel.route_rejection_reasons(),
            {
                "seeded": {"low_diversity": 2, "context_heavy": 1},
                "vanilla": {"context_heavy": 1},
            },
        )
        self.assertEqual(tel.route_breakdown()["seeded"]["rejected"], 3)


class TestRoutePrivacy(unittest.TestCase):
    """Метки маршрутов и счётчики — только перечисление и числа."""

    def test_breakdown_carries_no_text_or_identifiers(self) -> None:
        tel = GenerationTelemetry()
        tel.note_routes(
            attempted={CandidateRoute.VANILLA},
            present={CandidateRoute.VANILLA},
            winner=CandidateRoute.VANILLA,
        )
        tel.note_route_rejected(CandidateRoute.VANILLA, "context_heavy")

        allowed_keys = {str(route) for route in CandidateRoute}
        breakdown = tel.route_breakdown()
        self.assertEqual(set(breakdown), allowed_keys)
        for metrics in breakdown.values():
            for value in metrics.values():
                self.assertIsInstance(value, (int, float, type(None)))

        for route, reasons in tel.route_rejection_reasons().items():
            self.assertIn(route, allowed_keys)
            for reason, count in reasons.items():
                # Причины — фиксированные метки трассы, а не тексты кандидатов.
                self.assertRegex(reason, r"^[a-z0-9_ :.,'-]+$")
                self.assertIsInstance(count, int)


if __name__ == "__main__":
    unittest.main()


class TestReplyTempoObservability(unittest.TestCase):
    """E2-1/E2-2: темп ответов наблюдаем, а не выводится из формулы.

    Оба дефекта — недостижимая фаза отхода берста и отсутствие пер-чатового
    предела на обращения — зависят от реального темпа чата, который не
    измерялся ни разу. Пока нет счётчика со знаменателем, «канал не нагружен»
    неотличимо от «канал не измеряется» (§5 CLAUDE.md).
    """

    def test_tempo_is_a_distribution_not_a_mean(self) -> None:
        tel = GenerationTelemetry()
        # Порог оживлённости 12: границы диапазонов берутся от него.
        for rate in (0.5, 1.0, 4.0, 8.0, 20.0):
            tel.note_chat_tempo(rate, lively_at=12.0)

        values = tel.snapshot()
        self.assertEqual(values["tempo_observations"], 5)
        self.assertAlmostEqual(values["tempo_share_штиль"], 2 / 5)
        self.assertAlmostEqual(values["tempo_share_кипит"], 1 / 5)

    def test_tempo_absent_reads_as_not_observed(self) -> None:
        values = GenerationTelemetry().snapshot()
        self.assertIsNone(values["tempo_observations"])
        self.assertIsNone(values["tempo_share_кипит"])

    def test_mention_denominator_grows_without_an_answer(self) -> None:
        """Обращение без ответа — тоже нагрузка на путь без предела.

        Знаменатель и числитель — две разные точки: упоминание видно в
        `observe` до всех гейтов, ответ решается в `respond`. Первая редакция
        считала их одним вызовом с булевым флагом, и доля выходила
        тождественно равной единице (проводка проверяется в
        `tests/test_reply_pipeline.py::TestTempoCountersWiring`).
        """
        tel = GenerationTelemetry()
        tel.note_mention_seen()
        tel.note_mention_answered(at=100.0)
        tel.note_mention_seen()  # это обращение ответа не получило

        values = tel.snapshot()
        self.assertEqual(values["mentions_observed"], 2)
        self.assertAlmostEqual(values["mention_answer_share"], 0.5)

    def test_mention_peak_is_measured_over_a_sliding_hour(self) -> None:
        """Пик считается скользящим окном, а не фиксированными корзинами.

        Границы корзин `now // 3600` привязаны к произвольной эпохе, и всплеск,
        легший на стык, делился пополам: десять ответов на 59-й минуте и девять
        на 61-й давали «пик 10» при худшем реальном окне в 19. Занижение до
        двух раз и всегда в сторону «предел не нужен» — а именно этим числом
        решается O15. Тем же скользящим окном считает `within_hourly_cap`.
        """
        tel = GenerationTelemetry()
        for index in range(10):
            tel.note_mention_seen()
            tel.note_mention_answered(at=3540.0 + index)
        for index in range(9):
            tel.note_mention_seen()
            tel.note_mention_answered(at=3660.0 + index)

        self.assertEqual(
            tel.snapshot()["mention_answers_peak_hour"],
            19,
            "всплеск на стыке корзин разделён надвое",
        )

    def test_answers_further_than_an_hour_apart_do_not_stack(self) -> None:
        tel = GenerationTelemetry()
        tel.note_mention_seen()
        tel.note_mention_answered(at=0.0)
        tel.note_mention_seen()
        tel.note_mention_answered(at=7200.0)

        self.assertEqual(tel.snapshot()["mention_answers_peak_hour"], 1)

    def test_burst_phases_are_counted_separately(self) -> None:
        """Ноль в фазе отхода при ненулевом усилении — доказательство E2-1."""
        tel = GenerationTelemetry()
        for _ in range(4):
            tel.note_burst_phase(suppressing=False)

        values = tel.snapshot()
        self.assertEqual(values["burst_phase_replies"], 4)
        self.assertEqual(values["burst_suppress_share"], 0.0)

    def test_burst_share_is_none_without_replies(self) -> None:
        values = GenerationTelemetry().snapshot()
        self.assertIsNone(values["burst_phase_replies"])
        self.assertIsNone(values["burst_suppress_share"])


class TestAnchorSpliceObservability(unittest.TestCase):
    """W2-2: отложенный якорь, потерянный по дороге, перестал быть невидимым.

    Прогулка упирается в символьный предел раньше, чем в разыгранную позицию
    вклейки, и такой ответ помечается как `global` с обнулёнными счётчиками
    совпадений — в трассе он неотличим от ответа, у которого якоря не было
    вовсе. При этом канал отработал и всю прогулку держал джампы выключенными
    (ветка джампа стоит под `not anchor_pending`).
    """

    def test_deferred_and_spliced_are_separate(self) -> None:
        tel = GenerationTelemetry()
        tel.note_anchor_splice(spliced=True)
        tel.note_anchor_splice(spliced=False)
        tel.note_anchor_splice(spliced=False)

        values = tel.snapshot()
        self.assertEqual(values["anchor_splice_deferred"], 3)
        self.assertAlmostEqual(values["anchor_splice_share"], 1 / 3)

    def test_absent_reads_as_not_observed(self) -> None:
        values = GenerationTelemetry().snapshot()
        self.assertIsNone(values["anchor_splice_deferred"])
        self.assertIsNone(values["anchor_splice_share"])
