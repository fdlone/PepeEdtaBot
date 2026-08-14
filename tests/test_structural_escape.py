"""Structural Escape Gate: сколько различных траекторий доживает до окна (M3R-011).

Главный гейт проекта измеряется здесь, поэтому тесты закрепляют не только
арифметику, но и три решения, от которых зависит его смысл: рёбра как единица
сравнения, нормировка на меньшее множество и независимость счёта от порядка
кандидатов.
"""

from __future__ import annotations

import unittest

from tools.eval.config import load_thresholds
from tools.eval.metrics import (
    GenRecord,
    distinct_trajectories,
    edge_overlap,
    metric_values,
    trajectory_edges,
)
from tools.eval.report import _structural_escape_rows, evaluate_gates
from tools.eval.run import ConfigRun


def _t(text: str) -> tuple[str, ...]:
    return tuple(text.split())


class TestEdgeOverlap(unittest.TestCase):
    def test_edges_are_adjacent_pairs(self) -> None:
        self.assertEqual(
            trajectory_edges(_t("а б в")), frozenset({("а", "б"), ("б", "в")})
        )
        self.assertEqual(trajectory_edges(_t("а")), frozenset())

    def test_same_words_different_order_are_different_walks(self) -> None:
        """Мешок слов их не различит — а это разные проходки по цепи."""
        a = trajectory_edges(_t("пиво сегодня будет"))
        b = trajectory_edges(_t("сегодня пиво будет"))
        self.assertEqual(edge_overlap(a, b), 0.0)

    def test_subpath_is_the_same_trajectory(self) -> None:
        """Нормировка на меньшее множество: обрыв той же проходки — не новая."""
        full = trajectory_edges(_t("а б в г д"))
        cut = trajectory_edges(_t("а б в"))
        self.assertEqual(edge_overlap(full, cut), 1.0)

    def test_edgeless_candidates(self) -> None:
        empty = trajectory_edges(_t("слово"))
        other = trajectory_edges(_t("другое"))
        self.assertEqual(edge_overlap(empty, other), 1.0)  # оба без рёбер
        self.assertEqual(edge_overlap(empty, trajectory_edges(_t("а б"))), 0.0)


class TestDistinctTrajectories(unittest.TestCase):
    def test_identical_candidates_are_one(self) -> None:
        pool = [_t("а б в г")] * 5
        self.assertEqual(distinct_trajectories(pool), 1)

    def test_disjoint_candidates_are_counted_separately(self) -> None:
        pool = [_t("а б в"), _t("к л м"), _t("х ц ч")]
        self.assertEqual(distinct_trajectories(pool), 3)

    def test_shared_tail_is_one_trajectory(self) -> None:
        # 2 общих ребра из 3 = 0.67 >= 0.5 — одна проходка с разным зачином.
        pool = [_t("а б в г"), _t("д б в г")]
        self.assertEqual(distinct_trajectories(pool), 1)

    def test_empty_pool(self) -> None:
        self.assertEqual(distinct_trajectories([]), 0)

    def test_count_does_not_depend_on_input_order(self) -> None:
        """Отчёт сверяется бит-в-бит: порядкозависимое число недопустимо."""
        pool = [_t("а б в г"), _t("д б в г"), _t("к л м н"), _t("п р с т")]
        first = distinct_trajectories(pool)
        for shift in range(1, len(pool)):
            rotated = pool[shift:] + pool[:shift]
            self.assertEqual(distinct_trajectories(rotated), first)
        self.assertEqual(distinct_trajectories(list(reversed(pool))), first)

    def test_threshold_changes_the_grouping(self) -> None:
        pool = [_t("а б в г"), _t("д б в г")]  # overlap 0.67
        self.assertEqual(distinct_trajectories(pool, similar_at=0.5), 1)
        self.assertEqual(distinct_trajectories(pool, similar_at=0.8), 2)


def _record(pool: int, window: int, *, success: bool = True) -> GenRecord:
    return GenRecord(
        category="generic",
        prompt_content=("пиво",),
        reply_text="ответ",
        reply_content=("ответ",),
        success=success,
        latency_ms=10.0,
        pool_size=5,
        rejected_count=0,
        pool_ecb=pool,
        window_escape=window,
    )


class TestStructuralMetricsAreAPair(unittest.TestCase):
    def test_both_metrics_are_published(self) -> None:
        values = metric_values([_record(4, 2), _record(5, 3)])
        self.assertEqual(values["structural_pool_ecb"], [4.0, 5.0])
        self.assertEqual(values["structural_window_escape"], [2.0, 3.0])

    def test_failed_generations_stay_in_the_denominator(self) -> None:
        """Генерация без кандидатов дала ноль траекторий — выкинуть её значило
        бы мерить разнообразие только там, где оно случилось."""
        values = metric_values([_record(4, 2), _record(0, 0, success=False)])
        self.assertEqual(values["structural_window_escape"], [2.0, 0.0])


class TestStructuralGate(unittest.TestCase):
    def _run(self, pool: float, window: float) -> ConfigRun:
        return ConfigRun(
            config_id="C0",
            records=[_record(int(pool), int(window)) for _ in range(40)],
        )

    def _row(self, pool: float, window: float) -> tuple[str, str, str]:
        rows = _structural_escape_rows({"C0": self._run(pool, window)}, load_thresholds())
        return rows[0]

    def test_pass_needs_both_numbers(self) -> None:
        gate, verdict, detail = self._row(pool=5, window=3)
        self.assertEqual(verdict, "pass")
        self.assertIn("window escape", detail)
        self.assertIn("pool ECB", detail)  # пара, никогда не одно число
        self.assertIn("window distribution", detail)

    def test_window_below_minimum_fails(self) -> None:
        verdict, detail = self._row(pool=5, window=1)[1:]
        self.assertEqual(verdict, "fail")
        self.assertIn("window escape below", detail)

    def test_pool_floor_breach_fails_even_with_a_wide_window(self) -> None:
        """Нельзя купить разнообразие окна, ужав пул."""
        verdict, detail = self._row(pool=3, window=3)[1:]
        self.assertEqual(verdict, "fail")
        self.assertIn("pool ECB below", detail)

    def test_without_records_it_is_insufficient(self) -> None:
        rows = _structural_escape_rows(
            {"C0": ConfigRun(config_id="C0", records=[])}, load_thresholds()
        )
        self.assertEqual(rows[0][1], "insufficient data")

    def test_thresholds_are_registered(self) -> None:
        block = load_thresholds()["structural_escape"]
        self.assertEqual(block["edge_overlap_similar"], 0.5)
        self.assertEqual(block["window_escape_min"], 2.0)
        self.assertEqual(block["pool_ecb_min"], 4.0)
        self.assertTrue(block["requires_both_modes"])

    def test_one_mode_run_cannot_pass(self) -> None:
        """Окно определено через скор, а его шкала между режимами разная."""
        rows = evaluate_gates(
            {"C0": self._run(pool=5, window=3)}, load_thresholds(), None, "ctx"
        )
        verdict, detail = next(
            (row[1], row[2]) for row in rows if row[0].startswith("structural_escape")
        )
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("requires both context modes", detail)


if __name__ == "__main__":
    unittest.main()
