"""Knob census tooling (M3R-151, change knob-census).

The arms come from the registry's own domain hints, the classes from the
pre-registered rule, and the report from numbers only.
"""
from __future__ import annotations

import unittest

from app.config.registry import RUNTIME_FIELDS, FieldSpec, _bool, _float_in_range, _int_in_range
from tools.eval.config import load_thresholds
from tools.eval.knob_census import (
    CLASSIFICATION_METRICS,
    GATED_BY,
    NOT_SWEPT,
    classify_extreme,
    classify_knob,
    extremes,
    is_core_knob,
    plan,
)


class TestExtremes(unittest.TestCase):
    def test_range_hint_gives_both_bounds(self) -> None:
        spec = FieldSpec("x", "X", "0.5", _float_in_range(0.0, 1.0))
        self.assertEqual(extremes(spec), [("min", 0.0), ("max", 1.0)])

    def test_extreme_equal_to_default_is_skipped(self) -> None:
        spec = FieldSpec("x", "X", "0", _float_in_range(0.0, 0.7))
        self.assertEqual(extremes(spec), [("max", 0.7)])
        spec = FieldSpec("y", "Y", "10", _int_in_range(0, 10))
        self.assertEqual(extremes(spec), [("min", 0)])

    def test_boolean_is_flipped(self) -> None:
        self.assertEqual(extremes(FieldSpec("b", "B", "true", _bool())), [("flip", False)])
        self.assertEqual(extremes(FieldSpec("b", "B", "false", _bool())), [("flip", True)])

    def test_every_registry_knob_parses_or_is_listed(self) -> None:
        # A knob whose hint the census cannot read must be in NOT_SWEPT or
        # produce arms — silently dropping one would make it look inert.
        arms, skipped, sites = plan()
        planned = {arm.knob for arm in arms}
        for spec in RUNTIME_FIELDS:
            if spec.name in planned or spec.name in skipped:
                continue
            self.assertFalse(
                is_core_knob(spec.name, sites),
                f"{spec.name} is read by the generation core but neither planned nor listed",
            )

    def test_runtime_state_reads_count_as_sites(self) -> None:
        # runtime_state.py declares every knob (not a read) but also consumes a
        # few itself: the mood config and the rare-event daily cap. The first
        # census printed those as "dead".
        _, _, sites = plan()
        for name in ("mood_ewma_alpha", "rare_event_daily_cap"):
            self.assertIn("app/config/runtime_state.py", sites[name], name)
        self.assertNotIn("app/config/runtime_state.py", sites["markov_order"])

    def test_gated_children_get_a_parent_on_arm(self) -> None:
        arms, _, _ = plan()
        for child in GATED_BY:
            ids = {arm.arm_id for arm in arms if arm.knob == child}
            self.assertTrue(any(i.endswith("__gated") for i in ids), child)
            self.assertTrue(any(not i.endswith("__gated") for i in ids), child)

    def test_not_swept_knobs_produce_no_arm(self) -> None:
        arms, skipped, _ = plan()
        for knob in NOT_SWEPT:
            self.assertNotIn(knob, {arm.knob for arm in arms})
            self.assertIn(knob, skipped)


class TestClassification(unittest.TestCase):
    thresholds = load_thresholds()["knob_census"]

    @staticmethod
    def _samples(value: float, n: int = 200, jitter: float = 0.0) -> dict[str, list[float]]:
        return {
            metric: [value + ((i % 3) - 1) * jitter for i in range(n)]
            for metric in CLASSIFICATION_METRICS
        }

    def test_identical_samples_are_inert(self) -> None:
        cls, _ = classify_extreme(self._samples(0.2), self._samples(0.2), self.thresholds)
        self.assertEqual(cls, "inert")

    def test_large_significant_shift_is_strong(self) -> None:
        cls, deltas = classify_extreme(self._samples(0.2), self._samples(0.3), self.thresholds)
        self.assertEqual(cls, "strong")
        self.assertTrue(deltas["exact_context_copy_rate"][3])

    def test_small_but_resolved_shift_is_weak(self) -> None:
        base = self._samples(0.2, jitter=0.02)
        arm = self._samples(0.215, jitter=0.02)
        cls, _ = classify_extreme(base, arm, self.thresholds)
        self.assertEqual(cls, "weak")

    def test_missing_metric_is_not_inert(self) -> None:
        base = self._samples(0.2)
        arm = self._samples(0.2)
        arm["historical_meme_rate"] = None
        cls, _ = classify_extreme(base, arm, self.thresholds)
        self.assertEqual(cls, "weak")

    def test_knob_class_is_gated_when_only_the_parent_arm_moves(self) -> None:
        self.assertEqual(classify_knob([(False, "inert"), (True, "strong")]), "gated")
        self.assertEqual(classify_knob([(False, "inert"), (True, "inert")]), "inert")
        self.assertEqual(classify_knob([(False, "weak"), (True, "strong")]), "weak")
        self.assertEqual(classify_knob([(False, "strong"), (False, "inert")]), "strong")

    def test_rule_is_registered(self) -> None:
        for metric in CLASSIFICATION_METRICS:
            self.assertIn(metric, self.thresholds["tolerance"])
            self.assertIn(metric, self.thresholds["strong"])


if __name__ == "__main__":
    unittest.main()
