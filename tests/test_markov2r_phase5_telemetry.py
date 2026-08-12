"""Seeded-generation telemetry: two separate denominators (M2R-410, TZ §9.6)."""

from __future__ import annotations

import unittest

from app.core.generation_telemetry import GenerationTelemetry


class TestSeededTelemetry(unittest.TestCase):
    def test_present_and_win_denominators_are_separate(self) -> None:
        t = GenerationTelemetry()
        # 4 generations: 2 with a seeded candidate present, of which 1 won.
        t.note_seeded(present=False, won=False)
        t.note_seeded(present=True, won=False)
        t.note_seeded(present=True, won=True)
        t.note_seeded(present=False, won=False)
        snap = t.snapshot()
        self.assertEqual(snap["seeded_generations"], 4)
        self.assertAlmostEqual(snap["seeded_present_rate"], 2 / 4)
        # win rate is over PRESENT, not over all generations.
        self.assertAlmostEqual(snap["seeded_win_rate_given_present"], 1 / 2)

    def test_configured_but_never_anchoring_is_distinguishable_from_off(self) -> None:
        """Ratio non-zero but no anchor ever: present-count 0, not 'no data'."""
        configured = GenerationTelemetry()
        for _ in range(5):
            configured.note_seeded(present=False, won=False)
        snap = configured.snapshot()
        self.assertEqual(snap["seeded_generations"], 5)
        self.assertEqual(snap["seeded_present_rate"], 0.0)  # measured zero
        self.assertIsNone(snap["seeded_win_rate_given_present"])  # no denominator

        off = GenerationTelemetry()  # ratio 0: note_seeded never called
        off_snap = off.snapshot()
        self.assertEqual(off_snap["seeded_generations"], 0)
        self.assertIsNone(off_snap["seeded_present_rate"])  # None, not 0.0


if __name__ == "__main__":
    unittest.main()
