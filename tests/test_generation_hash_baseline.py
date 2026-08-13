"""The recorded generation baseline and the guard that checks it.

Why this exists: the baseline used to be a constant repeated in three
documents and several task lists, and nothing compared anything to it. It
drifted on the prod copy (5a72e2d4 -> 13e496c0) and the drift was found by
accident, weeks later, while reading unrelated numbers. The record is now a
file and the guard fails on a mismatch — these tests pin that behaviour, not
the hash itself (computing it takes a thousand generations).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.generation_hash import (  # noqa: E402
    BASELINE_PATH,
    SYNTHETIC_LABEL,
    check_against_baseline,
)

RECORD = {
    "snapshot": "synthetic",
    "hash": "a" * 64,
    "seeds": [101, 102, 103, 104],
    "per_seed": 250,
    "revision": "deadbee",
    "reason": "recorded for the test",
}
SEEDS = (101, 102, 103, 104)


class TestBaselineRecordFile(unittest.TestCase):
    """The shipped record must carry what makes a re-anchor auditable."""

    def test_record_is_complete(self) -> None:
        record = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        for key in ("snapshot", "hash", "seeds", "per_seed", "revision", "reason"):
            self.assertIn(key, record)
        self.assertEqual(record["snapshot"], SYNTHETIC_LABEL)
        self.assertEqual(len(record["hash"]), 64)
        # A value with no stated reason is indistinguishable from one tuned to
        # make the guard pass, which is the failure mode this file exists for.
        self.assertGreater(len(record["reason"]), 40)


class TestCheckAgainstBaseline(unittest.TestCase):
    def _with_record(self, record: dict[str, object] | None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "baseline.json"
        if record is not None:
            path.write_text(json.dumps(record), encoding="utf-8")
        return mock.patch("tools.generation_hash.BASELINE_PATH", path)

    def test_match_succeeds(self) -> None:
        with self._with_record(RECORD):
            code = check_against_baseline("a" * 64, "synthetic", SEEDS, 250)
        self.assertEqual(code, 0)

    def test_mismatch_fails_and_names_both_values(self) -> None:
        with self._with_record(RECORD), mock.patch("builtins.print") as printed:
            code = check_against_baseline("b" * 64, "synthetic", SEEDS, 250)
        self.assertEqual(code, 1)
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("a" * 64, output)
        self.assertIn("b" * 64, output)

    def test_unknown_snapshot_does_not_compare(self) -> None:
        # The prod copy has no record and must not be reported as a mismatch:
        # a false alarm there would train everyone to ignore the guard.
        with self._with_record(RECORD):
            code = check_against_baseline("b" * 64, "db_prod_copy/markov.db", SEEDS, 250)
        self.assertEqual(code, 0)

    def test_absent_record_does_not_compare(self) -> None:
        with self._with_record(None):
            code = check_against_baseline("b" * 64, "synthetic", SEEDS, 250)
        self.assertEqual(code, 0)

    def test_different_draw_budget_refuses_to_compare(self) -> None:
        # A hash over 10 generations is not a different value of the same
        # measurement — it is a different measurement.
        with self._with_record(RECORD), mock.patch("builtins.print") as printed:
            code = check_against_baseline("a" * 64, "synthetic", SEEDS, 10)
        self.assertEqual(code, 1)
        self.assertIn("NOT COMPARABLE", str(printed.call_args_list[0].args[0]))


if __name__ == "__main__":
    unittest.main()
