from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.eval_generation import evaluate_generation

BASELINE_PATH = Path(__file__).parents[1] / "tools" / "generation_baseline.json"


class TestGenerationEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_committed_baseline_matches_key_metrics(self) -> None:
        actual = await evaluate_generation(seed=20260620, generations=100)
        expected = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        latency_keys = {
            "avg_generation_latency_ms",
            "median_generation_latency_ms",
        }

        actual_metrics = {
            key: value for key, value in actual.items() if key not in latency_keys
        }
        expected_metrics = {
            key: value for key, value in expected.items() if key not in latency_keys
        }

        self.assertEqual(actual_metrics, expected_metrics)

    async def test_same_seed_produces_same_quality_metrics(self) -> None:
        first = await evaluate_generation(seed=1234, generations=12)
        second = await evaluate_generation(seed=1234, generations=12)

        latency_keys = {
            "avg_generation_latency_ms",
            "median_generation_latency_ms",
        }
        first_metrics = {
            key: value for key, value in first.items() if key not in latency_keys
        }
        second_metrics = {
            key: value for key, value in second.items() if key not in latency_keys
        }

        self.assertEqual(first_metrics, second_metrics)
