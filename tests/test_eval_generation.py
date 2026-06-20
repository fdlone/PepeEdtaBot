from __future__ import annotations

import unittest

from tools.eval_generation import evaluate_generation


class TestGenerationEvaluation(unittest.IsolatedAsyncioTestCase):
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
