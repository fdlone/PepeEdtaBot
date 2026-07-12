from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.eval_generation import evaluate_generation

BASELINE_PATH = Path(__file__).parents[1] / "tools" / "generation_baseline.json"
CASE_BASELINE_PATH = (
    Path(__file__).parents[1] / "tools" / "generation_baseline_case_preserved.json"
)


class TestGenerationEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_committed_baseline_matches_key_metrics(self) -> None:
        actual = await evaluate_generation(seed=20260620, generations=100)
        expected = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        latency_keys = {
            "avg_generation_latency_ms",
            "median_generation_latency_ms",
        }

        expected_metrics = {
            key: value for key, value in expected.items() if key not in latency_keys
        }
        actual_metrics = {key: actual[key] for key in expected_metrics}

        self.assertEqual(actual_metrics, expected_metrics)
        self.assertEqual(actual["leading_punctuation_rate"], 0.0)

    async def test_case_preserved_baseline_matches_key_metrics(self) -> None:
        actual = await evaluate_generation(
            seed=20260620,
            generations=100,
            normalize_lower=False,
        )
        expected = json.loads(CASE_BASELINE_PATH.read_text(encoding="utf-8"))
        latency_keys = {
            "avg_generation_latency_ms",
            "median_generation_latency_ms",
        }

        expected_metrics = {
            key: value for key, value in expected.items() if key not in latency_keys
        }
        actual_metrics = {key: actual[key] for key in expected_metrics}

        self.assertEqual(actual_metrics, expected_metrics)
        self.assertEqual(actual["leading_punctuation_rate"], 0.0)

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

    async def test_casefold_profile_records_casefold_resolution(self) -> None:
        actual = await evaluate_generation(
            seed=20260620,
            generations=100,
            normalize_lower=False,
            fuzzy_context_casefold=True,
        )

        self.assertGreater(actual["context_casefold_match_rate"], 0.0)
        self.assertEqual(actual["leading_punctuation_rate"], 0.0)

    async def test_stem_profile_records_stem_resolution(self) -> None:
        actual = await evaluate_generation(
            seed=20260620,
            generations=100,
            normalize_lower=False,
            fuzzy_context_stem=True,
        )

        self.assertGreater(actual["context_stem_match_rate"], 0.0)
        self.assertEqual(actual["leading_punctuation_rate"], 0.0)
