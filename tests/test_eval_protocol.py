"""Tests for the Markov 2.0R eval protocol runner (tools/eval, doc 05).

Covers: metric known-answer cases (§3), stdlib bootstrap (§4), deterministic
prompt generation (§1.2), run reproducibility on the synthetic snapshot (§1.3),
and the first hypothesis property tests — the TZ §19 pattern later phases
extend.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from tools.eval.bootstrap import bootstrap_ci, delta_ci
from tools.eval.config import load_matrix, load_thresholds, resolve_overrides
from tools.eval.metrics import (
    GenRecord,
    build_snapshot_idf,
    context_affinity,
    distinct_n,
    has_token_cycle,
    is_repetition_flagged,
    latency_percentiles,
    longest_common_run,
    meme_regression,
    metric_values,
    repeated_ngram_share,
)
from tools.eval.prompts import generate_prompts
from tools.eval.report import build_report, metrics_summary
from tools.eval.run import run_matrix
from tools.eval.synthetic import SYNTHETIC_CHAT_ID, build_synthetic_snapshot


def _record(**kwargs: object) -> GenRecord:
    base: dict = {
        "category": "generic",
        "prompt_content": ("пиво", "сегодня"),
        "reply_text": "ну такое",
        "reply_content": ("такое",),
        "success": True,
        "latency_ms": 10.0,
        "pool_size": 5,
        "rejected_count": 5,
    }
    base.update(kwargs)
    return GenRecord(**base)


class TestMetricHelpers(unittest.TestCase):
    def test_longest_common_run(self) -> None:
        self.assertEqual(longest_common_run(("а", "б", "в", "г"), ("б", "в", "г", "д")), 3)
        self.assertEqual(longest_common_run(("а", "б"), ("в", "г")), 0)
        self.assertEqual(longest_common_run((), ("а",)), 0)

    def test_has_token_cycle(self) -> None:
        self.assertTrue(has_token_cycle(("х", "у", "х", "у")))  # period 2
        self.assertTrue(has_token_cycle(("а", "б", "в", "а", "б", "в")))  # period 3
        self.assertTrue(has_token_cycle(("да", "да", "да", "да")))  # degenerate period 2
        self.assertFalse(has_token_cycle(("а", "б", "в", "г", "д")))
        self.assertFalse(has_token_cycle(("а", "б", "а")))  # too short to close a cycle

    def test_repetition_flag(self) -> None:
        self.assertTrue(is_repetition_flagged(("а", "б", "в", "а", "б", "в")))
        self.assertFalse(is_repetition_flagged(("а", "б", "в", "г")))

    def test_repeated_ngram_share_bounds(self) -> None:
        self.assertEqual(repeated_ngram_share(("а",), 2), 0.0)
        self.assertEqual(repeated_ngram_share(("а", "б", "а", "б"), 2), 2 / 3)

    def test_idf_and_affinity(self) -> None:
        idf, default_idf = build_snapshot_idf(
            [["пиво", "квас"], ["пиво", "чай"], ["кофе"]]
        )
        # "пиво" in 2 of 3 docs -> low idf; "кофе" in 1 -> higher.
        self.assertLess(idf["пиво"], idf["кофе"])
        full = context_affinity(("кофе",), ("кофе",), idf, default_idf)
        self.assertEqual(full, 1.0)
        none_shared = context_affinity(("чай",), ("кофе",), idf, default_idf)
        self.assertEqual(none_shared, 0.0)
        self.assertIsNone(context_affinity(("а",), (), idf, default_idf))

    def test_distinct_n(self) -> None:
        value, basis = distinct_n([("а", "б", "в"), ("а", "б", "г")], 2)
        # bigrams: (а,б) x2, (б,в), (б,г) -> 3 unique / 4 total
        self.assertEqual(basis, 4)
        self.assertAlmostEqual(value or 0.0, 0.75)
        self.assertEqual(distinct_n([], 2), (None, 0))

    def test_latency_percentiles(self) -> None:
        records = [_record(latency_ms=float(i)) for i in range(1, 101)]
        result = latency_percentiles(records)
        self.assertAlmostEqual(result["latency_p50"] or 0.0, 50.0, delta=1.0)
        self.assertAlmostEqual(result["latency_p95"] or 0.0, 95.0, delta=1.0)

    def test_meme_regression(self) -> None:
        hit = _record(category="meme-bait", meme_hits=frozenset({0}))
        miss = _record(category="meme-bait")
        self.assertEqual(meme_regression([hit, miss], 2), (False, [1]))
        self.assertEqual(meme_regression([hit], 1), (True, []))
        self.assertEqual(meme_regression([hit], 0), (None, []))

    def test_metric_values_insufficient_markers(self) -> None:
        values = metric_values([_record()])
        self.assertIsNone(values["seeded_present_rate"])
        self.assertIsNone(values["freshness_reflection"])
        self.assertEqual(values["generation_success_rate"], [1.0])
        self.assertEqual(values["candidate_accept_rate"], [0.5])


class TestBootstrap(unittest.TestCase):
    def test_constant_samples_collapse(self) -> None:
        point, lo, hi = bootstrap_ci([0.5] * 50)
        self.assertEqual((point, lo, hi), (0.5, 0.5, 0.5))

    def test_ci_is_reproducible(self) -> None:
        samples = [0.0, 1.0] * 25
        self.assertEqual(bootstrap_ci(samples), bootstrap_ci(samples))

    def test_delta_significance(self) -> None:
        zeros, ones = [0.0] * 60, [1.0] * 60
        point, lo, hi, significant = delta_ci(zeros, ones)
        self.assertEqual(point, 1.0)
        self.assertTrue(significant)
        _, _, _, not_significant = delta_ci(zeros, zeros)
        self.assertFalse(not_significant)


class TestBootstrapProperties(unittest.TestCase):
    @settings(max_examples=50, deadline=None)
    @given(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=60,
        )
    )
    def test_ci_bounds_ordered_and_bracket_range(self, samples: list[float]) -> None:
        point, lo, hi = bootstrap_ci(samples, resamples=100)
        self.assertLessEqual(lo, hi)
        self.assertGreaterEqual(point, min(samples) - 1e-9)
        self.assertLessEqual(point, max(samples) + 1e-9)

    @settings(max_examples=50, deadline=None)
    @given(
        st.lists(st.text(alphabet="абв", min_size=1, max_size=3), max_size=12),
        st.integers(min_value=2, max_value=3),
    )
    def test_repeated_share_in_unit_interval(self, tokens: list[str], n: int) -> None:
        share = repeated_ngram_share(tuple(tokens), n)
        self.assertGreaterEqual(share, 0.0)
        self.assertLessEqual(share, 1.0)


class TestConfigFiles(unittest.TestCase):
    def test_matrix_loads_and_resolves(self) -> None:
        configs = load_matrix()
        self.assertIn("C0", configs)
        self.assertIn("CF", configs)
        resolved_cf = resolve_overrides(configs["CF"], configs)
        resolved_c0 = resolve_overrides(configs["C0"], configs)
        self.assertEqual(resolved_cf, resolved_c0)  # no V2 feature exists yet
        self.assertEqual(resolved_c0["reply_flavor_strength"], 0.0)

    def test_thresholds_preregistered_gates_present(self) -> None:
        thresholds = load_thresholds()
        for gate in ("phase5_promotion", "phase6_anticycle", "phase7_order4", "performance"):
            self.assertIn(gate, thresholds)


class TestSyntheticProtocol(unittest.IsolatedAsyncioTestCase):
    """End-to-end determinism on the synthetic snapshot (doc 05 §1.3)."""

    async def test_prompt_generation_deterministic(self) -> None:
        db_path, temp_dir = await build_synthetic_snapshot(messages=80)
        try:
            first = generate_prompts(
                db_path, chat_id=SYNTHETIC_CHAT_ID, seed=42, snapshot_label="t"
            )
            second = generate_prompts(
                db_path, chat_id=SYNTHETIC_CHAT_ID, seed=42, snapshot_label="t"
            )
            self.assertEqual(first.version, second.version)
            self.assertEqual(first.categories, second.categories)
            for name, prompts in first.categories.items():
                self.assertGreaterEqual(len(prompts), 30, name)
            different_seed = generate_prompts(
                db_path, chat_id=SYNTHETIC_CHAT_ID, seed=43, snapshot_label="t"
            )
            self.assertNotEqual(first.version, different_seed.version)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    async def test_run_reproducible_and_report_builds(self) -> None:
        db_path, temp_dir = await build_synthetic_snapshot(messages=80)
        try:
            prompt_set = generate_prompts(
                db_path, chat_id=SYNTHETIC_CHAT_ID, seed=42, snapshot_label="t"
            )
            configs = load_matrix()
            pair = {c: configs[c] for c in ("C0", "CF")}
            first, skipped = await run_matrix(
                db_source=db_path,
                chat_id=SYNTHETIC_CHAT_ID,
                configs=pair,
                prompt_set=prompt_set,
                seeds=(42,),
                generations=16,
            )
            second, _ = await run_matrix(
                db_source=db_path,
                chat_id=SYNTHETIC_CHAT_ID,
                configs=pair,
                prompt_set=prompt_set,
                seeds=(42,),
                generations=16,
            )
            # Bit-for-bit: identical metric summaries (latency excluded there).
            self.assertEqual(metrics_summary(first), metrics_summary(second))
            self.assertEqual(first["CF"].shared_with, "C0")
            report = build_report(
                runs=first,
                skipped=skipped,
                prompt_set=prompt_set,
                thresholds=load_thresholds(),
                snapshot_label="synthetic",
                seeds=(42,),
                generations=16,
                revision="test",
                date="2026-08-11",
                notes=[],
            )
            for section in (
                "## Config matrix",
                "## Metrics table",
                "## Per-category breakdown",
                "## Gates",
                "## Verdict per phase",
            ):
                self.assertIn(section, report)
            self.assertIn("insufficient data", report)  # temporal/seeded honesty
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()


# Keep Path import honest for tooling that trims unused imports.
_ = Path
