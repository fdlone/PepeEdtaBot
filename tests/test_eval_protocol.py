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
from tools.eval.report import (
    build_report,
    evaluate_gates,
    manual_summary,
    metrics_summary,
)
from tools.eval.run import ConfigRun, run_matrix
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
        for gate in (
            "phase2_entropy",
            "phase5_promotion",
            "phase6_anticycle",
            "phase7_order4",
            "performance",
        ):
            self.assertIn(gate, thresholds)


class TestPhase7Gate(unittest.TestCase):
    """Phase 7 shadow order-4 gate (ADR-002): a single-dimension sample-size
    gate. Below the eligible bar it is insufficient; at or above it, a selected
    share below the threshold renders fail — closing the phase without building
    the order-4 index (change: markov2r-phase7-order4-verdict)."""

    @staticmethod
    def _run(*, eligible_per_seed: int, selected_share: float) -> ConfigRun:
        # One telemetry snapshot per protocol seed, as run_matrix produces.
        snap = {
            "shadow_order4_eligible": eligible_per_seed,
            "shadow_order4_selected_share": selected_share,
        }
        return ConfigRun(
            config_id="C0",
            records=[_record()],
            telemetry=[dict(snap) for _ in range(3)],
        )

    def test_sufficient_sample_never_selected_renders_fail(self) -> None:
        # 3 × 2000 = 6000 eligible (> 1000 bar), 0% selected (< 10% threshold).
        rows = evaluate_gates(
            {"C0": self._run(eligible_per_seed=2000, selected_share=0.0)},
            load_thresholds(),
        )
        verdict, detail = next(
            (row[1], row[2]) for row in rows if row[0] == "phase7_order4"
        )
        self.assertEqual(verdict, "fail")
        self.assertIn("0.0%", detail)

    def test_below_sample_bar_stays_insufficient(self) -> None:
        # 3 × 200 = 600 eligible (< 1000), verdict withheld regardless of share.
        rows = evaluate_gates(
            {"C0": self._run(eligible_per_seed=200, selected_share=0.0)},
            load_thresholds(),
        )
        verdict = next(row[1] for row in rows if row[0] == "phase7_order4")
        self.assertEqual(verdict, "insufficient data")


class TestPhase6Gate(unittest.TestCase):
    """Phase 6 rate×harm gate (ADR-015): a two-dimensional AND-gate closes when
    the detection arm is decisively below its threshold, without the manual harm
    round (change: markov2r-phase6-anticycle-verdict)."""

    @staticmethod
    def _run(cycle_share: float, count: int = 200) -> ConfigRun:
        # ``cycle_share`` of the replies carry a period-2 token cycle.
        cyclic = round(cycle_share * count)
        records = [
            _record(
                reply_content=(
                    ("х", "у", "х", "у") if index < cyclic else ("а", "б", "в", "г")
                ),
                has_cycle=index < cyclic,
            )
            for index in range(count)
        ]
        return ConfigRun(config_id="C0", records=records)

    def test_rare_cycles_close_the_phase_without_the_harm_round(self) -> None:
        # ~0.5% cyclic — well below the 0.05 detection bar, whole CI under it.
        rows = evaluate_gates({"C0": self._run(0.005)}, load_thresholds())
        gate, verdict, detail = next(
            row for row in rows if row[0] == "phase6_anticycle"
        )
        self.assertEqual(verdict, "close")
        self.assertIn("without implementation", detail)
        self.assertIn("harm round is not required", detail)

    def test_frequent_cycles_defer_to_the_missing_harm_arm(self) -> None:
        # 20% cyclic — above the detection bar, so the harm arm decides and its
        # manual component is missing.
        rows = evaluate_gates({"C0": self._run(0.20)}, load_thresholds())
        verdict, detail = next(
            (row[1], row[2]) for row in rows if row[0] == "phase6_anticycle"
        )
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("manual round", detail)

    def test_no_baseline_is_insufficient(self) -> None:
        rows = evaluate_gates({}, load_thresholds())
        verdict = next(row[1] for row in rows if row[0] == "phase6_anticycle")
        self.assertEqual(verdict, "insufficient data")


class TestPhase2Gate(unittest.TestCase):
    """The Phase 2 gate is four-part and asymmetric (doc 03 Phase 2 acceptance,
    thresholds pre-registered in eval_thresholds.yaml)."""

    @staticmethod
    def _run(
        config_id: str,
        replies: list[str],
        *,
        copy_share: float = 0.0,
        latency_ms: float = 10.0,
    ) -> ConfigRun:
        records = [
            _record(
                reply_text=reply,
                reply_content=tuple(reply.split()),
                affinity=0.2,
                is_copy=index < round(copy_share * len(replies)),
                latency_ms=latency_ms,
            )
            for index, reply in enumerate(replies)
        ]
        return ConfigRun(config_id=config_id, records=records)

    @staticmethod
    def _varied(count: int, *, unique: bool) -> list[str]:
        if unique:
            return [f"токен{i} слово{i} хвост{i}" for i in range(count)]
        return ["одно и то же"] * count

    def test_no_arm_reports_insufficient(self) -> None:
        rows = evaluate_gates({"C0": self._run("C0", self._varied(10, unique=False))},
                              load_thresholds())
        gate, verdict, _ = next(row for row in rows if row[0].startswith("phase2"))
        self.assertEqual(gate, "phase2_entropy")
        self.assertEqual(verdict, "insufficient data")

    def test_aliased_arm_is_not_a_verdict(self) -> None:
        baseline = self._run("C0", self._varied(10, unique=False))
        arm = ConfigRun(config_id="C1", records=baseline.records, shared_with="C0")
        rows = evaluate_gates({"C0": baseline, "C1": arm}, load_thresholds())
        verdict = next(row[1] for row in rows if row[0] == "phase2_entropy[C1]")
        self.assertEqual(verdict, "insufficient data")

    def test_diversity_gain_passes(self) -> None:
        rows = evaluate_gates(
            {
                "C0": self._run("C0", self._varied(60, unique=False)),
                "C1": self._run("C1", self._varied(60, unique=True)),
            },
            load_thresholds(),
        )
        verdict, detail = next(
            (row[1], row[2]) for row in rows if row[0] == "phase2_entropy[C1]"
        )
        self.assertEqual(verdict, "pass", detail)
        self.assertIn("distinct-2", detail)

    def test_no_measurable_effect_is_a_fail_not_a_pass(self) -> None:
        replies = self._varied(60, unique=False)
        rows = evaluate_gates(
            {"C0": self._run("C0", replies), "C1": self._run("C1", list(replies))},
            load_thresholds(),
        )
        verdict, detail = next(
            (row[1], row[2]) for row in rows if row[0] == "phase2_entropy[C1]"
        )
        self.assertEqual(verdict, "fail", detail)
        self.assertIn("did not rise", detail)

    def test_copy_rise_disqualifies_despite_diversity(self) -> None:
        rows = evaluate_gates(
            {
                "C0": self._run("C0", self._varied(60, unique=False)),
                "C1": self._run("C1", self._varied(60, unique=True), copy_share=0.5),
            },
            load_thresholds(),
        )
        verdict, detail = next(
            (row[1], row[2]) for row in rows if row[0] == "phase2_entropy[C1]"
        )
        self.assertEqual(verdict, "fail", detail)
        self.assertIn("copy rose", detail)

    def test_latency_over_budget_disqualifies(self) -> None:
        rows = evaluate_gates(
            {
                "C0": self._run("C0", self._varied(60, unique=False)),
                "C1": self._run("C1", self._varied(60, unique=True), latency_ms=900.0),
            },
            load_thresholds(),
        )
        verdict, detail = next(
            (row[1], row[2]) for row in rows if row[0] == "phase2_entropy[C1]"
        )
        self.assertEqual(verdict, "fail", detail)
        self.assertIn("p95 over budget", detail)


class TestManualEvalSummary(unittest.TestCase):
    """The section must describe the run it belongs to (spec: generation-eval).

    It used to print "not conducted" unconditionally, so a report could carry
    the manual numbers in its gate line and deny the round two sections later.
    """

    AGGREGATE = {
        "ranking_version": "b81b634",
        "raters": 3,
        "agreement": 0.17,
        "rated": 20,
        "real": 8,
        "control_rated": 20,
        "control_real": 2,
        "decoy_rated": 5,
        "decoy_real": 1,
        "_comment": "phrase-free note that must never reach the report",
    }

    def test_absent_rating_says_so(self) -> None:
        self.assertEqual(
            manual_summary(None),
            ["Not conducted in this run (first required at the Phase 4 gate)."],
        )

    def test_supplied_rating_is_reported(self) -> None:
        text = "\n".join(manual_summary(self.AGGREGATE))
        self.assertNotIn("Not conducted", text)
        for fragment in ("3", "0.17", "20", "8", "40%", "2", "10%", "5", "1", "20%"):
            self.assertIn(fragment, text)
        self.assertIn("b81b634", text)

    def test_unknown_keys_never_reach_the_report(self) -> None:
        # The aggregate sits next to verbatim chat phrases; only known count
        # keys are read, so a stray note cannot ride into a committed report.
        self.assertNotIn("phrase-free note", "\n".join(manual_summary(self.AGGREGATE)))

    def test_single_rater_states_agreement_is_unavailable(self) -> None:
        solo = {**self.AGGREGATE, "raters": 1, "agreement": None}
        self.assertIn("unavailable (single rater)", "\n".join(manual_summary(solo)))


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
