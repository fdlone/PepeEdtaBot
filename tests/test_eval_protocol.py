"""Tests for the Markov 2.0R eval protocol runner (tools/eval, doc 05).

Covers: metric known-answer cases (§3), stdlib bootstrap (§4), deterministic
prompt generation (§1.2), run reproducibility on the synthetic snapshot (§1.3),
and the first hypothesis property tests — the TZ §19 pattern later phases
extend.
"""
from __future__ import annotations

import math
import random
import unittest
from pathlib import Path
from statistics import mean
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.generation_telemetry import CandidateRoute
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
from tools.eval.prompts import PromptSet, generate_prompts
from tools.eval.report import (
    _apply_mode_requirement,
    build_report,
    evaluate_gates,
    manual_summary,
    metrics_summary,
)
from tools.eval.run import ConfigRun, run_config_seed, run_matrix
from tools.eval.synthetic import SYNTHETIC_CHAT_ID, build_synthetic_snapshot


def _prompt_set_stub() -> PromptSet:
    return PromptSet(
        version="stub",
        snapshot_label="stub",
        seed=42,
        categories={"generic": ["пиво сегодня"]},
        memes=[],
    )


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

    def test_meme_regression_counts_reproduced(self) -> None:
        # M3R-130: числитель доли, а не бинарный вердикт — вердикт формирует
        # отчёт относительно C0.
        hit = _record(category="meme-bait", meme_hits=frozenset({0}))
        miss = _record(category="meme-bait")
        self.assertEqual(meme_regression([hit, miss], 2), (1, [1]))
        self.assertEqual(meme_regression([hit], 1), (1, []))
        self.assertEqual(meme_regression([hit], 0), (0, []))

    def test_meme_regression_ignores_other_categories(self) -> None:
        # Мемы засчитываются только по своей категории промптов.
        generic = _record(category="generic", meme_hits=frozenset({0}))
        self.assertEqual(meme_regression([generic], 2), (0, [0, 1]))

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

    def test_paired_interval_is_narrower_than_independent_resampling(self) -> None:
        """Дельта считается по парам, а не по двум независимым выборкам.

        Матрица конфигураций — парный дизайн: армы идут по одним и тем же
        промптам и сидам, различаясь одной ручкой. Независимый ресэмплинг
        оценивал `Var_A + Var_B` вместо дисперсии парной разности и раздувал
        интервал тем сильнее, чем выше корреляция армов. Ошибка направленная:
        она рождала ложные вердикты «эффекта нет» — а такой вердикт в этом
        проекте закрывает направление окончательно.
        """
        rng = random.Random(7)
        shared = [rng.random() for _ in range(200)]
        arm_a = [value + rng.gauss(0, 0.02) for value in shared]
        arm_b = [value + 0.03 + rng.gauss(0, 0.02) for value in shared]

        point, lo, hi, significant = delta_ci(arm_a, arm_b)

        independent = sorted(
            mean(random.Random(index).choices(arm_b, k=len(arm_b)))
            - mean(random.Random(index + 5000).choices(arm_a, k=len(arm_a)))
            for index in range(400)
        )
        independent_width = independent[389] - independent[10]

        self.assertAlmostEqual(point, mean(arm_b) - mean(arm_a), places=9)
        self.assertLess(
            hi - lo,
            independent_width,
            "парный интервал должен быть уже независимого на коррелированных армах",
        )
        self.assertTrue(significant, "реальный сдвиг 0.03 обязан быть значимым")

    def test_incomplete_pairs_are_dropped_whole(self) -> None:
        """Полупара — не наблюдение разности.

        Часть метрик фильтруется по `record.success`, поэтому списки армов
        бывают разной длины. Наблюдение, где один арм не ответил, выбывает
        целиком: иначе разность считалась бы между несвязанными записями.
        """
        arm_a = [0.0] * 40
        arm_b = [1.0] * 40 + [99.0] * 5

        point, _, _, _ = delta_ci(arm_a, arm_b)

        self.assertEqual(point, 1.0, "хвост без пары попал в оценку")

    def test_arms_losing_different_prompts_stay_aligned(self) -> None:
        """Главный случай: армы не отвечают на РАЗНЫХ промптах.

        Метрики с фильтром по `success` раньше просто выбрасывали такие
        записи, и списки армов становились короче на разные позиции. Тогда
        пара по позиции сшивала разные промпты — то есть парная оценка была бы
        не честнее непарной, а хуже неё. Выравнивание метками `nan` (см.
        `metrics._aligned`) существует ровно ради этого случая.
        """
        # Промпты 0..4; арм A не ответил на 1, арм B — на 3.
        arm_a = [10.0, math.nan, 30.0, 40.0, 50.0]
        arm_b = [11.0, 21.0, 31.0, math.nan, 51.0]

        point, _, _, _ = delta_ci(arm_a, arm_b)

        # Остаются пары 0, 2, 4 — на каждой разность ровно 1.0.
        self.assertEqual(point, 1.0, "пары сшиты не по своим промптам")

    def test_single_arm_ci_ignores_the_alignment_markers(self) -> None:
        """Одиночному интервалу выравнивание не нужно — метки отбрасываются."""
        point, lo, hi = bootstrap_ci([1.0, math.nan, 1.0, math.nan])

        self.assertEqual((point, lo, hi), (1.0, 1.0, 1.0))


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
            # Опечатка в имени блока не ловится нигде больше: report.py читает
            # пороги через .get(...) с хардкод-дефолтом, поэтому незнакомый ключ
            # молча уводит гейт на дефолты и печатает вердикт как ни в чём не
            # бывало. Этот кортеж — единственная защита.
            "phase9_interp",
            "l1_hot_channel",
            "pool_composition",
            "selection_window",
            "performance",
        ):
            self.assertIn(gate, thresholds)


class TestMemeRegressionGate(unittest.TestCase):
    """M3R-130: набор с порогом поддержки, гейт относительно C0 по доле."""

    @staticmethod
    def _prompt_set(meme_count: int) -> PromptSet:
        from tools.eval.prompts import PromptSet as _PromptSet

        return _PromptSet(
            version="stub",
            snapshot_label="stub",
            seed=42,
            categories={"meme-bait": ["наживка"]},
            memes=[[f"мем{index}"] for index in range(meme_count)],
        )

    @staticmethod
    def _thresholds(**extra: object) -> dict:
        config: dict = {
            "support_min": 2,
            "reproduced_tolerance": 0.10,
            "set_min_size": 8,
        }
        config.update(extra)
        return {"meme_regression": config}

    @staticmethod
    def _arm(config_id: str, hits: set[int]) -> ConfigRun:
        return ConfigRun(
            config_id=config_id,
            records=[
                _record(category="meme-bait", meme_hits=frozenset(hits))
            ],
        )

    def _rows(
        self, runs: dict, memes: int = 10, thresholds: dict | None = None
    ) -> dict[str, tuple[str, str]]:
        from tools.eval.report import _meme_regression_rows

        rows = _meme_regression_rows(
            runs, self._prompt_set(memes), thresholds or self._thresholds()
        )
        return {row[0]: (row[1], row[2]) for row in rows}

    def test_arm_holding_memory_passes(self) -> None:
        rows = self._rows({
            "C0": self._arm("C0", set(range(8))),
            "C4": self._arm("C4", set(range(8))),
        })
        self.assertEqual(rows["meme_regression[C4]"][0], "pass")
        self.assertEqual(rows["meme_regression[C0]"][0], "baseline")
        self.assertIn("8/10", rows["meme_regression[C4]"][1])

    def test_arm_erasing_memes_fails_against_the_baseline(self) -> None:
        # C0 8/10, арм 5/10 — разрыв 0.30 при допуске 0.10.
        rows = self._rows({
            "C0": self._arm("C0", set(range(8))),
            "C4": self._arm("C4", set(range(5))),
        })
        verdict, detail = rows["meme_regression[C4]"]
        self.assertEqual(verdict, "fail")
        self.assertIn("50%", detail)
        self.assertIn("80%", detail)

    def test_drop_within_tolerance_is_not_a_failure(self) -> None:
        # 8/10 против 9/10 — один мем, 0.10 не превышено.
        rows = self._rows({
            "C0": self._arm("C0", set(range(9))),
            "C4": self._arm("C4", set(range(8))),
        })
        self.assertEqual(rows["meme_regression[C4]"][0], "pass")

    def test_set_below_minimum_is_insufficient_not_a_verdict(self) -> None:
        rows = self._rows(
            {"C0": self._arm("C0", {0}), "C4": self._arm("C4", set())},
            memes=5,
        )
        verdict, detail = rows["meme_regression"]
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("5 entries", detail)
        self.assertIn("8", detail)

    def test_empty_set_is_insufficient(self) -> None:
        rows = self._rows({"C0": self._arm("C0", set())}, memes=0)
        self.assertEqual(rows["meme_regression"][0], "insufficient data")

    def test_unregistered_thresholds_do_not_fall_back_to_defaults(self) -> None:
        rows = self._rows(
            {"C0": self._arm("C0", set(range(8)))},
            thresholds={"meme_regression": {}},
        )
        verdict, detail = rows["meme_regression"]
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("not registered", detail)


class TestMemeSupportFloor(unittest.TestCase):
    """Порог поддержки при построении набора (M3R-130, задача 2.1)."""

    def _snapshot(self, path: Path) -> None:
        import sqlite3

        con = sqlite3.connect(path)
        con.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, chat_id INTEGER, "
            "normalized_text TEXT)"
        )
        con.execute(
            "CREATE TABLE transitions (chat_id INTEGER, w1 TEXT, w2 TEXT, "
            "w3 TEXT, cnt INTEGER)"
        )
        con.execute(
            "CREATE TABLE chat_hot_ngrams (chat_id INTEGER, w1 TEXT, w2 TEXT, "
            "w3 TEXT, cnt INTEGER)"
        )
        messages = [
            f"сообщение номер {index} про котиков и погоду" for index in range(40)
        ]
        con.executemany(
            "INSERT INTO messages (chat_id, normalized_text) VALUES (1, ?)",
            [(text,) for text in messages],
        )
        con.executemany(
            "INSERT INTO transitions VALUES (1, ?, ?, ?, ?)",
            [("а", "б", token, 5) for token in ("котиков", "погоду", "номер")],
        )
        con.executemany(
            "INSERT INTO chat_hot_ngrams VALUES (1, ?, ?, ?, ?)",
            [
                ("котиков", "погоду", "", 2),   # поддержка 2 — проходит
                ("номер", "котиков", "", 1),    # поддержка 1 — отсеивается
            ],
        )
        con.commit()
        con.close()

    def test_floor_keeps_only_supported_ngrams(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "snapshot.db"
            self._snapshot(db)
            with_floor = generate_prompts(
                db, chat_id=1, seed=42, snapshot_label="t",
                per_category=4, meme_support_min=2,
            )
            without_floor = generate_prompts(
                db, chat_id=1, seed=42, snapshot_label="t",
                per_category=4, meme_support_min=1,
            )
        self.assertEqual(with_floor.memes, [["котиков", "погоду"]])
        self.assertIn(["номер", "котиков"], without_floor.memes)
        # Разный набор — разная версия: разрыв сопоставимости виден.
        self.assertNotEqual(with_floor.version, without_floor.version)


class TestPhase5CorpusPrecondition(unittest.TestCase):
    """gate-phase5-ndocs-floor: корпусное предусловие гейта фазы 5.

    Величины читаются из ИСХОДНОГО снапшота до оконной подготовки df
    (design D1): чтение рабочей копии всегда видело бы retention-окно,
    которое раннер сам и записал, и предусловие стало бы штампом.
    """

    FLOOR = 500
    CEILING = 0.60

    def setUp(self) -> None:
        import tempfile

        self.tmp = Path(tempfile.mkdtemp(prefix="test_ndocs_floor_"))
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True)
        )
        self.db = self.tmp / "source.db"

    def _make_source(
        self, *, n_docs: int, singles: int, plural: int, chat_id: int = 1
    ) -> None:
        import sqlite3

        con = sqlite3.connect(self.db)
        con.execute(
            "CREATE TABLE chat_model_volume "
            "(chat_id INTEGER PRIMARY KEY, n_docs INTEGER)"
        )
        con.execute(
            "CREATE TABLE markov_token_df "
            "(chat_id INTEGER, token TEXT, messages_seen INTEGER, "
            "PRIMARY KEY (chat_id, token))"
        )
        con.execute(
            "INSERT INTO chat_model_volume VALUES (?, ?)", (chat_id, n_docs)
        )
        rows = [(chat_id, f"один{i}", 1) for i in range(singles)]
        rows += [(chat_id, f"част{i}", 3) for i in range(plural)]
        con.executemany("INSERT INTO markov_token_df VALUES (?, ?, ?)", rows)
        con.commit()
        con.close()

    @staticmethod
    def _thresholds(**extra: object) -> dict:
        config: dict = {
            "n_docs_min": TestPhase5CorpusPrecondition.FLOOR,
            "df_singleton_share_max": TestPhase5CorpusPrecondition.CEILING,
        }
        config.update(extra)
        return {"phase5_promotion": config}

    def _verdict(self, facts: object, thresholds: dict | None = None) -> tuple[str, str]:
        from tools.eval.report import _phase5_arm_verdict

        baseline = ConfigRun(config_id="C0", records=[_record()])
        arm = ConfigRun(config_id="C4", records=[_record()])
        return _phase5_arm_verdict(
            baseline, arm, thresholds or self._thresholds(), facts
        )

    def test_facts_read_known_numbers_from_the_source(self) -> None:
        from tools.eval.run import read_df_corpus_facts

        self._make_source(n_docs=35, singles=140, plural=32)
        facts = read_df_corpus_facts(self.db, 1)
        self.assertEqual(facts.n_docs, 35)
        assert facts.singleton_share is not None
        self.assertAlmostEqual(facts.singleton_share, 140 / 172)
        self.assertIsNone(facts.error)

    def test_window_population_does_not_leak_into_the_verdict(self) -> None:
        # Ловушка design D1/Context §2: после оконной подготовки df рабочая
        # копия отдаёт другие числа; в вердикт обязаны идти прочитанные из
        # исходника. Перенос чтения после подготовки роняет оба assert'а.
        import shutil as _sh
        import sqlite3

        from tools.eval.run import read_df_corpus_facts

        self._make_source(n_docs=35, singles=140, plural=32)
        facts_before = read_df_corpus_facts(self.db, 1)

        working_copy = self.tmp / "copy.db"
        _sh.copyfile(self.db, working_copy)
        con = sqlite3.connect(working_copy)
        # Та же арифметика, что у _populate_token_df: окно из 1000 сообщений
        # перетирает n_docs и досыпает df.
        con.execute("UPDATE chat_model_volume SET n_docs = 1000 WHERE chat_id = 1")
        con.executemany(
            "INSERT INTO markov_token_df VALUES (?, ?, ?) "
            "ON CONFLICT(chat_id, token) DO UPDATE SET "
            "messages_seen = messages_seen + 1",
            [(1, f"окно{i}", 5) for i in range(400)],
        )
        con.commit()
        con.close()

        facts_after = read_df_corpus_facts(working_copy, 1)
        self.assertNotEqual(facts_after.n_docs, facts_before.n_docs)
        verdict, detail = self._verdict(facts_before)
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("n_docs 35", detail)
        self.assertNotIn("1000", detail.split("floor")[0])

    def test_empty_df_reads_as_df_empty_not_zero_division(self) -> None:
        from tools.eval.run import read_df_corpus_facts

        self._make_source(n_docs=5, singles=0, plural=0)
        facts = read_df_corpus_facts(self.db, 1)
        self.assertEqual(facts.n_docs, 5)
        self.assertIsNone(facts.singleton_share)
        self.assertIsNone(facts.error)
        verdict, detail = self._verdict(facts)
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("df", detail)

    def test_unreadable_source_is_insufficient_not_a_crash(self) -> None:
        from tools.eval.run import read_df_corpus_facts

        facts = read_df_corpus_facts(self.tmp / "no_such.db", 1)
        self.assertIsNotNone(facts.error)
        verdict, detail = self._verdict(facts)
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("unreadable", detail)

    def test_volume_shortfall_names_floor_and_measured_number(self) -> None:
        from tools.eval.run import DfCorpusFacts

        facts = DfCorpusFacts(n_docs=35, singleton_share=0.30)
        verdict, detail = self._verdict(facts)
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("35", detail)
        self.assertIn("500", detail)

    def test_shape_shortfall_names_ceiling_and_measured_share(self) -> None:
        from tools.eval.run import DfCorpusFacts

        facts = DfCorpusFacts(n_docs=800, singleton_share=0.81)
        verdict, detail = self._verdict(facts)
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("81%", detail)
        self.assertIn("60%", detail)

    def test_missing_threshold_keys_do_not_fall_back_to_code_defaults(self) -> None:
        # Design D4: дефолт в коде — это порог, не прошедший предрегистрацию.
        from tools.eval.run import DfCorpusFacts

        facts = DfCorpusFacts(n_docs=100_000, singleton_share=0.0)
        verdict, detail = self._verdict(facts, thresholds={"phase5_promotion": {}})
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("not registered", detail)

    def test_both_parts_met_lifts_the_precondition(self) -> None:
        # Предусловие снято — вердикт решают остальные условия гейта. На
        # минимальной фикстуре без seeded-телеметрии они сами дают
        # insufficient, но строки предусловия среди причин быть не должно.
        from tools.eval.run import DfCorpusFacts

        facts = DfCorpusFacts(n_docs=800, singleton_share=0.30)
        verdict, detail = self._verdict(facts)
        self.assertIn("prod corpus: n_docs 800", detail)
        self.assertNotIn("below the floor", detail)
        self.assertNotIn("above the ceiling", detail)
        self.assertNotIn("prod-accumulated df", detail)

    def test_report_prints_measured_quantities_next_to_the_verdict(self) -> None:
        from tools.eval.run import DfCorpusFacts

        facts = DfCorpusFacts(n_docs=35, singleton_share=0.81)
        runs = {
            "C0": ConfigRun(config_id="C0", records=[_record()]),
            "C4": ConfigRun(config_id="C4", records=[_record()]),
        }
        report = build_report(
            runs=runs,
            skipped=[],
            prompt_set=_prompt_set_stub(),
            thresholds=self._thresholds(),
            snapshot_label="test",
            seeds=(42,),
            generations=1,
            revision="test",
            date="2026-09-01",
            notes=[],
            df_facts=facts,
        )
        self.assertIn("n_docs 35 (floor 500)", report)
        self.assertIn("singleton share 81% (ceiling 60%)", report)


class TestRouteBreakdownSection(unittest.TestCase):
    """M3R-103, reporting half: the per-route table of the report.

    Two denominators printed separately (design D3), an off route is "not
    attempted" rather than a zero row (D4), and the bit-for-bit summary does
    not see route fields at all (D5)."""

    @staticmethod
    def _report(runs: dict[str, ConfigRun]) -> str:
        return build_report(
            runs=runs,
            skipped=[],
            prompt_set=_prompt_set_stub(),
            thresholds=load_thresholds(),
            snapshot_label="test",
            seeds=(42,),
            generations=1,
            revision="test",
            date="2026-09-01",
            notes=[],
        )

    @staticmethod
    def _row(report: str, config_id: str, route: str) -> str:
        return next(
            line
            for line in report.splitlines()
            if line.startswith(f"| {config_id} | {route} |")
        )

    def test_two_denominators_are_printed_separately(self) -> None:
        records = [
            _record(
                winner_route="seeded",
                pool_routes=("vanilla", "vanilla", "seeded"),
                affinity=0.4,
                latency_ms=30.0,
            ),
            _record(
                winner_route="vanilla",
                pool_routes=("vanilla", "vanilla", "vanilla"),
                affinity=0.2,
                latency_ms=10.0,
            ),
        ]
        arm = ConfigRun(
            config_id="C4",
            records=records,
            telemetry=[
                {
                    "route_seeded_attempts": 2,
                    "route_seeded_rejected_F2_context_copy": 3,
                    "route_vanilla_attempts": 2,
                }
            ],
        )
        report = self._report(
            {"C0": ConfigRun(config_id="C0", records=[_record()]), "C4": arm}
        )
        row = self._row(report, "C4", "seeded")
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        # config, route, attempts, pool share, presence, win given present,
        # winners' affinity, winners' copy, latency, rejected
        self.assertEqual(cells[2], "2")
        self.assertTrue(cells[3].startswith("0.167"))  # (1/3 + 0) / 2
        self.assertTrue(cells[4].startswith("0.500"))  # present in 1 of 2
        self.assertTrue(cells[5].startswith("1.000"))  # won the one it was in
        self.assertTrue(cells[6].startswith("0.400"))  # its winner's affinity
        self.assertEqual(cells[8], "30.0 / 10.0")
        self.assertEqual(cells[9], "F2_context_copy 3")

    def test_route_that_never_ran_is_not_a_zero_row(self) -> None:
        baseline = ConfigRun(
            config_id="C0",
            records=[_record(winner_route="vanilla", pool_routes=("vanilla",))],
            telemetry=[{"route_vanilla_attempts": 1}],
        )
        report = self._report({"C0": baseline})
        self.assertIn("not attempted", self._row(report, "C0", "seeded"))
        self.assertNotIn("not attempted", self._row(report, "C0", "vanilla"))

    def test_shared_configuration_has_no_rows_of_its_own(self) -> None:
        base = ConfigRun(config_id="C0", records=[_record(pool_routes=("vanilla",))])
        alias = ConfigRun(config_id="CF", records=base.records, shared_with="C0")
        report = self._report({"C0": base, "CF": alias})
        self.assertNotIn("| CF | vanilla |", report)

    def test_summary_is_blind_to_route_fields(self) -> None:
        plain = metrics_summary({"C0": ConfigRun(config_id="C0", records=[_record()])})
        attributed = metrics_summary(
            {
                "C0": ConfigRun(
                    config_id="C0",
                    records=[_record(winner_route="seeded", pool_routes=("seeded",))],
                )
            }
        )
        self.assertEqual(plain, attributed)

    def test_route_telemetry_sums_rejections_by_failure_class(self) -> None:
        from app.core.generation_telemetry import GenerationTelemetry
        from tools.eval.run import route_telemetry

        telemetry = GenerationTelemetry()
        telemetry.note_routes(
            attempted={"vanilla", "seeded"}, present={"vanilla"}, winner="vanilla"
        )
        telemetry.note_route_rejected("seeded", "short_context_copy")
        telemetry.note_route_rejected("seeded", "context_heavy")
        telemetry.note_route_rejected("seeded", "no such reason")
        flat = route_telemetry(telemetry)
        self.assertEqual(flat["route_seeded_attempts"], 1)
        self.assertEqual(flat["route_seeded_present"], 0)
        self.assertEqual(flat["route_vanilla_won"], 1)
        self.assertEqual(flat["route_seeded_rejected_F2_context_copy"], 2)
        self.assertEqual(flat["route_seeded_rejected_unmapped"], 1)


class TestHotSeedDraw(unittest.TestCase):
    """M3R-145: the harness draws L1 seeds the pipeline's way."""

    def test_roll_first_then_choose(self) -> None:
        from tools.eval.run import draw_hot_seed

        pool = [("пиво", "сегодня"), ("опять", "ты")]
        self.assertEqual(draw_hot_seed(pool, 0.0, random.Random(1)), (False, None))
        self.assertEqual(draw_hot_seed([], 1.0, random.Random(1)), (True, None))
        rolled, seed = draw_hot_seed(pool, 1.0, random.Random(1))
        self.assertTrue(rolled)
        self.assertIn(tuple(seed or ()), pool)

    def test_zero_chance_consumes_no_draw(self) -> None:
        from tools.eval.run import draw_hot_seed

        rng = random.Random(7)
        draw_hot_seed([("а", "б")], 0.0, rng)
        self.assertEqual(rng.random(), random.Random(7).random())

    def test_same_seed_same_choice(self) -> None:
        from tools.eval.run import draw_hot_seed

        pool = [(f"w{i}", "x") for i in range(20)]
        first = draw_hot_seed(pool, 1.0, random.Random(42))
        second = draw_hot_seed(pool, 1.0, random.Random(42))
        self.assertEqual(first, second)


class TestL1Gate(unittest.TestCase):
    """M3R-145 gate: coverage gates the verdict, the meme rate is must-improve
    in noctx, copy is must-not-worsen in both modes, and no round means no
    pass."""

    @staticmethod
    def _records(
        *,
        n: int = 20,
        seeded: int = 0,
        meme: bool = False,
        copy: bool = False,
        affinity: float = 0.3,
    ) -> list[GenRecord]:
        records = []
        for index in range(n):
            records.append(
                _record(
                    category="meme-bait",
                    meme_hits=frozenset({0}) if meme else frozenset(),
                    is_copy=copy,
                    affinity=affinity,
                    seed_drawn=index < seeded,
                    start_source="seed" if index < seeded else "global",
                )
            )
        return records

    def _rows(self, runs: dict[str, ConfigRun], mode: str) -> dict[str, tuple[str, str]]:
        rows = evaluate_gates(runs, load_thresholds(), None, mode, None, None)
        return {row[0]: (row[1], row[2]) for row in rows}

    @staticmethod
    def _arm_verdict(runs: dict[str, ConfigRun], mode: str) -> tuple[str, str]:
        # The arm verdict BEFORE the two-mode downgrade: a one-mode run always
        # reads `insufficient data` through evaluate_gates (M3R-140), so the
        # fail/pass logic is tested on the function that computes it.
        from tools.eval.report import _l1_arm_verdict

        return _l1_arm_verdict(runs["C0"], runs["C7a"], load_thresholds(), None, mode)

    def test_coverage_below_floor_is_insufficient_not_fail(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C7a": ConfigRun(config_id="C7a", records=self._records(meme=True, copy=True)),
        }
        verdict, detail = self._rows(runs, "noctx")["l1_hot_channel[C7a]"]
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("coverage below the floor", detail)

    def test_meme_rise_without_round_is_insufficient(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C7a": ConfigRun(config_id="C7a", records=self._records(seeded=4, meme=True)),
        }
        verdict, detail = self._rows(runs, "noctx")["l1_hot_channel[C7a]"]
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("historical_meme_rate Δ 1.000", detail)
        self.assertIn("connectedness round", detail)
        # The two-mode requirement is named too.
        self.assertIn("requires both context modes", detail)

    def test_no_meme_rise_is_a_fail(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C7a": ConfigRun(config_id="C7a", records=self._records(seeded=4)),
        }
        verdict, detail = self._arm_verdict(runs, "noctx")
        self.assertEqual(verdict, "fail")
        self.assertIn("did not rise significantly", detail)

    def test_copy_rise_fails_in_ctx_too(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C7a": ConfigRun(config_id="C7a", records=self._records(copy=True)),
        }
        verdict, detail = self._arm_verdict(runs, "ctx")
        self.assertEqual(verdict, "fail")
        self.assertIn("copy rose significantly", detail)
        self.assertIn("seed draw not modelled", detail)

    def test_affinity_drop_fails_in_ctx(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(affinity=0.5)),
            "C7a": ConfigRun(config_id="C7a", records=self._records(affinity=0.1)),
        }
        verdict, detail = self._arm_verdict(runs, "ctx")
        self.assertEqual(verdict, "fail")
        self.assertIn("affinity without copies dropped", detail)

    def test_no_arm_reports_insufficient(self) -> None:
        runs = {"C0": ConfigRun(config_id="C0", records=self._records())}
        verdict, _ = self._rows(runs, "noctx")["l1_hot_channel"]
        self.assertEqual(verdict, "insufficient data")

    def test_report_prints_hot_seed_counters(self) -> None:
        run = ConfigRun(
            config_id="C0",
            records=self._records(),
            telemetry=[{"hot_ngram_draws": 12, "hot_ngram_empty_rate": 1.0}],
        )
        report = build_report(
            runs={"C0": run},
            skipped=[],
            prompt_set=_prompt_set_stub(),
            thresholds=load_thresholds(),
            snapshot_label="test",
            seeds=(42,),
            generations=1,
            revision="test",
            date="2026-09-01",
            notes=[],
        )
        self.assertIn("hot-ngram seeds: 12 draws, empty 100%", report)


class TestStartSourceResolution(unittest.TestCase):
    """M3R-110 (D2): the winner's start source follows extension and
    mutation back to the traced attempt."""

    def test_follows_extension_then_mutation(self) -> None:
        from tools.eval.run import resolve_start_source

        attempts = {"а б": "context"}
        derived = {"а б в": "а б", "а г в": "а б в"}
        self.assertEqual(resolve_start_source("а г в", attempts, derived), "context")
        self.assertEqual(resolve_start_source("а б", attempts, derived), "context")

    def test_unknown_text_is_none_not_a_guess(self) -> None:
        from tools.eval.run import resolve_start_source

        self.assertIsNone(resolve_start_source("x", {"а": "global"}, {}))
        # A self-loop in the derivation must not spin.
        self.assertIsNone(resolve_start_source("x", {}, {"x": "x"}))


class TestPoolCompositionGate(unittest.TestCase):
    """M3R-110 gate: coverage is the shift of the context-start share, affinity
    is must-improve in ctx, the window may not narrow, and no round means no
    pass; noctx only checks that nothing moved."""

    @staticmethod
    def _records(
        *,
        n: int = 20,
        context_starts: int = 0,
        affinity: float = 0.3,
        copy: bool = False,
        window_escape: int = 2,
        pool_ecb: int = 5,
    ) -> list[GenRecord]:
        return [
            _record(
                start_source="context" if index < context_starts else "global",
                affinity=affinity,
                is_copy=copy,
                window_escape=window_escape,
                pool_ecb=pool_ecb,
            )
            for index in range(n)
        ]

    @staticmethod
    def _arm_verdict(runs: dict[str, ConfigRun], mode: str) -> tuple[str, str]:
        from tools.eval.report import _pool_arm_verdict

        return _pool_arm_verdict(runs["C0"], runs["C8b30"], load_thresholds(), None, mode)

    def test_unmoved_start_budget_is_insufficient_not_a_verdict(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(context_starts=4)),
            "C8b30": ConfigRun(
                config_id="C8b30", records=self._records(context_starts=4, affinity=0.9)
            ),
        }
        verdict, detail = self._arm_verdict(runs, "ctx")
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("start budget did not move", detail)

    def test_affinity_rise_without_round_is_insufficient(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(context_starts=4)),
            "C8b30": ConfigRun(
                config_id="C8b30", records=self._records(context_starts=10, affinity=0.9)
            ),
        }
        verdict, detail = self._arm_verdict(runs, "ctx")
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("context starts 50.0% vs 20.0%", detail)
        self.assertIn("connectedness round", detail)

    def test_narrowed_window_fails_despite_affinity(self) -> None:
        runs = {
            "C0": ConfigRun(
                config_id="C0", records=self._records(context_starts=4, window_escape=3)
            ),
            "C8b30": ConfigRun(
                config_id="C8b30",
                records=self._records(context_starts=10, affinity=0.9, window_escape=1),
            ),
        }
        verdict, detail = self._arm_verdict(runs, "ctx")
        self.assertEqual(verdict, "fail")
        self.assertIn("window escape dropped", detail)

    def test_flat_affinity_is_a_fail(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(context_starts=4)),
            "C8b30": ConfigRun(config_id="C8b30", records=self._records(context_starts=10)),
        }
        verdict, detail = self._arm_verdict(runs, "ctx")
        self.assertEqual(verdict, "fail")
        self.assertIn("did not rise significantly", detail)

    def test_noctx_with_nothing_moved_passes_at_arm_level(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C8b30": ConfigRun(config_id="C8b30", records=self._records()),
        }
        verdict, detail = self._arm_verdict(runs, "noctx")
        self.assertEqual(verdict, "pass")
        self.assertIn("inert without context", detail)

    def test_noctx_copy_rise_still_fails(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C8b30": ConfigRun(config_id="C8b30", records=self._records(copy=True)),
        }
        verdict, _ = self._arm_verdict(runs, "noctx")
        self.assertEqual(verdict, "fail")

    def test_rows_are_downgraded_to_two_modes(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C8b30": ConfigRun(config_id="C8b30", records=self._records()),
        }
        rows = {
            row[0]: row
            for row in evaluate_gates(runs, load_thresholds(), None, "noctx", None, None)
        }
        self.assertEqual(rows["pool_composition[C8b30]"][1], "insufficient data")
        self.assertIn("requires both context modes", rows["pool_composition[C8b30]"][2])


class TestSelectionWindowGate(unittest.TestCase):
    """M3R-100 gate: coverage is the drop of the single-trajectory share, the
    window escape is must-improve, affinity is must-not-worsen."""

    @staticmethod
    def _records(
        *,
        n: int = 20,
        single: int = 10,
        escape_multi: int = 3,
        affinity: float = 0.3,
        copy: bool = False,
    ) -> list[GenRecord]:
        return [
            _record(
                window_escape=1 if index < single else escape_multi,
                pool_ecb=5,
                affinity=affinity,
                is_copy=copy,
            )
            for index in range(n)
        ]

    @staticmethod
    def _arm_verdict(runs: dict[str, ConfigRun], mode: str) -> tuple[str, str]:
        from tools.eval.report import _selection_arm_verdict

        return _selection_arm_verdict(runs["C0"], runs["C9m50"], load_thresholds(), None, mode)

    def test_mean_rise_without_coverage_is_insufficient(self) -> None:
        # Same share of single-trajectory inputs; the multi ones got wider.
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(escape_multi=2)),
            "C9m50": ConfigRun(config_id="C9m50", records=self._records(escape_multi=4)),
        }
        verdict, detail = self._arm_verdict(runs, "ctx")
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("coverage below the floor", detail)

    def test_widened_window_without_round_is_insufficient(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(single=10)),
            "C9m50": ConfigRun(config_id="C9m50", records=self._records(single=4)),
        }
        verdict, detail = self._arm_verdict(runs, "ctx")
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("single_trajectory_share Δ -0.300", detail)
        self.assertIn("connectedness round", detail)

    def test_widened_window_passes_in_noctx(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(single=10)),
            "C9m50": ConfigRun(config_id="C9m50", records=self._records(single=4)),
        }
        verdict, _ = self._arm_verdict(runs, "noctx")
        self.assertEqual(verdict, "pass")

    def test_topicality_price_fails(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(single=10, affinity=0.5)),
            "C9m50": ConfigRun(
                config_id="C9m50", records=self._records(single=4, affinity=0.2)
            ),
        }
        verdict, detail = self._arm_verdict(runs, "ctx")
        self.assertEqual(verdict, "fail")
        self.assertIn("affinity without copies dropped", detail)

    def test_copy_price_fails(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(single=10)),
            "C9m50": ConfigRun(config_id="C9m50", records=self._records(single=4, copy=True)),
        }
        verdict, detail = self._arm_verdict(runs, "noctx")
        self.assertEqual(verdict, "fail")
        self.assertIn("copy rose significantly", detail)

    def test_rows_are_downgraded_to_two_modes(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(single=10)),
            "C9m50": ConfigRun(config_id="C9m50", records=self._records(single=4)),
        }
        rows = {
            row[0]: row
            for row in evaluate_gates(runs, load_thresholds(), None, "noctx", None, None)
        }
        self.assertEqual(rows["selection_window[C9m50]"][1], "insufficient data")
        self.assertIn("requires both context modes", rows["selection_window[C9m50]"][2])


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


class TestTwoModeRequirement(unittest.TestCase):
    """M3R-140: a gate declared two-mode is not passable on one mode."""

    def test_declared_gate_is_downgraded_and_says_why(self) -> None:
        row = _apply_mode_requirement(
            ("phase9_interp[C9a]", "pass", "six conditions met"),
            load_thresholds(),
            "ctx",
        )
        self.assertEqual(row[1], "insufficient data")
        self.assertIn("requires both context modes", row[2])
        self.assertIn("noctx not measured", row[2])
        self.assertIn("six conditions met", row[2])  # numbers survive

    def test_undeclared_gate_is_left_alone(self) -> None:
        """Закрытые фазы не переоткрываются задним числом: правило действует
        только на гейты, которые сами его объявили."""
        row = ("phase2_entropy[C1]", "fail", "distinct-2 delta insignificant")
        self.assertEqual(
            _apply_mode_requirement(row, load_thresholds(), "ctx"), row
        )

    def test_phase9_declares_it(self) -> None:
        self.assertTrue(
            load_thresholds()["phase9_interp"].get("requires_both_modes")
        )

    def test_phase5_declares_it(self) -> None:
        # Owner decision 2026-09-01, before the M2R-430 run: the seeded channel
        # acts in both modes, so its promotion may not pass on ctx alone.
        self.assertTrue(
            load_thresholds()["phase5_promotion"].get("requires_both_modes")
        )


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
            self.assertIn("mode=ctx", report)  # M3R-140: the mode is named
            # M3R-103 (reporting half): every successful record carries the
            # winner's route and one route per pool candidate, as attributed
            # by the generator; the report prints the per-route table.
            self.assertIn("## Per-route breakdown", report)
            routes = {str(route) for route in CandidateRoute}
            successful = [r for r in first["C0"].records if r.success]
            self.assertTrue(successful)
            for record in successful:
                self.assertIn(record.winner_route, routes)
                self.assertEqual(len(record.pool_routes), record.pool_size)
                self.assertTrue(set(record.pool_routes) <= routes)
            self.assertIn("route_vanilla_attempts", first["C0"].telemetry[0])
            # M3R-110 (D2): extension and mutation no longer lose the walk's
            # start source — every successful winner resolves to an attempt.
            self.assertTrue(all(r.start_source is not None for r in successful))
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    async def test_noctx_mode_runs_without_context_and_stays_paired(self) -> None:
        """M3R-140: same prompts, same order, no context tokens.

        The pairing is the point: if the two modes drew different prompts, a
        ctx-vs-noctx delta would be a delta of prompt sets, and the production
        weighting the roadmap asks for would be meaningless.
        """
        db_path, temp_dir = await build_synthetic_snapshot(messages=80)
        try:
            prompt_set = generate_prompts(
                db_path, chat_id=SYNTHETIC_CHAT_ID, seed=42, snapshot_label="t"
            )
            configs = {"C0": load_matrix()["C0"]}
            kwargs: dict[str, Any] = dict(
                db_source=db_path,
                chat_id=SYNTHETIC_CHAT_ID,
                configs=configs,
                prompt_set=prompt_set,
                seeds=(42,),
                generations=16,
            )
            ctx_runs, _ = await run_matrix(**kwargs)
            noctx_runs, _ = await run_matrix(**kwargs, context_mode="noctx")

            ctx_records = ctx_runs["C0"].records
            noctx_records = noctx_runs["C0"].records
            self.assertEqual(
                [r.prompt_content for r in ctx_records],
                [r.prompt_content for r in noctx_records],
            )
            # Not a pairing check but the mode's own signature: with no context
            # supplied, no reply can be an exact copy of one.
            self.assertTrue(all(not r.is_copy for r in noctx_records))
            self.assertEqual(
                noctx_runs["C0"].telemetry[0]["ctx_generation_share"], 0.0
            )
            self.assertEqual(ctx_runs["C0"].telemetry[0]["ctx_generation_share"], 1.0)
            # M3R-145: the L1 seed draw exists in noctx only — the pipeline
            # never seeds addressed replies. In ctx nothing is drawn and the
            # draw counter stays at zero; in noctx the roll is taken, and on
            # the synthetic snapshot (hot selection empty at the defaults) the
            # draws that happened all came back empty.
            self.assertTrue(all(not r.seed_drawn for r in ctx_records))
            self.assertEqual(ctx_runs["C0"].telemetry[0]["hot_ngram_draws"], 0)
            self.assertTrue(all(not r.seed_drawn for r in noctx_records))
            noctx_snapshot = noctx_runs["C0"].telemetry[0]
            if noctx_snapshot["hot_ngram_draws"]:
                self.assertEqual(noctx_snapshot["hot_ngram_empty_rate"], 1.0)

            summary = metrics_summary(noctx_runs, "noctx")
            self.assertEqual(summary["C0"]["context_mode"], "noctx")
            report = build_report(
                runs=noctx_runs,
                skipped=[],
                prompt_set=prompt_set,
                thresholds=load_thresholds(),
                snapshot_label="synthetic",
                seeds=(42,),
                generations=16,
                revision="test",
                date="2026-08-14",
                notes=[],
                context_mode="noctx",
            )
            self.assertIn("mode=noctx", report)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    async def test_structural_pair_is_collected_from_the_real_pool(self) -> None:
        """M3R-011: обе величины считаются из живого пула, а не подставляются.

        Проверяется соотношение, а не абсолют: окно — подмножество пула, значит
        различных траекторий в нём не может быть больше.
        """
        db_path, temp_dir = await build_synthetic_snapshot(messages=80)
        try:
            prompt_set = generate_prompts(
                db_path, chat_id=SYNTHETIC_CHAT_ID, seed=42, snapshot_label="t"
            )
            runs, _ = await run_matrix(
                db_source=db_path,
                chat_id=SYNTHETIC_CHAT_ID,
                configs={"C0": load_matrix()["C0"]},
                prompt_set=prompt_set,
                seeds=(42,),
                generations=16,
            )
            records = runs["C0"].records
            self.assertTrue(any(r.pool_ecb > 0 for r in records))
            for record in records:
                self.assertLessEqual(record.window_escape, record.pool_ecb)
                self.assertLessEqual(record.pool_ecb, max(record.pool_size, 1))
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    async def test_unknown_context_mode_is_refused(self) -> None:
        db_path, temp_dir = await build_synthetic_snapshot(messages=20)
        try:
            with self.assertRaises(ValueError):
                await run_config_seed(
                    db_source=db_path,
                    chat_id=SYNTHETIC_CHAT_ID,
                    overrides={},
                    prompt_set=generate_prompts(
                        db_path,
                        chat_id=SYNTHETIC_CHAT_ID,
                        seed=42,
                        snapshot_label="t",
                    ),
                    seed=42,
                    generations=4,
                    context_mode="both",
                )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()


# Keep Path import honest for tooling that trims unused imports.
_ = Path


class TestAssocPilot(unittest.TestCase):
    """assoc-route-pilot: presence gates the reading, an ECB drop or a copy rise
    reads `not viable`, and the vocabulary is never pass / fail."""

    @staticmethod
    def _records(*, n: int = 20, present: int = 0, copy: bool = False, ecb: int = 4):
        records = []
        for index in range(n):
            routes = ("assoc", "vanilla", "vanilla") if index < present else ("vanilla",) * 3
            records.append(
                _record(
                    category="topical",
                    is_copy=copy,
                    affinity=0.3,
                    pool_ecb=ecb,
                    window_escape=2,
                    pool_routes=routes,
                )
            )
        return records

    @staticmethod
    def _verdict(runs, mode="ctx"):
        from tools.eval.report import _assoc_pilot_verdict

        return _assoc_pilot_verdict(runs["C0"], runs["C10a40"], load_thresholds(), mode)

    def test_presence_below_floor_is_insufficient(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C10a40": ConfigRun(config_id="C10a40", records=self._records(present=1, copy=True)),
        }
        verdict, detail = self._verdict(runs)
        self.assertEqual(verdict, "insufficient data")
        self.assertIn("did not exercise", detail)

    def test_ecb_drop_is_not_viable(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records(ecb=5)),
            "C10a40": ConfigRun(config_id="C10a40", records=self._records(present=10, ecb=3)),
        }
        verdict, detail = self._verdict(runs)
        self.assertEqual(verdict, "not viable")
        self.assertIn("duplicate the walk", detail)

    def test_copy_rise_is_not_viable(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C10a40": ConfigRun(config_id="C10a40", records=self._records(present=10, copy=True)),
        }
        verdict, detail = self._verdict(runs)
        self.assertEqual(verdict, "not viable")
        self.assertIn("copy rose", detail)

    def test_clean_arm_is_viable_and_prints_the_four_questions(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C10a40": ConfigRun(config_id="C10a40", records=self._records(present=10)),
        }
        verdict, detail = self._verdict(runs)
        self.assertEqual(verdict, "viable")
        for label in ("Q1 builds", "Q2 pool ECB", "window escape", "p95", "Q4 affinity"):
            self.assertIn(label, detail)
        self.assertNotIn("pass", verdict)

    def test_gate_rows_downgrade_to_both_modes(self) -> None:
        runs = {
            "C0": ConfigRun(config_id="C0", records=self._records()),
            "C10a40": ConfigRun(config_id="C10a40", records=self._records(present=10)),
        }
        rows = evaluate_gates(runs, load_thresholds(), None, "ctx", None, None)
        row = {r[0]: (r[1], r[2]) for r in rows}["assoc_pilot[C10a40]"]
        self.assertEqual(row[0], "insufficient data")
        self.assertIn("requires both context modes", row[1])

    def test_no_arm_reports_insufficient(self) -> None:
        runs = {"C0": ConfigRun(config_id="C0", records=self._records())}
        rows = evaluate_gates(runs, load_thresholds(), None, "ctx", None, None)
        row = {r[0]: r[1] for r in rows}["assoc_pilot"]
        self.assertEqual(row, "insufficient data")
