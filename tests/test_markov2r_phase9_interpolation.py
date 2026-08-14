"""Phase 9: soft interpolation of the step distribution with order-2 (M2R-900..902).

The phase exists because 97.9% of order-3 states on this corpus offer exactly one
continuation (94.6% of visits), which is why entropy sampling (Phase 2) and the
temporal blend (Phase 3) both measured inert — they reweight candidates a state
already has. This merge ADDS candidates, so the invariants that matter are:
the disabled path must not merely be close but byte-identical; the merged order
must be deterministic because it feeds the RNG; and a projection that offers
nothing must degenerate rather than fail.
"""

from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.interpolation import OrderInterpolation
from app.core.markov import MarkovGenerator, tokenize
from app.core.temporal import TransitionRow
from app.infrastructure.database import Database

CHAT = -1001234567890


def _rows(*pairs: tuple[str, int]) -> list[TransitionRow]:
    """Transition rows in the token order the sources guarantee."""
    return [(token, count, 0.0, None) for token, count in sorted(pairs)]


class TestMergeArithmetic(unittest.TestCase):
    def test_weights_are_the_convex_combination(self) -> None:
        merged = OrderInterpolation(0.3).merge(
            _rows(("a", 10)), _rows(("a", 5), ("b", 5))
        )
        assert merged is not None
        self.assertEqual([row[0] for row in merged.rows], ["a", "b"])
        # a: 0.7*1.0 + 0.3*0.5 ; b: 0.3*0.5
        self.assertAlmostEqual(merged.weights[0], 0.85)
        self.assertAlmostEqual(merged.weights[1], 0.15)

    def test_result_is_a_distribution(self) -> None:
        merged = OrderInterpolation(0.5).merge(
            _rows(("a", 3), ("b", 1)), _rows(("a", 1), ("c", 9))
        )
        assert merged is not None
        self.assertAlmostEqual(sum(merged.weights), 1.0)

    def test_layers_are_normalized_before_merging_not_after(self) -> None:
        """Счётчики слоёв несопоставимы — вес определяет мнение, не объём.

        Слой order-2 здесь на три порядка больше по счётчикам. Если бы слияние
        шло по сырым счётчикам, он бы забрал почти всю массу независимо от β.
        """
        merged = OrderInterpolation(0.1).merge(
            _rows(("a", 2)), _rows(("a", 1000), ("b", 1000))
        )
        assert merged is not None
        weight_by_token = dict(
            zip([row[0] for row in merged.rows], merged.weights, strict=True)
        )
        self.assertAlmostEqual(weight_by_token["a"], 0.9 + 0.1 * 0.5)
        self.assertAlmostEqual(weight_by_token["b"], 0.1 * 0.5)

    def test_beta_one_is_the_projection_alone(self) -> None:
        merged = OrderInterpolation(1.0).merge(
            _rows(("a", 10)), _rows(("b", 3), ("c", 1))
        )
        assert merged is not None
        weight_by_token = dict(
            zip([row[0] for row in merged.rows], merged.weights, strict=True)
        )
        self.assertAlmostEqual(weight_by_token["a"], 0.0)
        self.assertAlmostEqual(weight_by_token["b"], 0.75)
        self.assertAlmostEqual(weight_by_token["c"], 0.25)

    def test_shared_token_appears_once(self) -> None:
        merged = OrderInterpolation(0.5).merge(
            _rows(("a", 1), ("b", 1)), _rows(("a", 1), ("b", 1), ("c", 1))
        )
        assert merged is not None
        tokens = [row[0] for row in merged.rows]
        self.assertEqual(tokens, ["a", "b", "c"])
        self.assertEqual(len(tokens), len(set(tokens)))

    def test_shared_token_keeps_the_order3_row(self) -> None:
        """У общего токена остаётся строка состояния, а не проекции."""
        merged = OrderInterpolation(0.5).merge(
            [("a", 42, 0.0, None)], [("a", 7, 0.0, None), ("b", 1, 0.0, None)]
        )
        assert merged is not None
        self.assertEqual(merged.rows[0], ("a", 42, 0.0, None))


class TestOrderAndDeterminism(unittest.TestCase):
    def test_merged_order_is_by_token(self) -> None:
        """Тот же порядок, которого сэмплер уже требует от строк переходов.

        Второй, конфликтующий порядок на том же пути дал бы расхождение,
        которое потом диагностируется как дрейф поведения.
        """
        merged = OrderInterpolation(0.4).merge(
            _rows(("яблоко", 1)), _rows(("арбуз", 1), ("банан", 1))
        )
        assert merged is not None
        tokens = [row[0] for row in merged.rows]
        self.assertEqual(tokens, sorted(tokens))

    def test_repeated_merge_is_identical(self) -> None:
        pool3, pool2 = _rows(("b", 2), ("d", 1)), _rows(("a", 3), ("b", 1), ("c", 2))
        first = OrderInterpolation(0.35).merge(pool3, pool2)
        second = OrderInterpolation(0.35).merge(pool3, pool2)
        assert first is not None and second is not None
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.weights, second.weights)


class TestDegenerateCases(unittest.TestCase):
    def test_disabled_returns_none(self) -> None:
        self.assertIsNone(
            OrderInterpolation(0.0).merge(_rows(("a", 1)), _rows(("b", 1)))
        )

    def test_empty_projection_degenerates(self) -> None:
        """Пустая проекция — вырождение в чистое P3, не ошибка и не пустой пул."""
        self.assertIsNone(OrderInterpolation(0.5).merge(_rows(("a", 1)), []))

    def test_empty_state_pool_degenerates(self) -> None:
        self.assertIsNone(OrderInterpolation(0.5).merge([], _rows(("a", 1))))

    def test_projection_adding_nothing_degenerates(self) -> None:
        """Проекция без новых токенов — это переваживание, а не расширение.

        Ровно то, что фазы 2 и 3 уже измерили инертным; вместо этого шаг
        остаётся на существующем пути с сырыми счётчиками.
        """
        self.assertIsNone(
            OrderInterpolation(0.5).merge(_rows(("a", 1)), _rows(("a", 9)))
        )


class TestTelemetryOfTheMerge(unittest.TestCase):
    def test_added_counts_only_new_tokens(self) -> None:
        merged = OrderInterpolation(0.5).merge(
            _rows(("a", 1)), _rows(("a", 1), ("b", 1), ("c", 1))
        )
        assert merged is not None
        self.assertEqual(merged.added, 2)

    def test_displacement_is_total_variation_from_pure_p3(self) -> None:
        merged = OrderInterpolation(0.3).merge(
            _rows(("a", 10)), _rows(("a", 5), ("b", 5))
        )
        assert merged is not None
        # |0.85-1.0| + |0.15-0.0| = 0.30 ; half of it is the TV distance
        self.assertAlmostEqual(merged.displacement, 0.15)

    def test_displacement_grows_with_beta(self) -> None:
        pool3, pool2 = _rows(("a", 10)), _rows(("a", 1), ("b", 1))
        low = OrderInterpolation(0.1).merge(pool3, pool2)
        high = OrderInterpolation(0.5).merge(pool3, pool2)
        assert low is not None and high is not None
        self.assertLess(low.displacement, high.displacement)


class TestNeutralityProperty(unittest.TestCase):
    @settings(max_examples=60, deadline=None)
    @given(
        st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=12),
        st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=12),
    )
    def test_zero_beta_never_merges(self, counts3: list[int], counts2: list[int]) -> None:
        """β = 0 не считает и не читает — возвращает None до любой арифметики.

        Проверяется именно ``None``, а не «веса совпали с P3»: совпадение весов
        означало бы, что арифметика всё-таки выполнилась, а инвариант фазы
        строже — раннего выхода, а не нулевого коэффициента.
        """
        pool3 = _rows(*[(f"t{i}", c) for i, c in enumerate(counts3)])
        pool2 = _rows(*[(f"u{i}", c) for i, c in enumerate(counts2)])
        self.assertIsNone(OrderInterpolation(0.0).merge(pool3, pool2))


class TestWalkIntegration(unittest.IsolatedAsyncioTestCase):
    """Ручка доходит до шага и обе политики кэша дают один вывод."""

    async def asyncSetUp(self) -> None:
        from app import log_masking

        log_masking.init_masking("phase9-interp")
        self.db_path = Path(f"test_p9_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        for text in (
            "красный дракон летит над городом ночью",
            "синий дракон летит под старым мостом днём",
            "зелёный кот сидит на тёплой крыше вечером",
            "жёлтый дракон летит сквозь холодный туман утром",
            "серый кот спит на широком подоконнике утром",
        ):
            await self.db.save_message_and_update_model(
                chat_id=CHAT, raw_text=text, tokens=tokenize(text)
            )

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)
        for suffix in ("-shm", "-wal"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)

    async def _walk(self, beta: float, *, seeds: range) -> list[str]:
        generator = MarkovGenerator(self.db.markov)
        return [
            await generator.generate_text(
                CHAT,
                max_chars=200,
                max_tokens=20,
                interpolation=OrderInterpolation(beta=beta),
                rng=random.Random(seed),
            )
            for seed in seeds
        ]

    async def test_beta_changes_the_walk(self) -> None:
        """Иначе вся проводка ручки декоративна и тесты сторожат пустоту."""
        neutral = await self._walk(0.0, seeds=range(16))
        interpolated = await self._walk(0.6, seeds=range(16))
        self.assertNotEqual(neutral, interpolated)

    async def test_repeated_run_at_same_beta_is_identical(self) -> None:
        self.assertEqual(
            await self._walk(0.3, seeds=range(12)),
            await self._walk(0.3, seeds=range(12)),
        )

    async def test_fresh_generator_matches_warm_one(self) -> None:
        """Прокси кэш-политик: холодный генератор против прогретого.

        Слои кэшируются раздельно и сливаются при чтении, поэтому слитое
        распределение не может разойтись между политиками — но утверждать это
        без проверки значило бы полагаться на замысел.
        """
        warm = MarkovGenerator(self.db.markov)
        # Прогреваем кэши слоёв полным проходом при том же β.
        for seed in range(8):
            await warm.generate_text(
                CHAT,
                max_chars=200,
                max_tokens=20,
                interpolation=OrderInterpolation(beta=0.3),
                rng=random.Random(seed),
            )
        warm_texts = [
            await warm.generate_text(
                CHAT,
                max_chars=200,
                max_tokens=20,
                interpolation=OrderInterpolation(beta=0.3),
                rng=random.Random(100 + seed),
            )
            for seed in range(8)
        ]
        cold = MarkovGenerator(self.db.markov)
        cold_texts = [
            await cold.generate_text(
                CHAT,
                max_chars=200,
                max_tokens=20,
                interpolation=OrderInterpolation(beta=0.3),
                rng=random.Random(100 + seed),
            )
            for seed in range(8)
        ]
        self.assertEqual(warm_texts, cold_texts)

    async def test_telemetry_reports_intent_and_effect(self) -> None:
        generator = MarkovGenerator(self.db.markov)
        for seed in range(12):
            await generator.generate_text(
                CHAT,
                max_chars=200,
                max_tokens=20,
                interpolation=OrderInterpolation(beta=0.5),
                rng=random.Random(seed),
            )
        snapshot = generator.telemetry.snapshot()
        self.assertIn("interp_step_coverage", snapshot)
        self.assertIn("mean_interp_displacement", snapshot)
        coverage = snapshot["interp_step_coverage"]
        assert coverage is not None
        self.assertGreater(coverage, 0.0)

    async def test_neutral_beta_leaves_telemetry_at_zero(self) -> None:
        """«Не выставлено» отличимо от «выставлено и ничего не добавило»."""
        generator = MarkovGenerator(self.db.markov)
        for seed in range(12):
            await generator.generate_text(
                CHAT, max_chars=200, max_tokens=20, rng=random.Random(seed)
            )
        snapshot = generator.telemetry.snapshot()
        self.assertEqual(snapshot["interp_step_coverage"], 0.0)
        self.assertEqual(snapshot["mean_interp_displacement"], 0.0)


if __name__ == "__main__":
    unittest.main()
