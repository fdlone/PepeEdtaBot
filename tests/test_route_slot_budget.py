"""Маршруты берут слоты изнутри бюджета пула (O10, ветвь 2).

До этого изменения маршрут добавлял кандидатов **сверх** заполненного пула:
замер на прод-копии 2026-09-01 дал ровно 5 кандидатов на арме C0 и в среднем
6.40 при максимуме 7 на C4 (`markov_seeded_candidate_ratio = 0.3`). Абсолютный
порог `pool_ecb_min: 4.0` при таком пуле сравнивает счёт с плывущим
знаменателем — ровно на тех правках (новые маршруты), ради которых заведён.

Здесь закреплено обратное: пул не растёт, маршрут конкурирует за места, а
незанятые им слоты достаются основному обходу.
"""
from __future__ import annotations

import random
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.core import gen_trace_log
from app.core.response_generator import (
    ResponseGenerator,
    route_slot_budget,
)
from tests.test_response_generator import (
    _learning_service,
    _request,
    _runtime_state,
    _score,
    _traced_generator,
)


class TestRouteSlotBudget(unittest.TestCase):
    """Формула бюджета (design D2)."""

    def test_neutral_ratio_gives_no_slots(self) -> None:
        self.assertEqual(route_slot_budget(5, 0.0), 0)

    def test_share_of_the_pool_not_an_addition(self) -> None:
        # 0.3 от пяти — два слота ИЗ пяти, а не два сверх пяти.
        self.assertEqual(route_slot_budget(5, 0.3), 2)

    def test_tiny_ratio_still_gets_one_slot(self) -> None:
        # Включённая ручка обязана давать хотя бы одну попытку.
        self.assertEqual(route_slot_budget(5, 0.05), 1)

    def test_top_of_the_range_never_exceeds_half_the_pool(self) -> None:
        # Верх диапазона ручки (0.7) не уводит пул в маршрутное большинство.
        self.assertEqual(route_slot_budget(5, 0.7), 2)
        self.assertEqual(route_slot_budget(10, 0.7), 5)

    def test_single_slot_pool_leaves_nothing_to_compete_for(self) -> None:
        # Потолок побеждает минимум: маршрут не вправе выставить пул, в котором
        # нет ни одного кандидата основного обхода.
        self.assertEqual(route_slot_budget(1, 0.7), 0)


class PoolCompositionTestCase(unittest.IsolatedAsyncioTestCase):
    """Общая оснастка: пул перехватывается на отборе, как в свип-харнессе."""

    def setUp(self) -> None:
        self.captured: list[list[object]] = []
        saved = gen_trace_log.log_selection

        def _capture(_chat_id, candidates, **_kwargs):
            self.captured.append(list(candidates or ()))

        gen_trace_log.log_selection = _capture  # type: ignore[assignment]
        self.addCleanup(
            lambda: setattr(gen_trace_log, "log_selection", saved)
        )

    @staticmethod
    def _seeded_generator(*, seeded_tokens: list[str] | None) -> AsyncMock:
        """Генератор, чей обход даёт разные тексты, а seeded — заданный."""
        generator = _traced_generator()
        walk_texts = iter(
            f"обычный кандидат номер {index} тут" for index in range(50)
        )
        generator.generate_text = AsyncMock(
            side_effect=lambda *a, **k: next(walk_texts)
        )
        generator.rank_seeds = AsyncMock(
            return_value=[
                SimpleNamespace(token=f"якорь{index}", score=1.0)
                for index in range(5)
            ]
        )
        generator.generate_seeded_candidate = AsyncMock(
            side_effect=(
                None
                if seeded_tokens is None
                else lambda *a, **k: list(seeded_tokens)
            ),
            return_value=None if seeded_tokens is None else list(seeded_tokens),
        )
        return generator

    async def _pool(
        self, state: MagicMock, generator: AsyncMock, *, target: int = 5
    ) -> list[object]:
        learning_service = _learning_service()
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=state,
            scorer=MagicMock(return_value=_score(1.0)),
        )
        with patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        ):
            await response_generator.generate_with_result(
                _request(), rng=random.Random(11), candidate_target=target
            )
        self.assertTrue(self.captured, "отбор не состоялся — пул пуст")
        return self.captured[-1]


class TestPoolStopsGrowing(PoolCompositionTestCase):
    async def test_route_takes_slots_from_inside_the_budget(self) -> None:
        """Пул не превышает целевой размер, и seeded в нём есть.

        Проверено мутацией (2026-09-01): возврат вызова маршрутной ветки за
        цикл попыток (как было до O10) роняет этот тест по существу —
        «6 not less than or equal to 5» на этой фикстуре; на живом арме C4
        тем же механизмом пул доходил до 7.
        """
        state = _runtime_state()
        state.markov_seeded_candidate_ratio = 0.3
        generator = self._seeded_generator(
            seeded_tokens=["затравочный", "кандидат", "из", "четырёх"]
        )

        pool = await self._pool(state, generator, target=5)

        self.assertLessEqual(len(pool), 5)
        routes = [candidate.route for candidate in pool]
        self.assertIn("seeded", routes)
        self.assertIn("vanilla", routes)

    async def test_route_that_produced_nothing_gives_its_slots_back(
        self,
    ) -> None:
        # Резерв с маршрутом ПОСЛЕ цикла оставил бы пул недозаполненным; здесь
        # обход добирает всё (design D1, отвергнутая альтернатива «а»).
        state = _runtime_state()
        state.markov_seeded_candidate_ratio = 0.3
        generator = self._seeded_generator(seeded_tokens=None)

        pool = await self._pool(state, generator, target=5)

        self.assertEqual(len(pool), 5)
        self.assertEqual({candidate.route for candidate in pool}, {"vanilla"})

    async def test_route_at_top_of_range_cannot_crowd_out_the_walk(
        self,
    ) -> None:
        state = _runtime_state()
        state.markov_seeded_candidate_ratio = 0.7
        generator = self._seeded_generator(
            seeded_tokens=["затравочный", "кандидат", "из", "четырёх"]
        )

        pool = await self._pool(state, generator, target=5)

        self.assertLessEqual(len(pool), 5)
        self.assertIn("vanilla", [candidate.route for candidate in pool])

    async def test_degenerate_chain_still_leaves_a_walk_candidate(self) -> None:
        # branching_aware_target ужимает цель ниже уже занятых слотов: цикл
        # прервётся сразу после первого принятого кандидата обхода. Проверяем,
        # что это не падение и не пул без обхода (риск из design).
        state = _runtime_state()
        state.markov_seeded_candidate_ratio = 0.3
        state.markov_branching_degenerate_max = 10.0  # любая цепь «вырождена»
        state.markov_branching_candidate_floor = 1
        generator = self._seeded_generator(
            seeded_tokens=["затравочный", "кандидат", "из", "четырёх"]
        )
        async def _walk_with_branching(*_a: object, **_k: object):
            """Проходка с ненулевым ветвлением — вход branching_aware_target."""
            text = await generator.generate_text()
            return text, SimpleNamespace(
                markov_order_used=3, start_source="global", mean_branching=1.0
            )

        generator.generate_text_with_trace = AsyncMock(
            side_effect=_walk_with_branching
        )

        pool = await self._pool(state, generator, target=5)

        self.assertIn("vanilla", [candidate.route for candidate in pool])


class TestNeutralRatioIsUnchanged(PoolCompositionTestCase):
    async def test_zero_ratio_never_runs_the_route(self) -> None:
        state = _runtime_state()
        state.markov_seeded_candidate_ratio = 0.0
        generator = self._seeded_generator(
            seeded_tokens=["затравочный", "кандидат", "из", "четырёх"]
        )

        pool = await self._pool(state, generator, target=5)

        self.assertEqual(len(pool), 5)
        self.assertEqual({candidate.route for candidate in pool}, {"vanilla"})
        generator.rank_seeds.assert_not_awaited()
        generator.generate_seeded_candidate.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
