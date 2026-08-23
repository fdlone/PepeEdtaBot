"""Воронка канала причуд L2: какой гейт его душит (observe-user-quirk-channel).

Канал живёт с 2026-07-16 и ни разу не был виден в чате, а из пяти условий
подряд наблюдался ровно один — порог. Свойство, которое здесь закрепляется:
каждый отказ засчитан **своему** гейту, у каждого гейта есть знаменатель, и
учёт не стоит ни одного розыгрыша — иначе он сдвинул бы поведение бота.
"""

from __future__ import annotations

import random
import unittest
from unittest.mock import AsyncMock, patch

from app.config.runtime_state import RuntimeState
from app.core.generation_telemetry import GenerationTelemetry, UserQuirkGate
from app.core.mood import NEUTRAL_MODIFIERS
from app.services.reply_pipeline import (
    IncomingMessage,
    MessageObservation,
    ReplyPipeline,
)
from tests.test_reply_pipeline import _incoming, _state

TODAY = "2026-08-23"


def _observation(*, address_reply: bool) -> MessageObservation:
    return MessageObservation(
        address_reply=address_reply,
        mood=None,
        mood_modifiers=NEUTRAL_MODIFIERS,
        chat_rhythm=None,
        learn_source="текст",
        tokens=["текст"],
        learnable=True,
        token_volume=10_000,
        enough_data=True,
        current_message_normalized="текст",
    )


def _legacy_decision(
    *,
    address_reply: bool,
    chance: float,
    can_fire: bool,
    interactions: int,
    threshold: int,
) -> bool:
    """Дословная копия условия ДО разложения на гейты.

    Оракул нужен именно в такой форме: правка обязана быть тождественной по
    решениям и по розыгрышам, а сравнивать её не с чем, если прежний код
    существует только в истории.
    """
    if not (
        address_reply and chance > 0.0 and can_fire and random.random() < chance
    ):
        return False
    return interactions >= threshold


class UserQuirkFunnelTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.learning_service = AsyncMock()
        self.learning_service.get_user_interaction_count.return_value = 0
        self.telemetry = GenerationTelemetry()
        self.generator = AsyncMock()
        self.generator.telemetry = self.telemetry
        self.masking = patch(
            "app.services.reply_pipeline.mask_chat_id", return_value="chat"
        )
        self.masking.start()
        self.addCleanup(self.masking.stop)

    def _pipeline(self, state: RuntimeState) -> ReplyPipeline:
        return ReplyPipeline(
            learning_service=self.learning_service,
            generator=self.generator,
            runtime_state=state,
        )

    async def _apply(
        self,
        state: RuntimeState,
        *,
        address_reply: bool = True,
        msg: IncomingMessage | None = None,
    ) -> list[str] | None:
        pipeline = self._pipeline(state)
        return await pipeline._apply_user_quirk(
            msg or _incoming(mentioned=True),
            _observation(address_reply=address_reply),
            "ответ по делу",
            TODAY,
        )

    @staticmethod
    def _quirk_state(**overrides: object) -> RuntimeState:
        defaults: dict[str, object] = dict(
            user_quirk_chance=1.0,
            user_quirk_min_interactions=1,
            user_quirk_name_share=0.0,
        )
        defaults.update(overrides)
        return _state(**defaults)


class TestEachGateIsCountedSeparately(UserQuirkFunnelTestCase):
    """2.4: отказ засчитывается тому гейту, который сработал, и только ему."""

    def _assert_only(self, gate: UserQuirkGate | None) -> None:
        rejected = {
            key: value for key, value in self.telemetry.quirk_rejected.items() if value
        }
        expected = {} if gate is None else {gate: 1}
        self.assertEqual(rejected, expected)
        self.assertEqual(self.telemetry.quirk_reached, 1)
        self.assertEqual(self.telemetry.quirk_fired, 1 if gate is None else 0)

    async def test_unaddressed_reply_stops_at_the_first_gate(self) -> None:
        await self._apply(self._quirk_state(), address_reply=False)
        self._assert_only(UserQuirkGate.ADDRESSED)

    async def test_zero_chance_stops_at_the_knob(self) -> None:
        await self._apply(self._quirk_state(user_quirk_chance=0.0))
        self._assert_only(UserQuirkGate.CHANCE)

    async def test_spent_daily_budget_stops_at_the_limit(self) -> None:
        state = self._quirk_state()
        state.note_user_quirk(1, 5, TODAY)
        await self._apply(state)
        self._assert_only(UserQuirkGate.DAILY_LIMIT)

    async def test_lost_roll_stops_at_the_draw(self) -> None:
        # Ролл проигран по построению: при чансе 1e-9 розыгрыш практически
        # никогда не проходит, но ручка ненулевая — предыдущий гейт пройден.
        random.seed(0)
        await self._apply(self._quirk_state(user_quirk_chance=1e-9))
        self._assert_only(UserQuirkGate.ROLL)

    async def test_short_counter_stops_at_the_threshold(self) -> None:
        self.learning_service.get_user_interaction_count.return_value = 0
        await self._apply(self._quirk_state(user_quirk_min_interactions=25))
        self._assert_only(UserQuirkGate.THRESHOLD)

    async def test_passing_every_gate_counts_as_fired(self) -> None:
        self.learning_service.get_user_interaction_count.return_value = 50
        parts = await self._apply(self._quirk_state())
        self.assertIsNotNone(parts)
        self._assert_only(None)


class TestFunnelArithmetic(UserQuirkFunnelTestCase):
    """2.2: знаменатель гейта = числитель предыдущего минус его отказы."""

    async def test_denominators_chain_down_to_the_fired_count(self) -> None:
        self.learning_service.get_user_interaction_count.return_value = 0
        await self._apply(self._quirk_state(), address_reply=False)
        await self._apply(self._quirk_state(user_quirk_chance=0.0))
        await self._apply(self._quirk_state(user_quirk_min_interactions=25))
        self.learning_service.get_user_interaction_count.return_value = 50
        await self._apply(self._quirk_state())

        snap = self.telemetry.snapshot()
        reached = snap["user_quirk_reached"]
        self.assertEqual(reached, 4)
        for gate in UserQuirkGate:
            self.assertEqual(snap[f"user_quirk_reached_{gate}"], reached)
            reached -= snap[f"user_quirk_rejected_{gate}"]  # type: ignore[operator]
        self.assertEqual(reached, snap["user_quirk_fired"])
        self.assertEqual(snap["user_quirk_fired"], 1)

    def test_untouched_channel_reports_zeros_rather_than_absence(self) -> None:
        # 2.1: нейтральная ручка обязана дать нули в обоих числах пары, а не
        # отсутствие полей — иначе «не спрашивали» неотличимо от «спрашивали».
        snap = GenerationTelemetry().snapshot()
        for gate in UserQuirkGate:
            self.assertEqual(snap[f"user_quirk_reached_{gate}"], 0)
            self.assertEqual(snap[f"user_quirk_rejected_{gate}"], 0)
        self.assertEqual(snap["user_quirk_fired"], 0)

    async def test_each_gate_is_counted_independently(self) -> None:
        # 2.1: два разных отказа не сливаются в один счётчик.
        await self._apply(self._quirk_state(), address_reply=False)
        await self._apply(self._quirk_state(user_quirk_chance=0.0))

        snap = self.telemetry.snapshot()
        self.assertEqual(snap["user_quirk_rejected_addressed"], 1)
        self.assertEqual(snap["user_quirk_rejected_chance"], 1)
        self.assertEqual(snap["user_quirk_reached_chance"], 1)


class TestAccountingIsSilent(UserQuirkFunnelTestCase):
    """3.2: учёт ничего не пишет в логи — там нечему быть замаскированным."""

    async def test_a_rejected_gate_logs_nothing(self) -> None:
        with self.assertNoLogs("app.services.reply_pipeline"):
            await self._apply(self._quirk_state(), address_reply=False)
            await self._apply(self._quirk_state(user_quirk_chance=0.0))
            await self._apply(self._quirk_state(user_quirk_min_interactions=25))


class TestAccountingIsFreeOfRandomness(UserQuirkFunnelTestCase):
    """2.5 и 2.3: ни одного лишнего розыгрыша, решения прежние."""

    async def _draws(self, state: RuntimeState, *, address_reply: bool) -> int:
        with patch(
            "app.services.reply_pipeline.random.random", wraps=random.random
        ) as draw:
            await self._apply(state, address_reply=address_reply)
        return int(draw.call_count)

    async def test_gates_before_the_draw_spend_nothing(self) -> None:
        self.assertEqual(await self._draws(self._quirk_state(), address_reply=False), 0)
        self.assertEqual(
            await self._draws(
                self._quirk_state(user_quirk_chance=0.0), address_reply=True
            ),
            0,
        )
        spent = self._quirk_state()
        spent.note_user_quirk(1, 5, TODAY)
        self.assertEqual(await self._draws(spent, address_reply=True), 0)

    async def test_the_draw_is_taken_exactly_once(self) -> None:
        self.learning_service.get_user_interaction_count.return_value = 50
        self.assertEqual(await self._draws(self._quirk_state(), address_reply=True), 1)

    async def test_decisions_match_the_pre_change_condition(self) -> None:
        """2.3: на серии входов при одном зерне решения совпадают с прежними."""
        cases = [
            (address_reply, chance, spent, interactions)
            for address_reply in (True, False)
            for chance in (0.0, 0.5, 1.0)
            for spent in (False, True)
            for interactions in (0, 25)
        ]
        for index, (address_reply, chance, spent, interactions) in enumerate(cases):
            with self.subTest(case=index):
                state = self._quirk_state(
                    user_quirk_chance=chance, user_quirk_min_interactions=25
                )
                if spent:
                    state.note_user_quirk(1, 5, TODAY)
                self.learning_service.get_user_interaction_count.return_value = (
                    interactions
                )

                random.seed(index)
                parts = await self._apply(state, address_reply=address_reply)

                random.seed(index)
                expected = _legacy_decision(
                    address_reply=address_reply,
                    chance=chance,
                    can_fire=not spent,
                    interactions=interactions,
                    threshold=25,
                )
                self.assertEqual(parts is not None, expected)


if __name__ == "__main__":
    unittest.main()
