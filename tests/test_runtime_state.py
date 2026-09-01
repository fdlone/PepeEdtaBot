from __future__ import annotations

import unittest
from collections import deque

from app.config.registry import RUNTIME_FIELDS
from app.config.runtime_state import RuntimeState


def make_runtime_state(**overrides: object) -> RuntimeState:
    """Состояние с дефолтами реестра, кроме намеренных отклонений ниже.

    Собирается из ``RUNTIME_FIELDS``, а не перечислением полей руками.
    Рукописное зеркало реестра расходится с ним молча и роняет разом всех, кто
    ходит через фикстуру: на M2R-900 одна новая ручка дала 71 падение с
    ``TypeError: missing 1 required positional argument``, из которого не видно
    ни причины, ни того, что виноват реестр, а не логика (O6). Теперь новая
    ручка приезжает со своим дефолтом и не ломает ничего.

    Отклонения ниже перенесены из прежнего рукописного списка **дословно** —
    ровно затем, чтобы поведение существующих тестов не изменилось.
    Индивидуальных обоснований у них нигде не записано, поэтому трогать их
    следует по одному и с прогоном полного сьюта, а не пачкой.
    """
    base: dict[str, object] = {
        spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS
    }
    base.update(
        {
            # Не из реестра: живое состояние процесса, у ``RuntimeState`` без
            # дефолтов. Значения маленькие — на них и проверяется вытеснение.
            "runtime_state_ttl_sec": 10,
            "runtime_state_max_chats": 2,
            # Отклонения от дефолтов реестра, сохранённые дословно.
            "normalize_lower": False,
            "fuzzy_context_casefold": False,
            "context_start_affinity": 3.0,
            "context_jump_boost": 1.0,
            "context_anchor_splice_probability": 0.0,
            "markov_jump_probability": 0.04,
            "markov_entropy_pivot": 0.5,
            "order_mix_probability": 0.0,
            "slot_mutation_probability": 0.0,
            "verbatim_penalty_strength": 1.0,
            "verbatim_extension_share": 0.0,
            "recent_reply_penalty_strength": 1.0,
            "length_context_adaptation": 0.0,
            "hot_ngram_seed_chance": 0.05,
            "rare_event_chance": 0.005,
            "false_start_chance": 0.03,
            "mood_mention_heated_share": 0.0,
        }
    )
    base.update(overrides)
    return RuntimeState(**base)


class TestFixtureMirrorsTheRegistry(unittest.TestCase):
    """Реестр и dataclass обязаны соглашаться друг с другом.

    Изначально этот тест страховал рукописное зеркало ``RUNTIME_FIELDS`` внутри
    фикстуры: новая ручка, не дописанная в ``base``, роняла не один тест, а всех,
    кто ходит через ``make_runtime_state`` — на M2R-900 это дало 71 падение с
    ``TypeError: missing 1 required positional argument``, из которого не видно
    ни причины, ни того, что виноват реестр, а не логика.

    Зеркала больше нет — фикстура собирается из реестра (O6), — но тест не стал
    лишним: он сменил охраняемый инвариант. Теперь он ловит **расхождение
    реестра с ``RuntimeState``**. Ручка, добавленная в ``RUNTIME_FIELDS``, но не
    объявленная в dataclass, роняет саму сборку; объявленная, но выпавшая из
    реестра — не приезжает в состояние и видна как отсутствующий атрибут.
    Оба случая читаются здесь одним внятным падением.
    """

    def test_every_runtime_field_is_buildable_by_the_fixture(self) -> None:
        state = make_runtime_state()
        missing = [
            spec.name for spec in RUNTIME_FIELDS if not hasattr(state, spec.name)
        ]
        self.assertEqual(
            missing, [], "ручки реестра не собираются фикстурой — добавьте их в base"
        )


class TestChatOverrides(unittest.TestCase):
    def test_chat_without_overrides_gets_the_same_object(self) -> None:
        # Not merely equal — identical. This is what keeps the untouched path
        # byte-for-byte what it was before overlays existed.
        state = make_runtime_state()
        self.assertIs(state.effective(100), state)

    def test_override_applies_only_to_its_chat(self) -> None:
        state = make_runtime_state(reply_probability=0.08)
        state.set_override(100, "reply_probability", 0.5)

        self.assertEqual(state.effective(100).reply_probability, 0.5)
        self.assertEqual(state.effective(200).reply_probability, 0.08)
        self.assertEqual(state.reply_probability, 0.08)

    def test_live_state_is_shared_not_split(self) -> None:
        # Anti-repeat, mood and counters must keep accumulating on the one
        # true state no matter which view wrote them.
        state = make_runtime_state()
        state.set_override(100, "reply_probability", 0.5)
        view = state.effective(100)

        view.note_reply_sent(100, now=1000.0)
        view.recent_replies[100] = deque(["ответ"], maxlen=20)
        view.note_rare_event(100, "2026-08-05")

        self.assertEqual(state.last_reply_ts[100], 1000.0)
        self.assertEqual(list(state.recent_replies[100]), ["ответ"])
        self.assertEqual(state.rare_events_today[100], ("2026-08-05", 1))

    def test_setting_an_override_does_not_reset_live_state(self) -> None:
        state = make_runtime_state()
        state.note_reply_sent(100, now=1000.0)
        state.recent_replies[100] = deque(["ответ"], maxlen=20)

        state.set_override(100, "reply_probability", 0.5)
        state.clear_override(100, "reply_probability")

        self.assertEqual(state.last_reply_ts[100], 1000.0)
        self.assertEqual(list(state.recent_replies[100]), ["ответ"])

    def test_clear_override_returns_chat_to_global(self) -> None:
        state = make_runtime_state(reply_probability=0.08)
        state.set_override(100, "reply_probability", 0.5)

        self.assertTrue(state.clear_override(100, "reply_probability"))
        self.assertEqual(state.effective(100).reply_probability, 0.08)
        self.assertIs(state.effective(100), state)
        # Nothing left to clear the second time.
        self.assertFalse(state.clear_override(100, "reply_probability"))

    def test_forget_chat_drops_overrides(self) -> None:
        state = make_runtime_state()
        state.set_override(100, "reply_probability", 0.5)
        state.note_chat_activity(100, now=10.0)

        state.forget_chat(100)

        self.assertEqual(state.chat_overrides, {})

    def test_inactive_chat_loses_overrides_with_the_rest_of_its_state(self) -> None:
        state = make_runtime_state()
        state.set_override(100, "reply_probability", 0.5)
        state.note_chat_activity(100, now=10.0)

        state.prune_inactive(now=21.0)

        self.assertEqual(state.chat_overrides, {})
        self.assertIs(state.effective(100), state)


class TestOverlayShareLiveState(unittest.TestCase):
    """Представление чата не должно уносить с собой служебное состояние.

    ``effective`` возвращает поверхностную копию, и инвариант «живое состояние
    у всех представлений одно» держался только для словарей. Единственный
    изменяемый скаляр — счётчик, по которому запускается уборка, — жил на
    копии-однодневке, и до базового состояния его инкременты не доходили: для
    чатов с переопределениями периодическая уборка по сроку не запускалась
    вовсе, а границу держал только предел по числу чатов.
    """

    def test_activity_through_a_view_advances_the_shared_counter(self) -> None:
        state = make_runtime_state()
        state.set_override(100, "reply_probability", 0.5)

        for tick in range(10):
            # Новое представление на каждое обновление — как в middleware.
            state.effective(100).note_chat_activity(100, now=float(tick))

        self.assertEqual(state._cleanup.value, 10)

    def test_ttl_eviction_fires_for_a_chat_with_overrides(self) -> None:
        state = make_runtime_state(runtime_state_ttl_sec=10)
        state.set_override(100, "reply_probability", 0.5)
        state.last_reply_ts[100] = 1.0
        state.effective(100).note_chat_activity(100, now=1.0)

        # 64 обращения — порог запуска уборки; к этому моменту чат 100 давно
        # неактивен по сроку.
        for tick in range(64):
            state.effective(200).note_chat_activity(200, now=100.0 + tick)

        self.assertNotIn(100, state.last_reply_ts)
        self.assertEqual(state.chat_overrides, {})

    def test_chat_without_overrides_still_gets_the_same_object(self) -> None:
        state = make_runtime_state()

        self.assertIs(state.effective(100), state)

        state.effective(100).note_chat_activity(100, now=1.0)
        self.assertEqual(state._cleanup.value, 1)


class TestRuntimeState(unittest.TestCase):
    def test_prune_inactive_removes_stale_chat_entries(self) -> None:
        state = make_runtime_state()
        state.last_reply_ts[100] = 1.0
        state.learned_messages[100] = 4
        state.recent_short_replies[100] = deque(["hi"], maxlen=5)
        state.recent_replies[100] = deque(["длинный недавний ответ"], maxlen=20)
        state.note_chat_activity(100, now=10.0)

        state.prune_inactive(now=21.0)

        self.assertNotIn(100, state.last_reply_ts)
        self.assertNotIn(100, state.learned_messages)
        self.assertNotIn(100, state.recent_short_replies)
        self.assertNotIn(100, state.recent_replies)

    def test_prune_inactive_keeps_recent_chat_entries(self) -> None:
        state = make_runtime_state()
        state.last_reply_ts[100] = 1.0
        state.learned_messages[100] = 4
        state.note_chat_activity(100, now=15.0)

        state.prune_inactive(now=20.0)

        self.assertIn(100, state.last_reply_ts)
        self.assertIn(100, state.learned_messages)

    def test_note_chat_activity_evicts_oldest_when_capacity_exceeded(self) -> None:
        state = make_runtime_state(runtime_state_max_chats=2)
        state.last_reply_ts[1] = 1.0
        state.last_reply_ts[2] = 2.0
        state.last_reply_ts[3] = 3.0

        state.note_chat_activity(1, now=1.0)
        state.note_chat_activity(2, now=2.0)
        state.note_chat_activity(3, now=3.0)
        state.prune_inactive(now=3.0)

        self.assertNotIn(1, state.last_reply_ts)
        self.assertIn(2, state.last_reply_ts)
        self.assertIn(3, state.last_reply_ts)

    def test_forget_chat_clears_all_runtime_maps(self) -> None:
        state = make_runtime_state()
        state.last_reply_ts[100] = 1.0
        state.learned_messages[100] = 4
        state.recent_short_replies[100] = deque(["hi"], maxlen=5)
        state.recent_replies[100] = deque(["длинный недавний ответ"], maxlen=20)
        state.recent_reply_times[100] = deque([1.0])
        state.note_rare_event(100, "2026-07-04")
        state.note_user_quirk(100, 1001, "2026-07-04")
        state.note_user_quirk(200, 1001, "2026-07-04")
        state.note_chat_activity(100, now=10.0)

        state.forget_chat(100)

        self.assertEqual(state.last_reply_ts, {})
        self.assertEqual(state.learned_messages, {})
        self.assertEqual(state.recent_short_replies, {})
        self.assertEqual(state.recent_replies, {})
        self.assertEqual(state.recent_reply_times, {})
        self.assertEqual(state.rare_events_today, {})
        # Only the forgotten chat's quirk stamps are swept.
        self.assertEqual(state.last_user_quirk_day, {(200, 1001): "2026-07-04"})

    def test_rare_event_cap_counts_per_day(self) -> None:
        state = make_runtime_state()
        self.assertTrue(state.can_fire_rare_event(1, "2026-07-04"))
        for _ in range(state.rare_event_daily_cap):
            state.note_rare_event(1, "2026-07-04")
        self.assertFalse(state.can_fire_rare_event(1, "2026-07-04"))
        # New day resets the counter.
        self.assertTrue(state.can_fire_rare_event(1, "2026-07-05"))
        state.note_rare_event(1, "2026-07-05")
        self.assertEqual(state.rare_events_today[1], ("2026-07-05", 1))

    def test_rare_event_zero_cap_never_fires(self) -> None:
        state = make_runtime_state(rare_event_daily_cap=0)
        self.assertFalse(state.can_fire_rare_event(1, "2026-07-04"))
        self.assertFalse(state.can_fire_rare_event(1, "2026-07-05"))

    def test_user_quirk_gate_is_per_user_per_utc_day(self) -> None:
        state = make_runtime_state()
        self.assertTrue(state.can_fire_user_quirk(1, 1001, "2026-07-13"))

        state.note_user_quirk(1, 1001, "2026-07-13")

        # Same user, same day: suppressed. Other user / other chat: allowed.
        self.assertFalse(state.can_fire_user_quirk(1, 1001, "2026-07-13"))
        self.assertTrue(state.can_fire_user_quirk(1, 2002, "2026-07-13"))
        self.assertTrue(state.can_fire_user_quirk(2, 1001, "2026-07-13"))
        # Next day: allowed again.
        self.assertTrue(state.can_fire_user_quirk(1, 1001, "2026-07-14"))

    def test_note_reply_sent_updates_last_ts_and_history(self) -> None:
        state = make_runtime_state()
        state.note_reply_sent(100, now=1000.0)
        state.note_reply_sent(100, now=1050.0)

        self.assertEqual(state.last_reply_ts[100], 1050.0)
        self.assertEqual(list(state.recent_reply_times[100]), [1000.0, 1050.0])

    def test_note_reply_sent_trims_history_older_than_one_hour(self) -> None:
        state = make_runtime_state()
        state.note_reply_sent(100, now=1000.0)
        # Advance more than an hour: the first timestamp falls out of the window.
        state.note_reply_sent(100, now=1000.0 + 3600.0 + 1.0)

        self.assertEqual(list(state.recent_reply_times[100]), [4601.0])

    def test_note_reply_sent_prompted_reply_skips_hourly_history(self) -> None:
        state = make_runtime_state()
        # A mention answer updates cooldown/burst but must not count against the
        # per-hour cap.
        state.note_reply_sent(100, now=1000.0, unprompted=False)

        self.assertEqual(state.last_reply_ts[100], 1000.0)
        self.assertNotIn(100, state.recent_reply_times)

    def test_note_reply_sent_mixes_prompted_and_unprompted(self) -> None:
        state = make_runtime_state()
        state.note_reply_sent(100, now=1000.0, unprompted=True)
        state.note_reply_sent(100, now=1050.0, unprompted=False)
        state.note_reply_sent(100, now=1100.0, unprompted=True)

        self.assertEqual(state.last_reply_ts[100], 1100.0)
        self.assertEqual(list(state.recent_reply_times[100]), [1000.0, 1100.0])


class TestReplySlotReservation(unittest.TestCase):
    """Бюджет ответа занимается в момент решения и возвращается, если ответа
    не было (O8)."""

    def test_reservation_occupies_the_budget_immediately(self) -> None:
        state = make_runtime_state()

        state.reserve_reply_slot(100, 1000.0)

        self.assertEqual(state.last_reply_ts[100], 1000.0)
        self.assertEqual(list(state.recent_reply_times[100]), [1000.0])

    def test_release_returns_the_budget(self) -> None:
        state = make_runtime_state()

        slot = state.reserve_reply_slot(100, 1000.0)
        state.release_reply_slot(slot)

        self.assertNotIn(100, state.last_reply_ts)
        self.assertNotIn(100, state.recent_reply_times)

    def test_release_restores_the_previous_cooldown(self) -> None:
        state = make_runtime_state()
        state.note_reply_sent(100, 900.0)

        slot = state.reserve_reply_slot(100, 1000.0)
        state.release_reply_slot(slot)

        self.assertEqual(state.last_reply_ts[100], 900.0)
        self.assertEqual(list(state.recent_reply_times[100]), [900.0])

    def test_release_removes_its_own_mark_not_the_last_one(self) -> None:
        """Между резервацией и откатом другой апдейт мог занять слот законно.

        Его резервация отменяться не должна: она сделана по собственной
        проверке и в силе. Поэтому откат снимает свою метку, а не последнюю,
        и не трогает кулдаун, если тот уже принадлежит не ему.
        """
        state = make_runtime_state()

        first = state.reserve_reply_slot(100, 1000.0)
        state.reserve_reply_slot(100, 1001.0)
        state.release_reply_slot(first)

        self.assertEqual(
            list(state.recent_reply_times[100]),
            [1001.0],
            "откат снял чужую метку",
        )
        self.assertEqual(
            state.last_reply_ts[100], 1001.0, "откат сбросил чужой кулдаун"
        )

    def test_mention_reservation_stays_out_of_the_hourly_cap(self) -> None:
        state = make_runtime_state()

        state.reserve_reply_slot(100, 1000.0, unprompted=False)

        self.assertEqual(state.last_reply_ts[100], 1000.0)
        self.assertNotIn(100, state.recent_reply_times)

    def test_release_is_safe_when_the_mark_already_aged_out(self) -> None:
        """Часовой срез мог вытеснить метку до отката — возвращать нечего."""
        state = make_runtime_state()

        slot = state.reserve_reply_slot(100, 1000.0)
        state.note_reply_sent(100, 1000.0 + 3601.0)

        state.release_reply_slot(slot)  # не должно бросать

        self.assertEqual(list(state.recent_reply_times[100]), [4601.0])


if __name__ == "__main__":
    unittest.main()
