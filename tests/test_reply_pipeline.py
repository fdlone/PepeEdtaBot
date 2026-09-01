"""Ветки политики ответа — на уровне сервиса, без моков Telegram.

Раньше каждая из этих веток проверялась только полным прогоном обработчика:
чтобы дойти до причуды завсегдатая, тест собирал `Message`, `from_user`,
`bot`, `reply` и ждал, пока конвейер доберётся до нужной строки. Здесь
конвейер получает факты (`IncomingMessage`) и функцию отправки, поэтому
проверяется решение, а не сборка сообщения.
"""
from __future__ import annotations

import unittest
from collections import deque
from collections.abc import Sequence
from unittest.mock import AsyncMock, patch

from app.config.registry import RUNTIME_FIELDS
from app.config.runtime_state import RuntimeState
from app.core.generation_telemetry import GenerationTelemetry
from app.services.meme_analyzer import MemeSettings
from app.services.reply_pipeline import (
    IncomingMessage,
    PartialDeliveryError,
    ReplyPipeline,
)


def _state(**overrides: object) -> RuntimeState:
    """Реестровые дефолты — та же конфигурация, с которой бот запускается.

    Каналы, которые роллят кубик поверх решения (редкие события, фальстарт,
    причуды завсегдатаев), по умолчанию выключены: иначе почти каждый тест
    иногда получал бы лишнюю часть ответа. Проверяющие их тесты включают чанс
    явно.
    """
    state = RuntimeState(
        **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS},
        runtime_state_ttl_sec=86_400,
        runtime_state_max_chats=2048,
    )
    state.rare_event_chance = 0.0
    state.false_start_chance = 0.0
    state.user_quirk_chance = 0.0
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


def _incoming(**overrides: object) -> IncomingMessage:
    facts: dict[str, object] = dict(
        chat_id=1,
        user_id=5,
        first_name="Аня",
        text="сегодня хорошая погода в городе",
        mentioned=False,
        is_reply=False,
        reply_context_text=None,
        bot_aliases=frozenset({"пепе"}),
        monotonic_now=1000.0,
    )
    facts.update(overrides)
    return IncomingMessage(**facts)  # type: ignore[arg-type]


class _Outbox:
    """Приёмник отправки: конвейеру достаточно «отправить части».

    Возвращает число доставленных частей — контракт ``Sender``. Конвейер
    отличает по нему «ответа в чате нет» от «ответ ушёл частично»: от этого
    зависит, вернуть ли бюджет ответа и запоминать ли форму в анти-повторе.
    """

    def __init__(self) -> None:
        self.sent: list[list[str]] = []

    async def __call__(self, parts: Sequence[str]) -> int:
        delivered = [part for part in parts if part]
        self.sent.append(list(parts))
        return len(delivered)


class ReplyPipelineTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.learning_service = AsyncMock()
        self.learning_service.get_token_volume.return_value = 10_000
        self.learning_service.get_hot_ngrams.return_value = []
        self.learning_service.get_user_interaction_count.return_value = 0
        self.generator = AsyncMock()
        # Настоящая телеметрия, а не мок: у AsyncMock каждый note_* возвращает
        # корутину, которую никто не ждёт, и счётчики не накапливаются — тест
        # проверял бы, что метод позвали, а не что число сошлось. Ровно из-за
        # этого первая редакция тестов счётчиков (O14/O15) не увидела двух
        # дефектов проводки.
        self.generator.telemetry = GenerationTelemetry()
        self.outbox = _Outbox()
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

    async def _run(
        self, state: RuntimeState, msg: IncomingMessage
    ) -> tuple[ReplyPipeline, object]:
        pipeline = self._pipeline(state)
        observation = await pipeline.observe(msg)
        assert observation is not None
        await pipeline.respond(msg, observation, self.outbox)
        return pipeline, observation

    def _generated(self, text: str) -> object:
        """Подменяет генератор ответов, оставляя политику нетронутой."""
        response_generator = AsyncMock()
        response_generator.generate.return_value = text
        return patch(
            "app.services.reply_pipeline.ResponseGenerator",
            return_value=response_generator,
        )


class TestObserve(ReplyPipelineTestCase):
    async def test_short_unaddressed_message_is_dropped(self) -> None:
        state = _state()
        pipeline = self._pipeline(state)

        observation = await pipeline.observe(_incoming(text="ок"))

        self.assertIsNone(observation)
        # Ритм чата отмечается до гейта: короткие реплики тоже задают темп.
        self.assertIn(1, state._last_chat_activity)

    async def test_short_addressed_message_survives_the_gate(self) -> None:
        state = _state()
        pipeline = self._pipeline(state)

        observation = await pipeline.observe(_incoming(text="ок", mentioned=True))

        assert observation is not None
        self.assertFalse(observation.learnable)
        self.assertTrue(observation.address_reply)

    async def test_leading_vocative_is_stripped_before_learning(self) -> None:
        state = _state()
        pipeline = self._pipeline(state)

        observation = await pipeline.observe(
            _incoming(text="пепе, что там по погоде", mentioned=True)
        )

        assert observation is not None
        self.assertEqual(observation.learn_source, "что там по погоде")

    async def test_mention_within_cooldown_is_demoted(self) -> None:
        state = _state(mention_cooldown_sec=60)
        state.last_mention_reply_ts[(1, 5)] = 990.0
        pipeline = self._pipeline(state)

        observation = await pipeline.observe(_incoming(mentioned=True))

        assert observation is not None
        self.assertFalse(observation.address_reply)


class TestRespond(ReplyPipelineTestCase):
    async def test_addressed_message_without_model_data_gets_a_fallback(
        self,
    ) -> None:
        self.learning_service.get_token_volume.return_value = 0
        state = _state(user_quirk_chance=0.5)
        msg = _incoming(mentioned=True)

        await self._run(state, msg)

        self.assertEqual(len(self.outbox.sent), 1)
        self.assertEqual(len(self.outbox.sent[0]), 1)
        # Обращение зафиксировано, даже когда ответить нечем.
        self.assertIn((1, 5), state.last_mention_reply_ts)
        self.learning_service.record_user_interaction.assert_awaited_once_with(1, 5)

    async def test_unaddressed_message_without_model_data_is_silent(self) -> None:
        self.learning_service.get_token_volume.return_value = 0
        state = _state()

        await self._run(state, _incoming())

        self.assertEqual(self.outbox.sent, [])

    async def test_cooldown_blocks_an_unprompted_reply(self) -> None:
        state = _state(reply_probability=1.0, min_cooldown_sec=600, reply_director_enabled=False)
        state.last_reply_ts[1] = 999.0

        with self._generated("сгенерированный ответ"):
            await self._run(state, _incoming())

        self.assertEqual(self.outbox.sent, [])

    async def test_generated_reply_is_sent_and_remembered(self) -> None:
        state = _state(reply_probability=1.0, min_cooldown_sec=0, reply_director_enabled=False)

        with self._generated("сгенерированный ответ про погоду"):
            await self._run(state, _incoming())

        self.assertEqual(self.outbox.sent, [["сгенерированный ответ про погоду"]])
        self.assertIn(1, state.recent_replies)

    async def test_failed_generation_answers_an_address_with_a_fallback(
        self,
    ) -> None:
        state = _state(reply_probability=1.0, min_cooldown_sec=0, reply_director_enabled=False)

        with self._generated(""):
            await self._run(state, _incoming(mentioned=True))

        self.assertEqual(len(self.outbox.sent), 1)
        self.assertIn((1, 5), state.last_mention_reply_ts)

    async def test_failed_generation_stays_silent_when_not_addressed(self) -> None:
        state = _state(reply_probability=1.0, min_cooldown_sec=0, reply_director_enabled=False)

        with self._generated(""):
            await self._run(state, _incoming())

        self.assertEqual(self.outbox.sent, [])

    async def test_not_enough_data_fallback_counts_as_a_sent_reply(self) -> None:
        """Фолбэк «мало данных» — отправленный ответ, как и фолбэк генерации."""
        self.learning_service.get_token_volume.return_value = 0
        state = _state()

        await self._run(state, _incoming(mentioned=True))

        # Кулдаун/берст видят отправленный фолбэк...
        self.assertEqual(state.last_reply_ts.get(1), 1000.0)
        # ...а часовой кап — нет (unprompted=False).
        self.assertNotIn(1, state.recent_reply_times)

    async def test_legacy_branch_honors_the_hourly_cap(self) -> None:
        """reply_max_per_hour — предохранитель и без директора (M2)."""
        state = _state(
            reply_probability=1.0,
            min_cooldown_sec=0,
            reply_director_enabled=False,
            reply_max_per_hour=1,
        )
        state.recent_reply_times[1] = deque([999.5])

        with self._generated("сгенерированный ответ"):
            await self._run(state, _incoming())

        self.assertEqual(self.outbox.sent, [])


class TestHotNgramSeed(ReplyPipelineTestCase):
    async def test_seed_is_taken_for_an_unprompted_reply(self) -> None:
        state = _state(
            reply_probability=1.0,
            min_cooldown_sec=0,
            reply_director_enabled=False,
            hot_ngram_seed_chance=1.0,
        )
        self.learning_service.get_hot_ngrams.return_value = [("горячая", "фраза")]

        with self._generated("ответ") as response_gen_cls:
            await self._run(state, _incoming())

        request = response_gen_cls.return_value.generate.await_args.args[0]
        self.assertEqual(request.seed, ["горячая", "фраза"])

    async def test_address_reply_is_never_seeded(self) -> None:
        state = _state(
            reply_probability=1.0,
            min_cooldown_sec=0,
            reply_director_enabled=False,
            hot_ngram_seed_chance=1.0,
        )
        self.learning_service.get_hot_ngrams.return_value = [("горячая", "фраза")]

        with self._generated("ответ") as response_gen_cls:
            await self._run(state, _incoming(mentioned=True))

        request = response_gen_cls.return_value.generate.await_args.args[0]
        self.assertIsNone(request.seed)
        self.learning_service.get_hot_ngrams.assert_not_awaited()


class TestUserQuirkAndRareEvent(ReplyPipelineTestCase):
    def _quirk_state(self) -> RuntimeState:
        return _state(
            reply_probability=1.0,
            min_cooldown_sec=0,
            reply_director_enabled=False,
            user_quirk_chance=1.0,
            user_quirk_min_interactions=1,
            user_quirk_name_share=0.0,
        )

    async def test_regular_gets_a_vocative_as_a_separate_first_part(self) -> None:
        state = self._quirk_state()
        self.learning_service.get_user_interaction_count.return_value = 50

        with self._generated("ответ по делу"):
            await self._run(state, _incoming(mentioned=True))

        self.assertEqual(len(self.outbox.sent[0]), 2)
        self.assertEqual(self.outbox.sent[0][1], "ответ по делу")

    async def test_below_threshold_keeps_a_plain_reply(self) -> None:
        state = self._quirk_state()
        self.learning_service.get_user_interaction_count.return_value = 0

        with self._generated("ответ по делу"):
            await self._run(state, _incoming(mentioned=True))

        self.assertEqual(self.outbox.sent, [["ответ по делу"]])

    async def test_quirked_reply_spends_no_rare_event_budget(self) -> None:
        """Один слом формы на ответ: причуда снимает ролл редкого события."""
        state = self._quirk_state()
        state.rare_event_chance = 1.0
        self.learning_service.get_user_interaction_count.return_value = 50

        with self._generated("ответ по делу"):
            await self._run(state, _incoming(mentioned=True))

        self.assertEqual(state.rare_events_today, {})

    async def test_unprompted_reply_is_never_quirked(self) -> None:
        state = self._quirk_state()
        state.rare_event_chance = 0.0
        self.learning_service.get_user_interaction_count.return_value = 50

        with self._generated("ответ по делу"):
            await self._run(state, _incoming())

        self.assertEqual(self.outbox.sent, [["ответ по делу"]])
        self.learning_service.get_user_interaction_count.assert_not_awaited()


class TestLearn(ReplyPipelineTestCase):
    async def test_learnable_message_is_recorded(self) -> None:
        state = _state()
        pipeline = self._pipeline(state)
        msg = _incoming()
        observation = await pipeline.observe(msg)
        assert observation is not None

        await pipeline.learn(msg, observation)

        self.learning_service.record_message.assert_awaited_once()
        self.assertEqual(state.learned_messages[1], 1)

    async def test_learn_passes_the_effective_short_half_life(self) -> None:
        """M2R-210: писатель гасит короткий слой runtime-значением ручки."""
        state = _state(markov_short_half_life_days=7.0)
        pipeline = self._pipeline(state)
        msg = _incoming()
        observation = await pipeline.observe(msg)
        assert observation is not None

        await pipeline.learn(msg, observation)

        kwargs = self.learning_service.record_message.await_args.kwargs
        self.assertEqual(kwargs["short_half_life_days"], 7.0)

    async def test_maintenance_is_a_step_of_its_own(self) -> None:
        """Обслуживание не спрятано внутри записи сообщения.

        Пока оно было там, сбой обслуживания стоил выученного текста.
        """
        state = _state()
        pipeline = self._pipeline(state)

        await pipeline.run_due_maintenance()

        self.learning_service.run_due_maintenance.assert_awaited_once()

    async def test_maintenance_gets_the_effective_meme_knobs(self) -> None:
        """M2R-300: /set-ручки мемо-пасса читаются в момент вызова."""
        state = _state(
            markov_meme_min_joint_count=5,
            markov_meme_min_support=20.0,
            markov_meme_recency_days=7.0,
            markov_collocation_max_entries=50,
        )
        pipeline = self._pipeline(state)

        await pipeline.run_due_maintenance()

        self.learning_service.run_due_maintenance.assert_awaited_once_with(
            MemeSettings(
                min_joint_count=5,
                min_support=20.0,
                recency_days=7.0,
                max_entries=50,
            )
        )

    async def test_unlearnable_message_is_not_recorded(self) -> None:
        state = _state()
        pipeline = self._pipeline(state)
        msg = _incoming(text="ок", mentioned=True)
        observation = await pipeline.observe(msg)
        assert observation is not None

        await pipeline.learn(msg, observation)

        self.learning_service.record_message.assert_not_awaited()


class TestTempoCountersWiring(ReplyPipelineTestCase):
    """Счётчики проверяются через конвейер, а не на объекте телеметрии.

    Первая редакция тестов дёргала `note_mention` напрямую и потому пропустила
    два дефекта проводки сразу: знаменатель считался по уже отфильтрованному
    `address_reply` (ветка «обращение без ответа» недостижима —
    `should_reply_to_message` возвращает True на любом упоминании), а фолбэк
    «мало данных» не попадал в числитель вовсе. Счётчик, заведённый ради
    решения по O15, не мог показать ничего, кроме 100%.
    """

    async def test_mention_denominator_counts_the_raw_mention(self) -> None:
        """Упоминание, погашенное кулдауном обращений, — тоже нагрузка."""
        state = _state(mention_cooldown_sec=60)
        state.last_mention_reply_ts[(1, 5)] = 990.0
        pipeline = self._pipeline(state)

        observation = await pipeline.observe(_incoming(mentioned=True))

        assert observation is not None
        self.assertFalse(observation.address_reply, "тест не проверяет гейт")
        self.assertEqual(
            self.generator.telemetry.mentions_observed,
            1,
            "погашенное кулдауном упоминание выпало из знаменателя",
        )

    async def test_answer_share_can_be_below_one(self) -> None:
        """Доля ответов обязана уметь отличаться от 100%."""
        state = _state(mention_cooldown_sec=60)
        state.last_mention_reply_ts[(1, 5)] = 990.0
        pipeline = self._pipeline(state)

        msg = _incoming(mentioned=True)
        observation = await pipeline.observe(msg)
        assert observation is not None
        with self._generated("ответ бота из нескольких слов"):
            await pipeline.respond(msg, observation, self.outbox)

        share = self.generator.telemetry.snapshot()["mention_answer_share"]
        self.assertIsNotNone(share)
        self.assertLess(share, 1.0, "доля ответов тождественно равна единице")

    async def test_not_enough_data_fallback_counts_as_an_answer(self) -> None:
        self.learning_service.get_token_volume.return_value = 0
        state = _state()
        await self._run(state, _incoming(mentioned=True))

        self.assertEqual(self.generator.telemetry.mentions_answered, 1)

    async def test_tempo_is_counted_even_for_a_dropped_message(self) -> None:
        """Темп считается до раннего возврата — на любом сообщении.

        Короткие и необучаемые сообщения тоже задают темп чата, и
        распределение, собранное только по «хорошим», отвечало бы не на тот
        вопрос. Проверяется проводка, а не арифметика счётчика: мутация места
        вызова (удаление `note_chat_tempo`) без этого теста оставляла весь
        сьют зелёным.
        """
        state = _state(mood_enabled=True)
        pipeline = self._pipeline(state)

        observation = await pipeline.observe(_incoming(text="ок"))

        self.assertIsNone(observation, "сообщение должно отсеяться гейтом")
        self.assertEqual(
            sum(self.generator.telemetry.tempo_buckets.values()),
            1,
            "темп не посчитан на отсеянном сообщении",
        )

    async def test_burst_phase_is_counted_for_an_unprompted_reply(self) -> None:
        """Самостоятельный ответ в окне усиления попадает в счётчик.

        Парный к `test_burst_phase_ignores_mention_answers`: тот проверяет
        отсутствие (assert == 0) и потому переживает удаление всего блока.
        Без положительного теста проводка не охраняется ничем, а именно по
        этим числам будет закрываться O14.
        """
        state = _state(
            reply_director_enabled=True,
            reply_probability_min=1.0,
            reply_probability_max=1.0,
            min_cooldown_sec=0,
        )
        state.last_reply_ts[1] = 999.0  # 1 с назад — внутри окна усиления
        pipeline = self._pipeline(state)

        msg = _incoming()
        observation = await pipeline.observe(msg)
        assert observation is not None
        with self._generated("ответ бота из нескольких слов"):
            await pipeline.respond(msg, observation, self.outbox)

        telemetry = self.generator.telemetry
        self.assertEqual(
            telemetry.burst_phase_boost, 1, "фаза усиления не посчитана"
        )
        self.assertEqual(telemetry.burst_phase_suppress, 0)

    async def test_burst_phase_ignores_mention_answers(self) -> None:
        """Ответ на обращение берст-фактор не проходил — считать его нельзя."""
        state = _state(reply_director_enabled=True)
        state.last_reply_ts[1] = 999.0  # внутри окна усиления
        pipeline = self._pipeline(state)

        msg = _incoming(mentioned=True)
        observation = await pipeline.observe(msg)
        assert observation is not None
        with self._generated("ответ бота из нескольких слов"):
            await pipeline.respond(msg, observation, self.outbox)

        telemetry = self.generator.telemetry
        self.assertEqual(
            telemetry.burst_phase_boost + telemetry.burst_phase_suppress,
            0,
            "ответ на обращение попал в счётчик фазы берста",
        )


class TestReplyBudgetUnderConcurrency(ReplyPipelineTestCase):
    """Пределы темпа держатся и при одновременной доставке (O8).

    aiogram заводит задачу на каждый апдейт, а `getUpdates` отдаёт их пачками
    до 100 штук. Пока бюджет занимался после отправки, вся пачка успевала
    пройти проверку кулдауна до первой записи — и кулдаун не давал того, что
    обещает, ровно в burst-режиме, ради которого заведён.
    """

    def _slow_generated(self, text: str) -> object:
        """Генерация, которая реально уступает циклу.

        Без уступки тест зелён по неправильной причине: `AsyncMock` возвращает
        значение, ни разу не приостановив корутину, поэтому `asyncio.gather`
        доводит каждую до конца по очереди и чередования — то есть самой
        гонки — не возникает. Проверено мутацией: с мгновенным моком возврат
        резервации на прежнее место тесты не ронял.
        """
        import asyncio

        response_generator = AsyncMock()

        async def _generate(*args: object, **kwargs: object) -> str:
            await asyncio.sleep(0)
            return text

        response_generator.generate = AsyncMock(side_effect=_generate)
        return patch(
            "app.services.reply_pipeline.ResponseGenerator",
            return_value=response_generator,
        )

    async def _respond_many(
        self, state: RuntimeState, count: int, **msg_overrides: object
    ) -> None:
        import asyncio

        pipeline = self._pipeline(state)
        messages = [
            _incoming(text=f"сегодня хорошая погода в городе {i}", **msg_overrides)
            for i in range(count)
        ]
        observations = [await pipeline.observe(m) for m in messages]
        await asyncio.gather(
            *(
                pipeline.respond(m, o, self.outbox)
                for m, o in zip(messages, observations, strict=True)
                if o is not None
            )
        )

    async def test_burst_inside_the_cooldown_yields_one_reply(self) -> None:
        state = _state(reply_director_enabled=False, reply_probability=1.0)
        with self._slow_generated("ответ бота из нескольких слов"):
            await self._respond_many(state, 5)

        self.assertEqual(
            len(self.outbox.sent),
            1,
            f"кулдаун обойдён пачкой: {len(self.outbox.sent)} ответов вместо одного",
        )

    async def test_burst_does_not_exceed_the_hourly_cap(self) -> None:
        state = _state(
            reply_director_enabled=False,
            reply_probability=1.0,
            min_cooldown_sec=0,
            reply_max_per_hour=2,
        )
        with self._slow_generated("ответ бота из нескольких слов"):
            await self._respond_many(state, 6)

        self.assertLessEqual(
            len(self.outbox.sent),
            2,
            f"часовой кап превышен: {len(self.outbox.sent)} ответов при пределе 2",
        )

    async def test_failed_generation_returns_the_budget(self) -> None:
        """Ответ не состоялся — бюджет возвращается следующему сообщению."""
        state = _state(reply_director_enabled=False, reply_probability=1.0)
        pipeline = self._pipeline(state)
        msg = _incoming()
        observation = await pipeline.observe(msg)
        assert observation is not None

        with self._generated(""):
            await pipeline.respond(msg, observation, self.outbox)

        self.assertEqual(self.outbox.sent, [])
        self.assertNotIn(
            1, state.recent_reply_times, "несостоявшийся ответ съел часовой бюджет"
        )
        self.assertNotIn(1, state.last_reply_ts, "несостоявшийся ответ занял кулдаун")

    async def test_send_failure_before_delivery_returns_the_budget(self) -> None:
        """Ничего не доставлено — ответа в чате нет, бюджет возвращается."""
        state = _state(reply_director_enabled=False, reply_probability=1.0)
        pipeline = self._pipeline(state)
        msg = _incoming()
        observation = await pipeline.observe(msg)
        assert observation is not None

        async def failing_send(parts: Sequence[str]) -> int:
            # Ничего не доставлено — отправитель бросает исходное исключение,
            # а не PartialDeliveryError: обёртка заведена только для частичной
            # доставки, чтобы не подменять тип у команд и обработчика ошибок.
            raise RuntimeError("Telegram недоступен")

        with self._generated("ответ бота из нескольких слов"):
            with self.assertRaises(RuntimeError):
                await pipeline.respond(msg, observation, failing_send)

        self.assertNotIn(
            1, state.recent_reply_times, "упавшая отправка съела часовой бюджет"
        )

    async def test_partial_delivery_keeps_the_budget_spent(self) -> None:
        """Часть ответа в чате — бюджет остаётся потраченным.

        Ответ бота бывает многочастным (фальстарт, двойное сообщение, причуда
        завсегдатая). Сбой на второй части оставляет первую в чате, и возврат
        бюджета снял бы кулдаун за ответ, который люди уже видят: следующее
        сообщение получило бы второй ответ поверх оборванного первого.
        """
        state = _state(reply_director_enabled=False, reply_probability=1.0)
        pipeline = self._pipeline(state)
        msg = _incoming()
        observation = await pipeline.observe(msg)
        assert observation is not None

        async def half_delivered_send(parts: Sequence[str]) -> int:
            raise PartialDeliveryError(1)

        with self._generated("ответ бота из нескольких слов"):
            with self.assertRaises(PartialDeliveryError):
                await pipeline.respond(msg, observation, half_delivered_send)

        self.assertIn(
            1,
            state.recent_reply_times,
            "частично доставленный ответ вернул бюджет",
        )
        self.assertIn(1, state.last_reply_ts, "частичная доставка сняла кулдаун")

    async def test_silent_zero_delivery_returns_the_budget(self) -> None:
        """Отправитель вернул 0 без исключения — ответа тоже нет."""
        state = _state(reply_director_enabled=False, reply_probability=1.0)
        pipeline = self._pipeline(state)
        msg = _incoming()
        observation = await pipeline.observe(msg)
        assert observation is not None

        async def silent_send(parts: Sequence[str]) -> int:
            return 0

        with self._generated("ответ бота из нескольких слов"):
            await pipeline.respond(msg, observation, silent_send)

        self.assertNotIn(1, state.recent_reply_times)
        self.assertNotIn(
            1, state.recent_replies, "анти-повтор запомнил неотправленное"
        )


if __name__ == "__main__":
    unittest.main()
