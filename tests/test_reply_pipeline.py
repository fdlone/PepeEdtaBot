"""Ветки политики ответа — на уровне сервиса, без моков Telegram.

Раньше каждая из этих веток проверялась только полным прогоном обработчика:
чтобы дойти до причуды завсегдатая, тест собирал `Message`, `from_user`,
`bot`, `reply` и ждал, пока конвейер доберётся до нужной строки. Здесь
конвейер получает факты (`IncomingMessage`) и функцию отправки, поэтому
проверяется решение, а не сборка сообщения.
"""
from __future__ import annotations

import unittest
from collections.abc import Sequence
from unittest.mock import AsyncMock, patch

from app.config.registry import RUNTIME_FIELDS
from app.config.runtime_state import RuntimeState
from app.services.reply_pipeline import IncomingMessage, ReplyPipeline


def _state(**overrides: object) -> RuntimeState:
    """Реестровые дефолты — та же конфигурация, с которой бот запускается.

    Каналы, которые роллят кубик поверх решения (редкие события, фальстарт),
    по умолчанию выключены: иначе почти каждый тест иногда получал бы лишнюю
    часть ответа. Проверяющие их тесты включают чанс явно.
    """
    state = RuntimeState(
        **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS},
        runtime_state_ttl_sec=86_400,
        runtime_state_max_chats=2048,
    )
    state.rare_event_chance = 0.0
    state.false_start_chance = 0.0
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
    """Приёмник отправки: конвейеру достаточно «отправить части»."""

    def __init__(self) -> None:
        self.sent: list[list[str]] = []

    async def __call__(self, parts: Sequence[str]) -> None:
        self.sent.append(list(parts))


class ReplyPipelineTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.learning_service = AsyncMock()
        self.learning_service.get_token_volume.return_value = 10_000
        self.learning_service.get_hot_ngrams.return_value = []
        self.generator = AsyncMock()
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

    async def test_unlearnable_message_is_not_recorded(self) -> None:
        state = _state()
        pipeline = self._pipeline(state)
        msg = _incoming(text="ок", mentioned=True)
        observation = await pipeline.observe(msg)
        assert observation is not None

        await pipeline.learn(msg, observation)

        self.learning_service.record_message.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
