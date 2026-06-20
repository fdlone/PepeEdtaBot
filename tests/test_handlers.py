"""Happy-path tests for all routers.

Each handler is called directly with faked Message / service objects.
No aiogram dispatcher is started; we only verify that handlers call
the right service methods and reply to the message.
"""
from __future__ import annotations

import unittest
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_message(
    *,
    text: str = "",
    chat_type: str = "supergroup",
    user_id: int = 1,
    chat_id: int = 100,
    is_bot: bool = False,
    reply_to: MagicMock | None = None,
) -> MagicMock:
    msg = MagicMock()
    msg.text = text
    msg.chat.id = chat_id
    msg.chat.type = chat_type
    msg.from_user.id = user_id
    msg.from_user.is_bot = is_bot
    msg.reply_to_message = reply_to
    msg.reply = AsyncMock()
    msg.bot = AsyncMock()
    msg.bot.send_chat_action = AsyncMock()
    return msg


def _fake_state(**kwargs: object) -> MagicMock:
    s = MagicMock()
    s.typing_min_ms = 0
    s.typing_max_ms = 0
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------

class TestCommonHandlers(unittest.IsolatedAsyncioTestCase):
    async def test_ping_replies_pong(self) -> None:
        from app.handlers.common import cmd_ping
        msg = _fake_message()
        await cmd_ping(msg)
        msg.reply.assert_awaited_once_with("pong")

    async def test_help_replies(self) -> None:
        from app.handlers.common import cmd_help
        msg = _fake_message()
        state = _fake_state()
        await cmd_help(msg, state)
        msg.reply.assert_awaited_once()

    async def test_stats_replies_with_stats(self) -> None:
        from app.handlers.common import cmd_stats
        msg = _fake_message()
        db = AsyncMock()
        db.get_stats = AsyncMock(return_value={
            "messages": 5, "starts2": 1, "starts3": 1,
            "transitions2": 2, "transitions3": 1, "transitions1": 3,
            "volume2": 10, "volume3": 5, "volume1": 15, "volume": 5,
        })
        state = _fake_state(min_tokens_for_model=50)
        await cmd_stats(msg, db, state)
        msg.reply.assert_awaited_once()


# ---------------------------------------------------------------------------
# admin.py
# ---------------------------------------------------------------------------

class TestAdminHandlers(unittest.IsolatedAsyncioTestCase):
    async def test_config_replies(self) -> None:
        from app.handlers.admin import cmd_config
        msg = _fake_message(text="/config")
        state = _fake_state()
        with patch("app.handlers.admin.format_config_message", return_value="cfg"):
            await cmd_config(msg, state)
        msg.reply.assert_awaited_once()

    async def test_set_no_args_shows_usage(self) -> None:
        from app.handlers.admin import cmd_set
        msg = _fake_message(text="/set")
        state = _fake_state()
        settings = MagicMock()
        await cmd_set(msg, state, settings)
        msg.reply.assert_awaited_once()
        assert "Использование" in msg.reply.call_args[0][0]

    async def test_set_valid_key_applies(self) -> None:
        from app.handlers.admin import cmd_set
        from runtime_state import RuntimeState
        msg = _fake_message(text="/set reply_probability 0.5")
        real_state = MagicMock(spec=RuntimeState)
        real_state.reply_probability = 0.1
        settings = MagicMock()
        with patch("app.handlers.admin.apply_runtime_setting") as mock_apply:
            await cmd_set(msg, real_state, settings)
            mock_apply.assert_called_once()
        msg.reply.assert_awaited_once()

    async def test_setprob_valid_value(self) -> None:
        from app.handlers.admin import cmd_setprob
        msg = _fake_message(text="/setprob 0.3")
        state = _fake_state(reply_probability=0.1)
        settings = MagicMock()
        await cmd_setprob(msg, state, settings)
        assert state.reply_probability == 0.3
        msg.reply.assert_awaited_once()

    async def test_setprob_invalid_value_replies_error(self) -> None:
        from app.handlers.admin import cmd_setprob
        msg = _fake_message(text="/setprob abc")
        state = _fake_state()
        settings = MagicMock()
        await cmd_setprob(msg, state, settings)
        msg.reply.assert_awaited_once()
        assert "диапазон" in msg.reply.call_args[0][0]

    async def test_clear_without_confirm_asks_confirmation(self) -> None:
        from app.handlers.admin import cmd_clear
        msg = _fake_message(text="/clear")
        db = AsyncMock()
        state = _fake_state()
        generator = MagicMock()
        settings = MagicMock()
        with patch("app.handlers.admin.format_clear_confirmation_message", return_value="confirm?"):
            await cmd_clear(msg, db, state, generator, settings)
        db.clear_chat.assert_not_called()
        msg.reply.assert_awaited_once()

    async def test_clear_with_confirm_clears_chat(self) -> None:
        from app.handlers.admin import cmd_clear
        msg = _fake_message(text="/clear confirm")
        db = AsyncMock()
        state = _fake_state()
        generator = MagicMock()
        settings = MagicMock()
        await cmd_clear(msg, db, state, generator, settings)
        db.clear_chat.assert_awaited_once_with(msg.chat.id)
        msg.reply.assert_awaited_once()

    # --- fallback handlers для unauthorized админ-команд ---

    async def test_set_denied_replies_with_explanation(self) -> None:
        from app.handlers.admin import cmd_set_denied
        msg = _fake_message(text="/set foo bar")
        state = _fake_state()
        await cmd_set_denied(msg, state)
        msg.reply.assert_awaited_once()
        text = msg.reply.call_args[0][0]
        assert "OWNER_ID" in text
        assert "админ" in text.lower()

    async def test_setprob_denied_replies_with_explanation(self) -> None:
        from app.handlers.admin import cmd_setprob_denied
        msg = _fake_message(text="/setprob 0.5")
        state = _fake_state()
        await cmd_setprob_denied(msg, state)
        msg.reply.assert_awaited_once()
        assert "OWNER_ID" in msg.reply.call_args[0][0]

    async def test_clear_denied_replies_with_explanation(self) -> None:
        from app.handlers.admin import cmd_clear_denied
        msg = _fake_message(text="/clear confirm")
        state = _fake_state()
        await cmd_clear_denied(msg, state)
        msg.reply.assert_awaited_once()
        text = msg.reply.call_args[0][0]
        assert "Недостаточно прав" in text
        # /clear имеет специфичный текст, отличный от общего «доступна OWNER_ID...»
        assert "OWNER_ID" in text or "админ" in text.lower()

    def test_admin_router_handler_registration_order(self) -> None:
        """
        Защита от случайной перестановки fallback-handlers.

        Aiogram перебирает handlers Router'а в порядке регистрации
        (см. aiogram/dispatcher/event/telegram.py:111-130: for handler
        in self.handlers с return на первом match). Если cmd_set_denied
        окажется зарегистрирован раньше cmd_set, fallback всегда будет
        выигрывать — admin-команда никогда не выполнится для реальных
        админов, и баг тихо пройдёт CI.
        """
        from app.handlers.admin import router

        callbacks = [h.callback for h in router.message.handlers]
        names = [cb.__name__ for cb in callbacks]

        pairs = [
            ("cmd_set", "cmd_set_denied"),
            ("cmd_setprob", "cmd_setprob_denied"),
            ("cmd_clear", "cmd_clear_denied"),
        ]
        for protected, fallback in pairs:
            assert protected in names, f"{protected} not registered"
            assert fallback in names, f"{fallback} not registered"
            assert names.index(protected) < names.index(fallback), (
                f"{fallback} зарегистрирован раньше {protected}: "
                f"fallback всегда будет выигрывать, защищённый handler "
                f"никогда не вызовется. Поменяйте порядок в admin.py."
            )


# ---------------------------------------------------------------------------
# pivo.py
# ---------------------------------------------------------------------------

class TestPivoHandlers(unittest.IsolatedAsyncioTestCase):
    def test_pivo_router_handlers_are_group_only(self) -> None:
        from app.filters import GroupOnly
        from app.handlers.pivo import router

        expected = {
            "cmd_pivo",
            "cmd_pivo_on",
            "cmd_pivo_off",
            "cmd_pivo_privacy",
        }

        handlers = {
            handler.callback.__name__: handler
            for handler in router.message.handlers
            if handler.callback.__name__ in expected
        }

        self.assertEqual(set(handlers), expected)
        for name, handler in handlers.items():
            callbacks = [filter_object.callback for filter_object in handler.filters]
            self.assertTrue(
                any(isinstance(callback, GroupOnly) for callback in callbacks),
                f"{name} must be registered with GroupOnly",
            )

    async def test_pivo_calls_build_and_replies(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2))
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.build_call_message.assert_awaited_once_with(
            chat_id=msg.chat.id,
            caller_user_id=msg.from_user.id,
            planned_time=None,
            target=None,
            explicit_mentions=(),
        )
        pivo_service.consume_daily_call_quota.assert_awaited_once_with(
            chat_id=msg.chat.id,
            user_id=msg.from_user.id,
            is_admin_or_owner=False,
        )
        msg.reply.assert_awaited_once()

    async def test_pivo_passes_parsed_arguments_to_service(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message(text="/pivo 20:00 watch movie @friend")
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 1))
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.consume_daily_call_quota.assert_awaited_once()
        pivo_service.build_call_message.assert_awaited_once_with(
            chat_id=msg.chat.id,
            caller_user_id=msg.from_user.id,
            planned_time="20:00",
            target="watch movie",
            explicit_mentions=("@friend",),
        )
        msg.reply.assert_awaited_once()

    async def test_pivo_rejects_over_limit_mentions_without_spending_quota(self) -> None:
        from app.handlers.pivo import cmd_pivo
        from app.services.pivo_service import PivoCallLimitError

        msg = _fake_message(text="/pivo @one @two @three")
        pivo_service = AsyncMock()
        pivo_service.build_call_message = AsyncMock(
            side_effect=PivoCallLimitError(
                "В /pivo можно указывать не больше 2 явных упоминаний за раз."
            )
        )
        state = _fake_state()
        bot = AsyncMock()
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.build_call_message.assert_awaited_once()
        pivo_service.consume_daily_call_quota.assert_not_called()
        msg.reply.assert_awaited_once()
        assert "не больше 2" in msg.reply.call_args[0][0]

    async def test_pivo_rejects_when_daily_quota_is_exhausted(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(
            allowed=False,
            limit=3,
            usage_day="2026-05-12",
        )
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2))
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.build_call_message.assert_awaited_once()
        pivo_service.consume_daily_call_quota.assert_awaited_once()
        msg.reply.assert_awaited_once()
        assert "Лимит /pivo" in msg.reply.call_args[0][0]

    async def test_pivo_uses_admin_daily_quota_path(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2))
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(
            return_value=[MagicMock(user=MagicMock(id=msg.from_user.id))]
        )
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.consume_daily_call_quota.assert_awaited_once_with(
            chat_id=msg.chat.id,
            user_id=msg.from_user.id,
            is_admin_or_owner=True,
        )
        pivo_service.build_call_message.assert_awaited_once()
        msg.reply.assert_awaited_once()

    async def test_pivo_owner_bypasses_daily_quota(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message(user_id=42)
        pivo_service = AsyncMock()
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2))
        state = _fake_state()
        bot = AsyncMock()
        settings = MagicMock(owner_id=42)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.consume_daily_call_quota.assert_not_called()
        bot.get_chat_administrators.assert_not_called()
        pivo_service.build_call_message.assert_awaited_once()
        msg.reply.assert_awaited_once()

    async def test_pivo_refunds_quota_when_reply_fails(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        msg.reply = AsyncMock(side_effect=RuntimeError("telegram failed"))
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2))
        pivo_service.refund_daily_call_quota = AsyncMock()
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        with self.assertRaises(RuntimeError):
            await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.consume_daily_call_quota.assert_awaited_once()
        pivo_service.refund_daily_call_quota.assert_awaited_once_with(
            chat_id=msg.chat.id,
            user_id=msg.from_user.id,
            usage_day=quota.usage_day,
        )

    async def test_pivo_on_subscribes_user(self) -> None:
        from app.handlers.pivo import cmd_pivo_on
        msg = _fake_message(is_bot=False)
        pivo_service = AsyncMock()
        state = _fake_state()
        await cmd_pivo_on(msg, pivo_service, state)
        pivo_service.subscribe.assert_awaited_once()
        msg.reply.assert_awaited_once()

    async def test_pivo_on_ignores_bots(self) -> None:
        from app.handlers.pivo import cmd_pivo_on
        msg = _fake_message(is_bot=True)
        pivo_service = AsyncMock()
        state = _fake_state()
        await cmd_pivo_on(msg, pivo_service, state)
        pivo_service.subscribe.assert_not_called()
        msg.reply.assert_awaited_once()

    async def test_pivo_off_unsubscribes_user(self) -> None:
        from app.handlers.pivo import cmd_pivo_off
        msg = _fake_message()
        pivo_service = AsyncMock()
        state = _fake_state()
        await cmd_pivo_off(msg, pivo_service, state)
        pivo_service.unsubscribe.assert_awaited_once()
        msg.reply.assert_awaited_once()

    async def test_pivo_privacy_replies(self) -> None:
        from app.handlers.pivo import cmd_pivo_privacy
        msg = _fake_message()
        state = _fake_state()
        await cmd_pivo_privacy(msg, state)
        msg.reply.assert_awaited_once()


# ---------------------------------------------------------------------------
# learning.py — extract_context_tokens
# ---------------------------------------------------------------------------

class TestExtractContextTokens(unittest.TestCase):
    def _call(self, **kwargs: object) -> list[str]:
        from app.handlers.learning import extract_context_tokens
        defaults: dict[str, object] = dict(
            message=MagicMock(reply_to_message=None),
            current_text="hello world",
            normalize_lower=False,
            max_tokens=10,
            only_for_replies=False,
            include_current_message=True,
        )
        defaults.update(kwargs)
        return extract_context_tokens(**defaults)  # type: ignore[arg-type]

    def test_returns_tokens_from_current_text(self) -> None:
        tokens = self._call(current_text="один два три")
        self.assertGreater(len(tokens), 0)

    def test_only_for_replies_returns_empty_when_no_reply(self) -> None:
        msg = MagicMock()
        msg.reply_to_message = None
        tokens = self._call(message=msg, only_for_replies=True)
        self.assertEqual(tokens, [])

    def test_max_tokens_respected(self) -> None:
        tokens = self._call(
            current_text="один два три четыре пять шесть семь восемь девять десять одиннадцать",
            max_tokens=3,
        )
        self.assertLessEqual(len(tokens), 3)


class TestReplyHumanizedResilience(unittest.IsolatedAsyncioTestCase):
    """`reply_humanized` is the single point that calls Telegram's chat-action
    API. A 5xx or network blip there must not block the actual reply — the
    helper catches the exception and proceeds to `message.reply`."""

    async def test_send_chat_action_failure_does_not_block_reply(self) -> None:
        from app.handlers._helpers import reply_humanized

        msg = _fake_message()
        msg.bot.send_chat_action = AsyncMock(
            side_effect=RuntimeError("telegram chat-action 5xx")
        )

        await reply_humanized(msg, "ответ", typing_min_ms=0, typing_max_ms=0)

        msg.reply.assert_awaited_once_with("ответ")

    async def test_send_chat_action_called_when_bot_present(self) -> None:
        from app.handlers._helpers import reply_humanized

        msg = _fake_message()
        await reply_humanized(msg, "ответ", typing_min_ms=0, typing_max_ms=0)
        msg.bot.send_chat_action.assert_awaited_once()
        msg.reply.assert_awaited_once_with("ответ")

    async def test_pivo_on_still_subscribes_even_when_chat_action_fails(self) -> None:
        from app.handlers.pivo import cmd_pivo_on

        msg = _fake_message(is_bot=False)
        msg.bot.send_chat_action = AsyncMock(
            side_effect=RuntimeError("transient telegram error")
        )
        pivo_service = AsyncMock()
        state = _fake_state()

        await cmd_pivo_on(msg, pivo_service, state)

        pivo_service.subscribe.assert_awaited_once()
        msg.reply.assert_awaited_once()


class TestLearningMessageLength(unittest.TestCase):
    def test_learning_message_length_boundaries(self) -> None:
        from app.handlers.learning import (
            MAX_LEARN_MESSAGE_CHARS,
            is_learnable_message_length,
        )

        self.assertTrue(is_learnable_message_length("x" * MAX_LEARN_MESSAGE_CHARS))
        self.assertFalse(is_learnable_message_length("x" * (MAX_LEARN_MESSAGE_CHARS + 1)))

    def test_learning_token_boundaries(self) -> None:
        from app.handlers.learning import has_enough_tokens_for_learning

        self.assertFalse(has_enough_tokens_for_learning(["один"]))
        self.assertTrue(has_enough_tokens_for_learning(["один", "два"]))


class TestLearningHandler(unittest.IsolatedAsyncioTestCase):
    def _reply_state(self, **overrides: object) -> MagicMock:
        values: dict[str, object] = {
            "normalize_lower": False,
            "learned_messages": {},
            "recent_short_replies": {},
            "min_tokens_for_model": 10,
            "min_cooldown_sec": 0,
            "last_reply_ts": {},
            "reply_probability": 0.0,
            "use_reply_context": False,
            "reply_context_last_tokens": 3,
            "reply_context_bias": 1.8,
            "reply_context_start_bias": 2.2,
            "max_reply_chars": 280,
            "max_reply_tokens": 45,
            "randomness_strength": 0.0,
            "repetition_penalty_strength": 1.0,
            "markov_order": 3,
            "enable_backoff": True,
            "backoff_min_order": 1,
        }
        values.update(overrides)
        return _fake_state(**values)

    async def test_single_token_message_is_not_learned(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="hello")
        learning_service = AsyncMock()
        generator = AsyncMock()
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
        )

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
            )

        learning_service.record_message.assert_not_called()
        msg.reply.assert_not_awaited()

    async def test_threshold_crossing_message_is_learned_without_replying_to_itself(
        self,
    ) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe threshold crossing")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=9)
        learning_service.record_message = AsyncMock(return_value=11)
        generator = AsyncMock()
        state = self._reply_state()

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
            )

        learning_service.get_token_volume.assert_awaited_once_with(msg.chat.id)
        learning_service.record_message.assert_awaited_once()
        generator.generate_text.assert_not_awaited()
        msg.reply.assert_awaited_once_with("Пока мало материала, поболтайте ещё 🙂")

    async def test_current_incoming_message_copy_is_rejected(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = AsyncMock()
        generator.generate_text = AsyncMock(side_effect=["pepe ответь", "Нормально"])
        state = self._reply_state()

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
            )

        self.assertEqual(generator.generate_text.await_count, 2)
        learning_service.record_message.assert_awaited_once()
        msg.reply.assert_awaited_once_with("Нормально")

    async def test_message_is_persisted_when_cooldown_blocks_reply(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="обычное сообщение")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        generator = AsyncMock()
        state = self._reply_state(
            last_reply_ts={msg.chat.id: 10**20},
            min_cooldown_sec=60,
            reply_probability=1.0,
        )

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
            )

        learning_service.record_message.assert_awaited_once()
        generator.generate_text.assert_not_awaited()
        msg.reply.assert_not_awaited()

    async def test_message_is_persisted_when_generation_fails(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe generate something")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=103)
        generator = AsyncMock()
        generator.generate_text = AsyncMock(return_value="")
        state = self._reply_state()

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
            )

        learning_service.record_message.assert_awaited_once()
        self.assertGreater(generator.generate_text.await_count, 0)
        msg.reply.assert_awaited_once_with(
            "Собираю мысли... Напишите ещё пару сообщений 🙂"
        )

    async def test_prefix_rejection_gets_extra_retry_budget(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь нормально")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(
            side_effect=[True, True, True, True, False]
        )
        generator = AsyncMock()
        generator.generate_text = AsyncMock(
            side_effect=[
                "один два три четыре",
                "один два три четыре",
                "один два три четыре",
                "один два три четыре",
                "новый ответ теперь",
            ]
        )
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            reply_context_last_tokens=3,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.0,
            repetition_penalty_strength=1.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
        )

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
            )

        self.assertEqual(generator.generate_text.await_count, 5)
        msg.reply.assert_awaited_once_with("новый ответ теперь")

    async def test_short_reply_skips_training_prefix_filter(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=True)
        generator = AsyncMock()
        generator.generate_text = AsyncMock(return_value="Привет")
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            recent_short_replies={},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            reply_context_last_tokens=3,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.0,
            repetition_penalty_strength=1.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
        )

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
            )

        learning_service.is_verbatim_copy.assert_not_awaited()
        msg.reply.assert_awaited_once_with("Привет")
        self.assertEqual(list(state.recent_short_replies[msg.chat.id]), ["привет"])

    async def test_recent_short_reply_is_retried_instead_of_sent(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = AsyncMock()
        generator.generate_text = AsyncMock(side_effect=["Привет", "Нормально"])
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            recent_short_replies={msg.chat.id: deque(["привет"], maxlen=5)},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            reply_context_last_tokens=3,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.0,
            repetition_penalty_strength=1.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
        )

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
            )

        self.assertEqual(generator.generate_text.await_count, 2)
        msg.reply.assert_awaited_once_with("Нормально")
        self.assertEqual(
            list(state.recent_short_replies[msg.chat.id]),
            ["привет", "нормально"],
        )

    async def test_handler_increases_randomness_on_generation_retries(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь нормально")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(
            side_effect=[True, True, False]
        )
        generator = AsyncMock()
        generator.generate_text = AsyncMock(
            side_effect=[
                "первый ответ заметно длиннее",
                "второй ответ заметно длиннее",
                "третий ответ заметно длиннее",
            ]
        )
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            reply_context_last_tokens=3,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.5,
            repetition_penalty_strength=1.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
        )

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
            )

        strengths = [
            call.kwargs["randomness_strength"]
            for call in generator.generate_text.await_args_list
        ]
        self.assertEqual(strengths, sorted(strengths))
        self.assertEqual(strengths[0], 0.5)
        self.assertGreater(strengths[-1], strengths[0])
        msg.reply.assert_awaited_once_with("третий ответ заметно длиннее")
