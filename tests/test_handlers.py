"""Happy-path tests for all routers.

Each handler is called directly with faked Message / service objects.
No aiogram dispatcher is started; we only verify that handlers call
the right service methods and reply to the message.
"""
from __future__ import annotations

import random
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _traced_generator() -> AsyncMock:
    """MarkovGenerator mock whose generate_text_with_trace delegates to the
    plain generate_text AsyncMock tests configure, wrapping the text in the
    (text, trace) tuple the ResponseGenerator consumes."""
    generator = AsyncMock()

    async def _delegate(*args: object, **kwargs: object) -> tuple[str, SimpleNamespace]:
        text = await generator.generate_text(*args, **kwargs)
        return text, SimpleNamespace(markov_order_used=3, start_source="global")

    generator.generate_text_with_trace = AsyncMock(side_effect=_delegate)
    return generator


def _pivo_stub() -> AsyncMock:
    """PivoService mock: on_text_message refreshes the sender's /pivo profile."""
    service = AsyncMock()
    service.refresh_member = AsyncMock()
    return service


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
    msg.answer = AsyncMock()
    msg.bot = AsyncMock()
    msg.bot.send_chat_action = AsyncMock()
    return msg


def _fake_state(**kwargs: object) -> MagicMock:
    s = MagicMock()
    s.typing_min_ms = 0
    s.typing_max_ms = 0
    s.typing_per_char_ms = 0
    s.recent_fallbacks = {}
    s.recent_replies = {}
    s.recent_reply_penalty_strength = 1.0
    s.verbatim_penalty_strength = 0.0
    s.length_mode_weights = (0.25, 0.55, 0.2)
    s.length_context_adaptation = 0.0
    # Deterministic selection and untouched reply text so handler tests can
    # assert generated candidates verbatim.
    s.candidate_selection_temperature = 0.0
    s.reply_flavor_strength = 0.0
    # Emoji channel off by default so generation tests assert reply text verbatim
    # and don't hit the learning service's emoji-stats read.
    s.emoji_append_chance = 0.0
    s.pivo_recent_pool_window = 5
    s.pivo_temporal_flavor_chance = 0.5
    # M4 jump off by default so generation tests stay deterministic; the markov
    # characterization tests drive jump_probability directly.
    s.markov_jump_probability = 0.0
    s.context_jump_boost = 1.0
    s.verbatim_extension_share = 0.0
    s.order_mix_probability = 0.0
    s.slot_mutation_probability = 0.0
    # L1 hot-ngram channel off by default so learn/reply tests stay
    # deterministic; dedicated hot-ngram tests enable it explicitly.
    s.hot_ngram_seed_chance = 0.0
    s.hot_ngram_min_count = 3
    s.hot_ngram_recency_share = 0.5
    # L3 rare events off by default so reply tests assert a single message;
    # dedicated rare-event tests enable the chances explicitly.
    s.rare_event_chance = 0.0
    s.false_start_chance = 0.0
    s.rare_event_daily_cap = 3
    s.user_quirk_chance = 0.0
    s.user_quirk_min_interactions = 25
    s.rare_events_today = {}
    # Bind the real cap methods so handler tests exercise actual budget logic.
    from app.config.runtime_state import RuntimeState

    s.can_fire_rare_event = (
        lambda chat_id, today: RuntimeState.can_fire_rare_event(s, chat_id, today)
    )
    s.note_rare_event = (
        lambda chat_id, today: RuntimeState.note_rare_event(s, chat_id, today)
    )
    # L2 quirk gate: real once-a-day logic for the same reason as the caps.
    s.last_user_quirk_day = {}
    s.can_fire_user_quirk = (
        lambda chat_id, user_id, today: RuntimeState.can_fire_user_quirk(
            s, chat_id, user_id, today
        )
    )
    s.note_user_quirk = (
        lambda chat_id, user_id, today: RuntimeState.note_user_quirk(
            s, chat_id, user_id, today
        )
    )
    # Mood off by default so the existing behavioural assertions see the
    # unmodulated path; dedicated mood tests enable it explicitly.
    s.mood_enabled = False
    # M2 director off by default so the flat reply_probability path is exercised
    # by the existing tests; dedicated director tests enable it explicitly.
    s.reply_director_enabled = False
    # Off by default so generated reply text is asserted verbatim; a bare
    # MagicMock attribute would be truthy and trigger reply capitalization.
    s.auto_capitalize_replies = False
    # Mention anti-flood gate off by default so existing mention-driven tests
    # keep their guaranteed-reply behaviour; dedicated tests enable it.
    s.mention_cooldown_sec = 0
    s.last_mention_reply_ts = {}
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
            "transitions2": 2, "transitions3": 1,
            "volume2": 10, "volume3": 5, "volume": 5,
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
        from app.config.runtime_state import RuntimeState
        from app.handlers.admin import cmd_set

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
        pivo_service = AsyncMock()
        with patch("app.handlers.admin.format_clear_confirmation_message", return_value="confirm?"):
            await cmd_clear(msg, db, state, generator, settings, pivo_service)
        db.clear_chat.assert_not_called()
        pivo_service.clear_chat_data.assert_not_called()
        msg.reply.assert_awaited_once()

    async def test_clear_with_confirm_clears_chat(self) -> None:
        from app.handlers.admin import cmd_clear
        msg = _fake_message(text="/clear confirm")
        db = AsyncMock()
        state = _fake_state()
        generator = MagicMock()
        settings = MagicMock()
        pivo_service = AsyncMock()
        await cmd_clear(msg, db, state, generator, settings, pivo_service)
        db.clear_chat.assert_awaited_once_with(msg.chat.id)
        pivo_service.clear_chat_data.assert_awaited_once_with(msg.chat.id)
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
    def setUp(self) -> None:
        from app.filters import admin_or_owner

        # The admin-id cache is process-global; reset it so the shared chat_id
        # across these tests does not leak admin sets between cases.
        admin_or_owner._admin_cache.clear()

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
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2, {}))
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
            recent_pool_window=5,
            temporal_flavor_chance=0.5,
            now=ANY,
        )
        pivo_service.consume_daily_call_quota.assert_awaited_once_with(
            chat_id=msg.chat.id,
            user_id=msg.from_user.id,
            is_admin_or_owner=False,
        )
        msg.reply.assert_awaited_once()
        pivo_service.record_pool_usage.assert_awaited_once_with(
            msg.chat.id, {}, recent_pool_window=5
        )

    async def test_pivo_passes_parsed_arguments_to_service(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message(text="/pivo 20:00 watch movie @friend")
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 1, {}))
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
            recent_pool_window=5,
            temporal_flavor_chance=0.5,
            now=ANY,
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
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2, {}))
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.build_call_message.assert_awaited_once()
        pivo_service.consume_daily_call_quota.assert_awaited_once()
        # N3: a quota-rejected call must not rotate the anti-repeat pools.
        pivo_service.record_pool_usage.assert_not_called()
        msg.reply.assert_awaited_once()
        assert "Лимит /pivo" in msg.reply.call_args[0][0]

    async def test_pivo_uses_admin_daily_quota_path(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2, {}))
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
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2, {}))
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
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2, {}))
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

class TestComputeTypingDelay(unittest.TestCase):
    def test_zero_per_char_keeps_base_range(self) -> None:
        from app.handlers._helpers import compute_typing_delay_ms

        rng = random.Random(1)
        for _ in range(20):
            delay = compute_typing_delay_ms(500, 350, 1100, 0, rng=rng)
            self.assertGreaterEqual(delay, 350)
            self.assertLessEqual(delay, 1100)

    def test_longer_text_waits_longer(self) -> None:
        from app.handlers._helpers import compute_typing_delay_ms

        short = compute_typing_delay_ms(10, 350, 350, 12, rng=random.Random(1))
        long = compute_typing_delay_ms(200, 350, 350, 12, rng=random.Random(1))
        self.assertEqual(short, 350 + 10 * 12)
        self.assertEqual(long, 350 + 200 * 12)

    def test_delay_is_capped(self) -> None:
        from app.handlers._helpers import (
            TYPING_HARD_CAP_MS,
            compute_typing_delay_ms,
        )

        delay = compute_typing_delay_ms(4000, 350, 1100, 200, rng=random.Random(1))
        self.assertEqual(delay, TYPING_HARD_CAP_MS)

    def test_cap_never_cuts_below_configured_max(self) -> None:
        from app.handlers._helpers import compute_typing_delay_ms

        delay = compute_typing_delay_ms(0, 5000, 5000, 0, rng=random.Random(1))
        self.assertEqual(delay, 5000)


class TestPivoChatActionResilience(unittest.IsolatedAsyncioTestCase):
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


class TestReplyHumanizedSequence(unittest.IsolatedAsyncioTestCase):
    async def test_single_part_replies_once(self) -> None:
        from app.handlers._helpers import reply_humanized_sequence

        msg = _fake_message(text="x")
        with patch("app.handlers._helpers.asyncio.sleep", new=AsyncMock()):
            await reply_humanized_sequence(msg, ["один"], 0, 0)
        msg.reply.assert_awaited_once_with("один")
        msg.answer.assert_not_awaited()

    async def test_two_parts_reply_then_answer_with_two_pauses(self) -> None:
        from app.handlers._helpers import reply_humanized_sequence

        msg = _fake_message(text="x")
        sleep = AsyncMock()
        with patch("app.handlers._helpers.asyncio.sleep", new=sleep):
            await reply_humanized_sequence(msg, ["раз", "два"], 0, 0)
        msg.reply.assert_awaited_once_with("раз")
        msg.answer.assert_awaited_once_with("два")
        self.assertEqual(sleep.await_count, 2)

    async def test_chat_action_failure_does_not_block_parts(self) -> None:
        from app.handlers._helpers import reply_humanized_sequence

        msg = _fake_message(text="x")
        msg.bot.send_chat_action = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("app.handlers._helpers.asyncio.sleep", new=AsyncMock()):
            await reply_humanized_sequence(msg, ["раз", "два"], 0, 0)
        msg.reply.assert_awaited_once_with("раз")
        msg.answer.assert_awaited_once_with("два")

    async def test_empty_parts_are_skipped(self) -> None:
        from app.handlers._helpers import reply_humanized_sequence

        msg = _fake_message(text="x")
        with patch("app.handlers._helpers.asyncio.sleep", new=AsyncMock()):
            await reply_humanized_sequence(msg, ["", "два"], 0, 0)
        msg.reply.assert_awaited_once_with("два")
        msg.answer.assert_not_awaited()


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


class TestStripLeadingBotVocative(unittest.TestCase):
    aliases = frozenset({"pepe", "пепе"})

    def test_strips_leading_alias_with_separators(self) -> None:
        from app.handlers.learning import strip_leading_bot_vocative

        for text, expected in (
            ("Пепе, расскажи анекдот", "расскажи анекдот"),
            ("pepe: what is up", "what is up"),
            ("Пепе — скажи что-нибудь", "скажи что-нибудь"),
            ("  пепе,  привет всем", "привет всем"),
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    strip_leading_bot_vocative(text, self.aliases), expected
                )

    def test_strips_leading_alias_without_separator(self) -> None:
        # SIM-8: a bare "<alias> ..." address is as common as the comma form;
        # unstripped it taught the corpus the bot's own name.
        from app.handlers.learning import strip_leading_bot_vocative

        for text, expected in (
            ("Пепе хороший бот", "хороший бот"),
            ("пепе кто гнойный пидор", "кто гнойный пидор"),
            ("  pepe   what is up", "what is up"),
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    strip_leading_bot_vocative(text, self.aliases), expected
                )

    def test_preserves_bare_alias_with_no_content(self) -> None:
        from app.handlers.learning import strip_leading_bot_vocative

        for text in ("Пепе", "пепе  ", " pepe"):
            with self.subTest(text=text):
                self.assertEqual(strip_leading_bot_vocative(text, self.aliases), text)

    def test_preserves_mid_sentence_alias_and_other_vocatives(self) -> None:
        from app.handlers.learning import strip_leading_bot_vocative

        for text in ("Ребята, привет", "скажи пепе, что делать", "Москва, я люблю"):
            with self.subTest(text=text):
                self.assertEqual(strip_leading_bot_vocative(text, self.aliases), text)

    def test_empty_aliases_is_noop(self) -> None:
        from app.handlers.learning import strip_leading_bot_vocative

        text = "Пепе, привет"
        self.assertEqual(strip_leading_bot_vocative(text, frozenset()), text)


class TestLearningHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        mask_patcher = patch(
            "app.core.response_generator.mask_chat_id",
            return_value="chat",
        )
        mask_patcher.start()
        self.addCleanup(mask_patcher.stop)

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
            "fuzzy_context_casefold": False,
            "reply_context_bias": 1.8,
            "reply_context_start_bias": 2.2,
            "context_start_affinity": 3.0,
            "max_reply_chars": 280,
            "max_reply_tokens": 45,
            "randomness_strength": 0.0,
            "repetition_penalty_strength": 1.0,
            "recent_reply_penalty_strength": 1.0,
            "length_mode_weights": (0.25, 0.55, 0.2),
            "length_context_adaptation": 0.0,
            "markov_order": 3,
            "enable_backoff": True,
        }
        values.update(overrides)
        return _fake_state(**values)

    async def test_single_token_message_is_not_learned(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="hello")
        learning_service = AsyncMock()
        generator = _traced_generator()
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
                _pivo_stub(),
            )

        learning_service.record_message.assert_not_called()
        msg.reply.assert_not_awaited()

    async def test_vocative_stripped_and_pii_redacted_before_learning(self) -> None:
        from app.handlers.learning import on_text_message

        # Leading bot vocative + a phone number: the vocative is stripped from the
        # text fed to record_message, and the phone is redacted out of the model
        # tokens (via sanitize_text), keeping the corpus clean.
        msg = _fake_message(text="pepe, мой телефон +380441234567 ладно")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
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
                _pivo_stub(),
            )

        learning_service.record_message.assert_awaited_once()
        kwargs = learning_service.record_message.await_args.kwargs
        self.assertEqual(kwargs["raw_text"], "мой телефон +380441234567 ладно")
        self.assertEqual(kwargs["tokens"], ["мой", "телефон", "ладно"])

    async def test_threshold_crossing_message_is_learned_without_replying_to_itself(
        self,
    ) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe threshold crossing")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=9)
        learning_service.record_message = AsyncMock(return_value=11)
        generator = _traced_generator()
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
                _pivo_stub(),
            )

        learning_service.get_token_volume.assert_awaited_once_with(msg.chat.id)
        learning_service.record_message.assert_awaited_once()
        generator.generate_text.assert_not_awaited()
        from app.presentation.fallback_phrases import (
            LATE_NIGHT_FALLBACK_PHRASES,
            NOT_ENOUGH_DATA_PHRASES,
        )

        msg.reply.assert_awaited_once()
        # The handler injects the wall clock, so in the small hours the late-night
        # pool is a valid extension (S4). Accept either so the assertion does not
        # depend on the time the CI run happens to execute.
        self.assertIn(
            msg.reply.await_args.args[0],
            set(NOT_ENOUGH_DATA_PHRASES) | set(LATE_NIGHT_FALLBACK_PHRASES),
        )

    async def test_current_incoming_message_copy_is_rejected(self) -> None:
        from app.handlers.learning import on_text_message

        # SIM-8: the leading alias is stripped before learning, so the echo
        # gate compares against the stripped text ("ответь мне").
        msg = _fake_message(text="pepe ответь мне")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=["ответь мне", "Нормально"] + [""] * 8
        )
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
                _pivo_stub(),
            )

        self.assertEqual(generator.generate_text.await_count, 10)
        learning_service.record_message.assert_awaited_once()
        msg.reply.assert_awaited_once_with("Нормально")

    async def test_message_is_persisted_when_cooldown_blocks_reply(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="обычное сообщение")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        generator = _traced_generator()
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
                _pivo_stub(),
            )

        learning_service.record_message.assert_awaited_once()
        generator.generate_text.assert_not_awaited()
        msg.reply.assert_not_awaited()

    async def test_message_is_persisted_when_generation_fails(self) -> None:
        from app.core.response_generator import GENERATION_ATTEMPT_BUDGET
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe generate something")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=103)
        generator = _traced_generator()
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
                _pivo_stub(),
            )

        learning_service.record_message.assert_awaited_once()
        self.assertEqual(
            generator.generate_text.await_count,
            GENERATION_ATTEMPT_BUDGET,
        )
        self.assertTrue(
            all(
                call.kwargs["attempt_budget"] == 1
                for call in generator.generate_text.await_args_list
            )
        )
        generation_rngs = [
            call.kwargs["rng"] for call in generator.generate_text.await_args_list
        ]
        self.assertTrue(all(rng is generation_rngs[0] for rng in generation_rngs))
        from app.presentation.fallback_phrases import (
            GENERATION_FAILED_PHRASES,
            LATE_NIGHT_FALLBACK_PHRASES,
        )

        msg.reply.assert_awaited_once()
        # See note above: the wall clock makes the late-night pool a valid
        # extension in the small hours, so accept either pool time-independently.
        self.assertIn(
            msg.reply.await_args.args[0],
            set(GENERATION_FAILED_PHRASES) | set(LATE_NIGHT_FALLBACK_PHRASES),
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
        generator = _traced_generator()
        # Empty strings after each verbatim candidate make the extension pass
        # fail, preserving the reject-and-retry flow this test exercises.
        generator.generate_text = AsyncMock(
            side_effect=[
                "один два три четыре",
                "",
                "один два три четыре",
                "",
                "один два три четыре",
                "",
                "один два три четыре",
                "",
                "новый ответ теперь",
            ]
            + [""] * 6
        )
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            fuzzy_context_casefold=False,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            context_start_affinity=3.0,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.0,
            repetition_penalty_strength=1.0,
            recent_reply_penalty_strength=1.0,
            length_mode_weights=(0.25, 0.55, 0.2),
            length_context_adaptation=0.0,
            markov_order=3,
            enable_backoff=True,
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
                _pivo_stub(),
            )

        # 10 attempt walks + 4 failed extension walks after each verbatim hit.
        self.assertEqual(generator.generate_text.await_count, 14)
        msg.reply.assert_awaited_once_with("новый ответ теперь")

    async def test_short_reply_skips_training_prefix_filter(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=True)
        generator = _traced_generator()
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
            fuzzy_context_casefold=False,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            context_start_affinity=3.0,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.0,
            repetition_penalty_strength=1.0,
            recent_reply_penalty_strength=1.0,
            length_mode_weights=(0.25, 0.55, 0.2),
            length_context_adaptation=0.0,
            markov_order=3,
            enable_backoff=True,
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
                _pivo_stub(),
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
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=["Привет", "Нормально"] + [""] * 8
        )
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            recent_short_replies={msg.chat.id: deque(["привет"], maxlen=5)},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            fuzzy_context_casefold=False,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            context_start_affinity=3.0,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.0,
            repetition_penalty_strength=1.0,
            recent_reply_penalty_strength=1.0,
            length_mode_weights=(0.25, 0.55, 0.2),
            length_context_adaptation=0.0,
            markov_order=3,
            enable_backoff=True,
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
                _pivo_stub(),
            )

        self.assertEqual(generator.generate_text.await_count, 10)
        msg.reply.assert_awaited_once_with("Нормально")
        self.assertEqual(
            list(state.recent_short_replies[msg.chat.id]),
            ["привет", "нормально"],
        )

    async def test_sent_reply_is_recorded_for_full_anti_repeat(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            return_value="новый развёрнутый ответ бота."
        )
        state = self._reply_state(recent_replies={})

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("новый развёрнутый ответ бота.")
        self.assertEqual(
            list(state.recent_replies[msg.chat.id]),
            ["новый развёрнутый ответ бота"],
        )

    async def test_mood_enabled_tracks_state_and_passes_modifiers(self) -> None:
        from app.core.mood import MoodConfig, modifiers_for_mood
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        state = self._reply_state(recent_replies={})
        state.mood_enabled = True
        state.chat_mood = {}
        state.mood_modulation_strength = 1.0
        state.mood_config = MagicMock(
            return_value=MoodConfig(
                ewma_alpha=0.3,
                lively_rate_per_min=12.0,
                sleepy_rate_per_min=2.0,
                heated_intensity=0.4,
                max_rate_per_min=120.0,
            )
        )

        with (
            patch("app.handlers.learning.ResponseGenerator") as response_gen_cls,
            patch("app.handlers.learning.mask_chat_id", return_value="chat"),
        ):
            response_gen_cls.return_value.generate = AsyncMock(return_value="ответ")
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        # Mood was tracked for the chat...
        self.assertIn(msg.chat.id, state.chat_mood)
        tracked = state.chat_mood[msg.chat.id]
        self.assertIn(tracked.mood, ("sleepy", "calm", "lively", "heated"))
        # ...and the matching modifiers were handed to the generator.
        _, kwargs = response_gen_cls.call_args
        self.assertEqual(
            kwargs["mood_modifiers"], modifiers_for_mood(tracked.mood, 1.0)
        )

    def _director_state(self, **overrides: object) -> MagicMock:
        from app.core.mood import MoodConfig

        values: dict[str, object] = {
            "reply_director_enabled": True,
            "mood_enabled": False,
            "chat_mood": {},
            "recent_reply_times": {},
            "mood_modulation_strength": 1.0,
            "mood_lively_rate_per_min": 12.0,
            "reply_probability_min": 0.02,
            "reply_probability_max": 0.30,
            "reply_burst_boost_sec": 180,
            "reply_burst_boost_mult": 2.0,
            "reply_burst_suppress_sec": 600,
            "reply_burst_suppress_mult": 0.5,
            "reply_max_per_hour": 20,
            "min_cooldown_sec": 0,
            "recent_replies": {},
        }
        values.update(overrides)
        state = self._reply_state(**values)
        state.mood_config = lambda: MoodConfig(
            ewma_alpha=0.3,
            lively_rate_per_min=12.0,
            sleepy_rate_per_min=2.0,
            heated_intensity=0.4,
            max_rate_per_min=120.0,
        )
        return state

    async def test_director_hourly_cap_blocks_unprompted_reply(self) -> None:
        from collections import deque

        from app.handlers.learning import on_text_message

        # Unprompted message (no alias, no reply-to) in a chat that already hit
        # its per-hour reply budget: the director must gate the reply, but the
        # message is still learned.
        msg = _fake_message(text="просто болтаю тут в чате")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        generator = _traced_generator()
        state = self._director_state(
            reply_max_per_hour=20,
            recent_reply_times={msg.chat.id: deque([10000.0] * 20)},
        )

        with (
            patch("app.handlers.learning.mask_chat_id", return_value="chat"),
            patch("app.handlers.learning.time.monotonic", return_value=10000.0),
            patch("app.handlers.learning.random.random", return_value=0.0),
        ):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        generator.generate_text.assert_not_awaited()
        msg.reply.assert_not_awaited()
        learning_service.record_message.assert_awaited_once()

    async def test_director_allows_reply_when_momentum_clears_probability(self) -> None:
        from app.handlers.learning import on_text_message

        # random forced to 0.0 → any positive momentum-derived probability fires.
        msg = _fake_message(text="просто болтаю тут в чате")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="ответ бота готов")
        state = self._director_state(recent_reply_times={})

        with (
            patch("app.handlers.learning.mask_chat_id", return_value="chat"),
            patch("app.handlers.learning.time.monotonic", return_value=10000.0),
            patch("app.handlers.learning.random.random", return_value=0.0),
        ):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("ответ бота готов")

    async def test_emojis_recorded_from_message(self) -> None:
        from app.handlers.learning import on_text_message

        # A message carrying emojis feeds the per-chat emoji stats even though the
        # word model drops them. Recorded regardless of whether a reply fires.
        msg = _fake_message(text="хаха 😂 огонь 🔥")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        state = self._reply_state(emoji_append_chance=0.15)

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.record_emojis.assert_awaited_once_with(
            msg.chat.id, {"😂": 1, "🔥": 1}
        )

    async def test_reply_appends_chat_emoji(self) -> None:
        from app.handlers.learning import on_text_message

        # With the channel certain (chance 1.0) and stats present, the reply ends
        # on a frequency-sampled emoji from this chat.
        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_emoji_stats = AsyncMock(return_value={"🍺": 5})
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="ответ бота готов")
        state = self._reply_state(recent_replies={}, emoji_append_chance=1.0)

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("ответ бота готов 🍺")

    async def test_learnable_message_records_hot_ngrams(self) -> None:
        from app.core.hot_ngrams import extract_content_ngrams
        from app.core.markov import tokenize
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="крутой бобёр пришёл")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        state = self._reply_state(hot_ngram_seed_chance=0.05)

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        expected = extract_content_ngrams(
            tokenize("крутой бобёр пришёл", normalize_lower=False)
        )
        learning_service.record_hot_ngrams.assert_awaited_once_with(
            msg.chat.id, expected
        )

    async def test_hot_ngram_recording_disabled_at_zero_chance(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="крутой бобёр пришёл")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        state = self._reply_state()  # hot_ngram_seed_chance = 0.0 default

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.record_hot_ngrams.assert_not_awaited()

    async def test_unprompted_reply_seeded_on_roll(self) -> None:
        from app.handlers.learning import on_text_message

        # No mention; reply_probability 1.0 and the patched roll 0.0 win both
        # the reply gate and the seed gate.
        msg = _fake_message(text="обычное сообщение в чате")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_hot_ngrams = AsyncMock(
            return_value=[("крутой", "бобёр")]
        )
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="ответ бота готов")
        state = self._reply_state(
            reply_probability=1.0,
            hot_ngram_seed_chance=1.0,
            recent_replies={},
        )

        with (
            patch("app.handlers.learning.mask_chat_id", return_value="chat"),
            patch("app.handlers.learning.random.random", return_value=0.0),
        ):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.get_hot_ngrams.assert_awaited_once_with(
            msg.chat.id,
            min_count=state.hot_ngram_min_count,
            recency_share=state.hot_ngram_recency_share,
        )
        first_call = generator.generate_text.await_args_list[0]
        self.assertEqual(first_call.kwargs["seed_tokens"], ["крутой", "бобёр"])
        msg.reply.assert_awaited_once()

    async def test_mention_reply_never_seeded(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="ответ бота готов")
        state = self._reply_state(
            hot_ngram_seed_chance=1.0,
            recent_replies={},
        )

        with (
            patch("app.handlers.learning.mask_chat_id", return_value="chat"),
            patch("app.handlers.learning.random.random", return_value=0.0),
        ):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.get_hot_ngrams.assert_not_awaited()
        first_call = generator.generate_text.await_args_list[0]
        self.assertIsNone(first_call.kwargs["seed_tokens"])
        msg.reply.assert_awaited_once()

    async def test_no_hot_ngrams_means_no_seed(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="обычное сообщение в чате")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_hot_ngrams = AsyncMock(return_value=[])
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="ответ бота готов")
        state = self._reply_state(
            reply_probability=1.0,
            hot_ngram_seed_chance=1.0,
            recent_replies={},
        )

        with (
            patch("app.handlers.learning.mask_chat_id", return_value="chat"),
            patch("app.handlers.learning.random.random", return_value=0.0),
        ):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        first_call = generator.generate_text.await_args_list[0]
        self.assertIsNone(first_call.kwargs["seed_tokens"])
        msg.reply.assert_awaited_once()

    async def test_rare_event_false_start_sends_filler_then_reply(self) -> None:
        from app.core.reply_flavor import FALSE_START_FILLERS
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="настоящий ответ бота")
        state = self._reply_state(
            recent_replies={},
            false_start_chance=1.0,
            rare_events_today={},
        )

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once()
        self.assertIn(msg.reply.await_args.args[0], FALSE_START_FILLERS)
        msg.answer.assert_awaited_once_with("настоящий ответ бота")
        self.assertEqual(state.rare_events_today[msg.chat.id][1], 1)

    async def test_rare_event_respects_daily_cap(self) -> None:
        from datetime import UTC, datetime

        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="настоящий ответ бота")
        state = self._reply_state(
            recent_replies={},
            false_start_chance=1.0,
            # UTC, как в хендлере: локальная дата около полуночи дала бы флаки.
            rare_events_today={
                msg.chat.id: (datetime.now(UTC).date().isoformat(), 3)
            },
            rare_event_daily_cap=3,
        )

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("настоящий ответ бота")
        msg.answer.assert_not_awaited()

    async def test_zero_chances_send_plain_reply(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="настоящий ответ бота")
        state = self._reply_state(recent_replies={})  # both chances 0.0 default

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("настоящий ответ бота")
        msg.answer.assert_not_awaited()

    async def test_recent_full_reply_is_retried_instead_of_sent(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=[
                "недавно отправленный полный ответ.",
                "совсем свежий полный ответ",
            ]
            + [""] * 8
        )
        state = self._reply_state(
            recent_replies={
                msg.chat.id: deque(
                    ["недавно отправленный полный ответ"], maxlen=20
                )
            },
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
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("совсем свежий полный ответ")

    async def test_handler_increases_randomness_on_generation_retries(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь нормально")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(
            side_effect=[True, True, False]
        )
        generator = _traced_generator()
        # Empty strings after each verbatim candidate make the extension pass
        # fail, preserving the reject-and-retry escalation this test measures.
        generator.generate_text = AsyncMock(
            side_effect=[
                "первый ответ заметно длиннее",
                "",
                "второй ответ заметно длиннее",
                "",
                "третий ответ заметно длиннее",
            ]
            + [""] * 7
        )
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            fuzzy_context_casefold=False,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            context_start_affinity=3.0,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.5,
            repetition_penalty_strength=1.0,
            recent_reply_penalty_strength=1.0,
            length_mode_weights=(0.25, 0.55, 0.2),
            length_context_adaptation=0.0,
            markov_order=3,
            enable_backoff=True,
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
                _pivo_stub(),
            )

        strengths = [
            call.kwargs["randomness_strength"]
            for call in generator.generate_text.await_args_list
            # Extension walks (max_tokens=10) run at base strength; only the
            # attempt walks carry the escalation this test measures.
            if call.kwargs.get("max_tokens") != 10
        ]
        self.assertEqual(strengths, sorted(strengths))
        self.assertEqual(strengths[0], 0.5)
        self.assertGreater(strengths[-1], strengths[0])
        msg.reply.assert_awaited_once_with("третий ответ заметно длиннее")


# ---------------------------------------------------------------------------
# learning.py — mention anti-flood gate (N1)
# ---------------------------------------------------------------------------

class TestMentionCooldownGate(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        mask_patcher = patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        )
        mask_patcher.start()
        self.addCleanup(mask_patcher.stop)

    def _state(self, **overrides: object) -> MagicMock:
        values: dict[str, object] = {
            "normalize_lower": False,
            "learned_messages": {},
            "recent_short_replies": {},
            "min_tokens_for_model": 10,
            "min_cooldown_sec": 0,
            "last_reply_ts": {},
            "reply_probability": 0.0,
            "use_reply_context": False,
            "fuzzy_context_casefold": False,
            "reply_context_bias": 1.8,
            "reply_context_start_bias": 2.2,
            "context_start_affinity": 3.0,
            "max_reply_chars": 280,
            "max_reply_tokens": 45,
            "randomness_strength": 0.0,
            "repetition_penalty_strength": 1.0,
            "recent_reply_penalty_strength": 1.0,
            "length_mode_weights": (0.25, 0.55, 0.2),
            "length_context_adaptation": 0.0,
            "markov_order": 3,
            "enable_backoff": True,
            "mention_cooldown_sec": 30,
        }
        values.update(overrides)
        return _fake_state(**values)

    def _services(self) -> tuple[AsyncMock, AsyncMock]:
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="сгенерированный ответ бота")
        return learning_service, generator

    async def _dispatch(self, msg: MagicMock, learning_service: AsyncMock,
                        generator: AsyncMock, state: MagicMock) -> None:
        from app.handlers.learning import on_text_message

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

    async def test_first_mention_replies_and_records_timestamp(self) -> None:
        learning_service, generator = self._services()
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once()
        state.note_mention_reply.assert_called_once_with(
            msg.chat.id, msg.from_user.id, ANY
        )

    async def test_mention_within_cooldown_is_demoted_to_unprompted_path(self) -> None:
        import time as time_module

        learning_service, generator = self._services()
        state = self._state()
        msg = _fake_message(text="pepe расскажи ещё")
        state.last_mention_reply_ts = {
            (msg.chat.id, msg.from_user.id): time_module.monotonic()
        }

        await self._dispatch(msg, learning_service, generator, state)

        # Gated mention: no guaranteed reply (reply_probability=0 → silent),
        # but the message is still learned as usual.
        msg.reply.assert_not_awaited()
        state.note_mention_reply.assert_not_called()
        learning_service.record_message.assert_awaited_once()

    async def test_gate_disabled_keeps_guaranteed_mention_reply(self) -> None:
        import time as time_module

        learning_service, generator = self._services()
        state = self._state(mention_cooldown_sec=0)
        msg = _fake_message(text="pepe расскажи снова")
        state.last_mention_reply_ts = {
            (msg.chat.id, msg.from_user.id): time_module.monotonic()
        }

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once()

    async def test_other_user_is_not_gated_by_someone_elses_mention(self) -> None:
        import time as time_module

        learning_service, generator = self._services()
        state = self._state()
        msg = _fake_message(text="pepe расскажи", user_id=2)
        state.last_mention_reply_ts = {
            (msg.chat.id, 1): time_module.monotonic()
        }

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once()


class TestUserQuirks(unittest.IsolatedAsyncioTestCase):
    """L2: vocative quirks for regulars whose direct address got answered."""

    def setUp(self) -> None:
        mask_patcher = patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        )
        mask_patcher.start()
        self.addCleanup(mask_patcher.stop)

    def _state(self, **overrides: object) -> MagicMock:
        values: dict[str, object] = {
            "normalize_lower": False,
            "learned_messages": {},
            "recent_short_replies": {},
            "min_tokens_for_model": 10,
            "min_cooldown_sec": 0,
            "last_reply_ts": {},
            "reply_probability": 0.0,
            "use_reply_context": False,
            "fuzzy_context_casefold": False,
            "reply_context_bias": 1.8,
            "reply_context_start_bias": 2.2,
            "context_start_affinity": 3.0,
            "max_reply_chars": 280,
            "max_reply_tokens": 45,
            "randomness_strength": 0.0,
            "repetition_penalty_strength": 1.0,
            "recent_reply_penalty_strength": 1.0,
            "length_mode_weights": (0.25, 0.55, 0.2),
            "length_context_adaptation": 0.0,
            "markov_order": 3,
            "enable_backoff": True,
            "recent_replies": {},
            # Chance 1.0 keeps the roll deterministic (random() < 1.0 always).
            "user_quirk_chance": 1.0,
            "user_quirk_min_interactions": 25,
        }
        values.update(overrides)
        return _fake_state(**values)

    def _services(
        self, *, interactions: int = 25
    ) -> tuple[AsyncMock, AsyncMock]:
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_user_interaction_count = AsyncMock(
            return_value=interactions
        )
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="сгенерированный ответ бота")
        return learning_service, generator

    async def _dispatch(self, msg: MagicMock, learning_service: AsyncMock,
                        generator: AsyncMock, state: MagicMock) -> None:
        from app.handlers.learning import on_text_message

        with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

    @staticmethod
    def _today_iso() -> str:
        from datetime import UTC, datetime

        # UTC, как в хендлере: локальная дата около полуночи дала бы флаки.
        return datetime.now(UTC).date().isoformat()

    async def test_regular_gets_vocative_as_separate_first_message(self) -> None:
        from app.presentation.fallback_phrases import USER_QUIRK_VOCATIVES

        learning_service, generator = self._services(interactions=25)
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once()
        self.assertIn(msg.reply.await_args.args[0], USER_QUIRK_VOCATIVES)
        msg.answer.assert_awaited_once_with("сгенерированный ответ бота")
        self.assertEqual(
            state.last_user_quirk_day,
            {(msg.chat.id, msg.from_user.id): self._today_iso()},
        )
        learning_service.record_user_interaction.assert_awaited_once_with(
            msg.chat.id, msg.from_user.id
        )

    async def test_below_threshold_sends_plain_reply(self) -> None:
        learning_service, generator = self._services(interactions=24)
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once_with("сгенерированный ответ бота")
        msg.answer.assert_not_awaited()
        # Below threshold: no daily stamp, so a later regular can still quirk.
        self.assertEqual(state.last_user_quirk_day, {})

    async def test_same_user_same_day_is_suppressed(self) -> None:
        learning_service, generator = self._services(interactions=100)
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")
        state.last_user_quirk_day = {
            (msg.chat.id, msg.from_user.id): self._today_iso()
        }

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once_with("сгенерированный ответ бота")
        msg.answer.assert_not_awaited()
        # The daily gate is checked before the chance roll and the DB read.
        learning_service.get_user_interaction_count.assert_not_awaited()

    async def test_unprompted_reply_is_never_quirked(self) -> None:
        learning_service, generator = self._services(interactions=100)
        state = self._state(reply_probability=1.0)
        msg = _fake_message(text="просто сообщение в чат без обращения")

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once_with("сгенерированный ответ бота")
        msg.answer.assert_not_awaited()
        learning_service.record_user_interaction.assert_not_awaited()
        learning_service.get_user_interaction_count.assert_not_awaited()

    async def test_not_enough_data_fallback_counts_but_never_quirks(self) -> None:
        learning_service, generator = self._services(interactions=100)
        learning_service.get_token_volume = AsyncMock(return_value=5)
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        # The user did interact -> counter bumped; but a fallback answer never
        # carries a vocative.
        learning_service.record_user_interaction.assert_awaited_once_with(
            msg.chat.id, msg.from_user.id
        )
        learning_service.get_user_interaction_count.assert_not_awaited()
        msg.reply.assert_awaited_once()
        msg.answer.assert_not_awaited()
        self.assertEqual(state.last_user_quirk_day, {})

    async def test_generation_failed_fallback_counts_but_never_quirks(self) -> None:
        learning_service, generator = self._services(interactions=100)
        generator.generate_text = AsyncMock(return_value="")
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        learning_service.record_user_interaction.assert_awaited_once_with(
            msg.chat.id, msg.from_user.id
        )
        learning_service.get_user_interaction_count.assert_not_awaited()
        msg.reply.assert_awaited_once()
        msg.answer.assert_not_awaited()
        self.assertEqual(state.last_user_quirk_day, {})

    async def test_demoted_mention_neither_counts_nor_quirks(self) -> None:
        import time as time_module

        learning_service, generator = self._services(interactions=100)
        state = self._state(mention_cooldown_sec=30)
        msg = _fake_message(text="pepe расскажи ещё")
        state.last_mention_reply_ts = {
            (msg.chat.id, msg.from_user.id): time_module.monotonic()
        }

        await self._dispatch(msg, learning_service, generator, state)

        # Demoted to the unprompted path (reply_probability 0.0 -> silence):
        # regular status cannot be farmed faster than the mention cooldown.
        learning_service.record_user_interaction.assert_not_awaited()
        learning_service.get_user_interaction_count.assert_not_awaited()
        msg.reply.assert_not_awaited()

    async def test_quirked_reply_skips_rare_event_roll(self) -> None:
        from app.presentation.fallback_phrases import USER_QUIRK_VOCATIVES

        learning_service, generator = self._services(interactions=25)
        # false_start_chance 1.0 would otherwise always fire a filler.
        state = self._state(false_start_chance=1.0, rare_events_today={})
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        # One shape break per reply: vocative + reply, no filler, and the
        # rare-event daily budget is untouched.
        msg.reply.assert_awaited_once()
        self.assertIn(msg.reply.await_args.args[0], USER_QUIRK_VOCATIVES)
        msg.answer.assert_awaited_once_with("сгенерированный ответ бота")
        self.assertEqual(state.rare_events_today, {})

    async def test_zero_chance_disables_channel_entirely(self) -> None:
        learning_service, generator = self._services(interactions=100)
        state = self._state(user_quirk_chance=0.0)
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        # Master gate: no counter writes, no reads at all.
        learning_service.record_user_interaction.assert_not_awaited()
        learning_service.get_user_interaction_count.assert_not_awaited()
        msg.reply.assert_awaited_once_with("сгенерированный ответ бота")

    async def test_bot_senders_never_feed_counters(self) -> None:
        # Anonymous admins post as GroupAnonymousBot (a bot account): the
        # handler's early is_bot return must keep them out of the counters.
        learning_service, generator = self._services(interactions=100)
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь", is_bot=True)

        await self._dispatch(msg, learning_service, generator, state)

        learning_service.record_user_interaction.assert_not_awaited()
        msg.reply.assert_not_awaited()
