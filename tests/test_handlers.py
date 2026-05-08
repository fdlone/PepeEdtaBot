"""Happy-path tests for all routers.

Each handler is called directly with faked Message / service objects.
No aiogram dispatcher is started; we only verify that handlers call
the right service methods and reply to the message.
"""
from __future__ import annotations

import unittest
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


# ---------------------------------------------------------------------------
# pivo.py
# ---------------------------------------------------------------------------

class TestPivoHandlers(unittest.IsolatedAsyncioTestCase):
    async def test_pivo_calls_build_and_replies(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        pivo_service.build_call_message = AsyncMock(return_value=("Выходи пить!", 2))
        state = _fake_state()
        await cmd_pivo(msg, pivo_service, state)
        pivo_service.build_call_message.assert_awaited_once()
        msg.reply.assert_awaited_once()

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
