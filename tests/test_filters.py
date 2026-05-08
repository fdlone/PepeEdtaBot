"""Tests for GroupOnly, AdminOrOwner filters and ThrottlingMiddleware."""
from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

from aiogram.enums import ChatType


def _make_message(chat_type: ChatType, user_id: int = 1, chat_id: int = 100) -> MagicMock:
    msg = MagicMock()
    msg.chat.type = chat_type
    msg.chat.id = chat_id
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    return msg


def _make_settings(owner_id=None) -> MagicMock:
    s = MagicMock()
    s.owner_id = owner_id
    return s


def _make_admin_member(user_id: int) -> MagicMock:
    m = MagicMock()
    m.user.id = user_id
    return m


class TestGroupOnly(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from app.filters import GroupOnly
        self.f = GroupOnly()

    async def test_group_passes(self) -> None:
        msg = _make_message(ChatType.GROUP)
        self.assertTrue(await self.f(msg))

    async def test_supergroup_passes(self) -> None:
        msg = _make_message(ChatType.SUPERGROUP)
        self.assertTrue(await self.f(msg))

    async def test_private_blocked(self) -> None:
        msg = _make_message(ChatType.PRIVATE)
        self.assertFalse(await self.f(msg))

    async def test_channel_blocked(self) -> None:
        msg = _make_message(ChatType.CHANNEL)
        self.assertFalse(await self.f(msg))


class TestAdminOrOwner(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from app.filters import AdminOrOwner
        self.f = AdminOrOwner()

    async def test_owner_allowed(self) -> None:
        msg = _make_message(ChatType.GROUP, user_id=42)
        bot = AsyncMock()
        settings = _make_settings(owner_id=42)
        self.assertTrue(await self.f(msg, bot, settings))
        bot.get_chat_administrators.assert_not_called()

    async def test_chat_admin_allowed(self) -> None:
        msg = _make_message(ChatType.GROUP, user_id=7)
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[_make_admin_member(7)])
        settings = _make_settings(owner_id=None)
        self.assertTrue(await self.f(msg, bot, settings))

    async def test_non_admin_blocked(self) -> None:
        msg = _make_message(ChatType.GROUP, user_id=99)
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[_make_admin_member(7)])
        settings = _make_settings(owner_id=None)
        self.assertFalse(await self.f(msg, bot, settings))

    async def test_no_from_user_blocked(self) -> None:
        msg = _make_message(ChatType.GROUP)
        msg.from_user = None
        bot = AsyncMock()
        settings = _make_settings(owner_id=1)
        self.assertFalse(await self.f(msg, bot, settings))

    async def test_api_error_falls_back_to_denied(self) -> None:
        msg = _make_message(ChatType.GROUP, user_id=5)
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(side_effect=Exception("network error"))
        settings = _make_settings(owner_id=None)
        self.assertFalse(await self.f(msg, bot, settings))

    async def test_owner_bypasses_admin_check_even_when_api_would_fail(self) -> None:
        msg = _make_message(ChatType.GROUP, user_id=1)
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(side_effect=Exception("should not be called"))
        settings = _make_settings(owner_id=1)
        self.assertTrue(await self.f(msg, bot, settings))


class TestThrottlingMiddleware(unittest.IsolatedAsyncioTestCase):
    def _make_cmd_message(self, text: str, user_id: int = 1, chat_id: int = 100) -> MagicMock:
        msg = MagicMock()
        msg.text = text
        msg.from_user = MagicMock()
        msg.from_user.id = user_id
        msg.chat.id = chat_id
        return msg

    def _make_middleware(self) -> object:
        from app.middlewares import ThrottlingMiddleware
        return ThrottlingMiddleware(limits={"pivo": 30.0, "clear": 10.0})

    async def test_first_call_passes(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/pivo")
        result = await mw(handler, msg, {})
        self.assertEqual(result, "ok")
        handler.assert_awaited_once()

    async def test_second_call_throttled(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/pivo")
        await mw(handler, msg, {})
        result = await mw(handler, msg, {})
        self.assertIsNone(result)
        handler.assert_awaited_once()

    async def test_different_users_not_throttled(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg1 = self._make_cmd_message("/pivo", user_id=1)
        msg2 = self._make_cmd_message("/pivo", user_id=2)
        await mw(handler, msg1, {})
        result = await mw(handler, msg2, {})
        self.assertEqual(result, "ok")

    async def test_different_commands_not_throttled(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg_pivo = self._make_cmd_message("/pivo")
        msg_clear = self._make_cmd_message("/clear")
        await mw(handler, msg_pivo, {})
        result = await mw(handler, msg_clear, {})
        self.assertEqual(result, "ok")

    async def test_unthrottled_command_always_passes(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/ping")
        for _ in range(3):
            result = await mw(handler, msg, {})
            self.assertEqual(result, "ok")

    async def test_non_command_message_passes(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("hello world")
        result = await mw(handler, msg, {})
        self.assertEqual(result, "ok")

    async def test_bot_username_suffix_stripped(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/pivo@mybot")
        await mw(handler, msg, {})
        result = await mw(handler, msg, {})
        self.assertIsNone(result)

    async def test_no_from_user_passes(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/pivo")
        msg.from_user = None
        result = await mw(handler, msg, {})
        self.assertEqual(result, "ok")
