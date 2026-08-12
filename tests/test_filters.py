"""Tests for GROUP_ONLY, AdminOrOwner filters and ThrottlingMiddleware."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestGroupOnly(unittest.TestCase):
    def setUp(self) -> None:
        from app.filters import GROUP_ONLY
        self.f = GROUP_ONLY

    def test_group_passes(self) -> None:
        msg = _make_message(ChatType.GROUP)
        self.assertTrue(self.f.resolve(msg))

    def test_supergroup_passes(self) -> None:
        msg = _make_message(ChatType.SUPERGROUP)
        self.assertTrue(self.f.resolve(msg))

    def test_private_blocked(self) -> None:
        msg = _make_message(ChatType.PRIVATE)
        self.assertFalse(self.f.resolve(msg))

    def test_channel_blocked(self) -> None:
        msg = _make_message(ChatType.CHANNEL)
        self.assertFalse(self.f.resolve(msg))


class TestAdminOrOwner(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from app.filters import AdminOrOwner, admin_or_owner

        admin_or_owner._admin_cache.clear()
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

    async def test_admin_lookup_is_cached_within_ttl(self) -> None:
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[_make_admin_member(7)])
        settings = _make_settings(owner_id=None)
        # Two different users in the same chat: only the first triggers an API call.
        self.assertTrue(await self.f(_make_message(ChatType.GROUP, user_id=7), bot, settings))
        self.assertFalse(await self.f(_make_message(ChatType.GROUP, user_id=99), bot, settings))
        bot.get_chat_administrators.assert_awaited_once()


class TestAdminCacheEviction(unittest.IsolatedAsyncioTestCase):
    """Growth policy for the one in-memory state that used to lack it."""

    async def asyncSetUp(self) -> None:
        from app.filters import admin_or_owner

        self.mod = admin_or_owner
        self.mod.reset_admin_cache()
        self.addCleanup(self.mod.reset_admin_cache)

    def _put(self, chat_id: int, cached_at: float) -> None:
        self.mod._admin_cache[chat_id] = (cached_at, frozenset({7}))

    async def test_stale_entries_are_dropped_not_just_overwritten(self) -> None:
        now = 10_000.0
        self._put(1, now - self.mod.ADMIN_CACHE_TTL_SECONDS - 1)
        self._put(2, now - 1)

        self.mod._prune_admin_cache(now)

        self.assertNotIn(1, self.mod._admin_cache)
        self.assertIn(2, self.mod._admin_cache)

    async def test_overflow_evicts_least_recently_cached(self) -> None:
        now = 10_000.0
        # All fresh, so only the cap can evict — oldest cached go first.
        for i in range(self.mod.ADMIN_CACHE_MAX_CHATS + 10):
            self._put(i, now - (self.mod.ADMIN_CACHE_TTL_SECONDS - 1) + i * 0.001)

        self.mod._prune_admin_cache(now)

        self.assertLessEqual(
            len(self.mod._admin_cache), self.mod.ADMIN_CACHE_MAX_CHATS
        )
        self.assertNotIn(0, self.mod._admin_cache)
        self.assertIn(self.mod.ADMIN_CACHE_MAX_CHATS + 9, self.mod._admin_cache)

    async def test_cache_stays_bounded_across_many_chats(self) -> None:
        from app.filters import AdminOrOwner

        f = AdminOrOwner()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[_make_admin_member(7)])
        settings = _make_settings(owner_id=None)

        for chat_id in range(self.mod.ADMIN_CACHE_MAX_CHATS + 200):
            msg = _make_message(ChatType.GROUP, user_id=7)
            msg.chat.id = chat_id
            await f(msg, bot, settings)

        self.assertLessEqual(
            len(self.mod._admin_cache),
            self.mod.ADMIN_CACHE_MAX_CHATS + self.mod._ADMIN_CACHE_CLEANUP_EVERY,
        )

    async def test_eviction_does_not_change_the_access_decision(self) -> None:
        from app.filters import AdminOrOwner

        f = AdminOrOwner()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[_make_admin_member(7)])
        settings = _make_settings(owner_id=None)
        msg = _make_message(ChatType.GROUP, user_id=7)

        self.assertTrue(await f(msg, bot, settings))
        self.mod.reset_admin_cache()
        # Evicted: the list is refetched and the verdict is the same.
        self.assertTrue(await f(msg, bot, settings))
        self.assertEqual(bot.get_chat_administrators.await_count, 2)

    async def test_eviction_keeps_denial_fail_closed(self) -> None:
        from app.filters import AdminOrOwner

        f = AdminOrOwner()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(side_effect=Exception("network error"))
        settings = _make_settings(owner_id=None)
        msg = _make_message(ChatType.GROUP, user_id=7)

        self.assertFalse(await f(msg, bot, settings))
        # A failed lookup must not be cached as "allowed" — or as anything.
        self.assertEqual(self.mod._admin_cache, {})
        self.assertFalse(await f(msg, bot, settings))


class TestOwnerOnly(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from app.filters import OwnerOnly
        self.f = OwnerOnly()

    async def test_owner_allowed(self) -> None:
        msg = _make_message(ChatType.GROUP, user_id=42)
        settings = _make_settings(owner_id=42)
        self.assertTrue(await self.f(msg, settings))

    async def test_chat_admin_blocked(self) -> None:
        # Unlike AdminOrOwner, being a chat admin must never be enough here.
        msg = _make_message(ChatType.GROUP, user_id=7)
        settings = _make_settings(owner_id=None)
        self.assertFalse(await self.f(msg, settings))

    async def test_other_user_blocked(self) -> None:
        msg = _make_message(ChatType.GROUP, user_id=99)
        settings = _make_settings(owner_id=42)
        self.assertFalse(await self.f(msg, settings))

    async def test_no_owner_configured_blocked(self) -> None:
        msg = _make_message(ChatType.GROUP, user_id=42)
        settings = _make_settings(owner_id=None)
        self.assertFalse(await self.f(msg, settings))

    async def test_no_from_user_blocked(self) -> None:
        msg = _make_message(ChatType.GROUP)
        msg.from_user = None
        settings = _make_settings(owner_id=1)
        self.assertFalse(await self.f(msg, settings))

    async def test_private_chat_still_allowed_for_owner(self) -> None:
        # OwnerOnly itself doesn't restrict chat type; /set additionally
        # chains GROUP_ONLY at the router level.
        msg = _make_message(ChatType.PRIVATE, user_id=42)
        settings = _make_settings(owner_id=42)
        self.assertTrue(await self.f(msg, settings))


class TestThrottlingMiddleware(unittest.IsolatedAsyncioTestCase):
    def _make_cmd_message(self, text: str, user_id: int = 1, chat_id: int = 100) -> MagicMock:
        from aiogram.types import Message
        msg = MagicMock()
        msg.__class__ = Message  # isinstance(msg, Message) == True
        msg.text = text
        msg.from_user = MagicMock()
        msg.from_user.id = user_id
        msg.chat.id = chat_id
        msg.reply = AsyncMock()
        return msg

    def _make_middleware(self, notify_on_throttle: set[str] | None = None) -> object:
        from app.middlewares import ThrottlingMiddleware
        return ThrottlingMiddleware(
            limits={"pivo": 30.0, "clear": 10.0},
            notify_on_throttle=notify_on_throttle,
        )

    async def test_first_call_passes(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/pivo")
        result = await mw(handler, msg, {})
        self.assertEqual(result, "ok")
        handler.assert_awaited_once()

    async def test_first_call_passes_even_if_process_started_recently(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/pivo")
        with patch("app.middlewares.throttling.time.monotonic", return_value=1.0):
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

    async def test_same_user_in_another_chat_not_throttled(self) -> None:
        # One noisy participant must not mute the bot for a different chat.
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg1 = self._make_cmd_message("/pivo", user_id=1, chat_id=100)
        msg2 = self._make_cmd_message("/pivo", user_id=1, chat_id=200)
        await mw(handler, msg1, {})
        result = await mw(handler, msg2, {})
        self.assertEqual(result, "ok")

    async def test_private_chat_is_throttled_like_a_group(self) -> None:
        # /config is reachable in a private chat, which is exactly where
        # nobody would notice it being looped.
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        private = self._make_cmd_message("/clear", user_id=7, chat_id=7)
        self.assertEqual(await mw(handler, private, {}), "ok")
        self.assertIsNone(await mw(handler, private, {}))

    async def test_different_commands_not_throttled(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg_pivo = self._make_cmd_message("/pivo")
        msg_clear = self._make_cmd_message("/clear")
        await mw(handler, msg_pivo, {})
        result = await mw(handler, msg_clear, {})
        self.assertEqual(result, "ok")

    async def test_clear_confirm_not_throttled_by_initial_clear_prompt(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg_clear = self._make_cmd_message("/clear")
        msg_confirm = self._make_cmd_message("/clear confirm")
        await mw(handler, msg_clear, {})
        result = await mw(handler, msg_confirm, {})
        self.assertEqual(result, "ok")
        self.assertEqual(handler.await_count, 2)

    async def test_repeated_clear_confirm_is_throttled(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg_confirm = self._make_cmd_message("/clear confirm")
        await mw(handler, msg_confirm, {})
        result = await mw(handler, msg_confirm, {})
        self.assertIsNone(result)
        handler.assert_awaited_once()

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

    async def test_command_addressed_to_another_bot_skips_throttling(self) -> None:
        # "/clear@OtherBot confirm" is another bot's traffic; it must neither
        # be throttled nor burn this bot's cooldown.
        from app.middlewares import ThrottlingMiddleware

        mw = ThrottlingMiddleware(limits={"clear": 10.0}, bot_username="MyBot")
        handler = AsyncMock(return_value="ok")
        foreign = self._make_cmd_message("/clear@OtherBot confirm")
        for _ in range(2):
            self.assertEqual(await mw(handler, foreign, {}), "ok")
        # And the cooldown for this bot's own /clear stays untouched.
        own = self._make_cmd_message("/clear confirm")
        self.assertEqual(await mw(handler, own, {}), "ok")

    async def test_command_addressed_to_this_bot_is_throttled_case_insensitively(
        self,
    ) -> None:
        from app.middlewares import ThrottlingMiddleware

        mw = ThrottlingMiddleware(limits={"clear": 10.0}, bot_username="MyBot")
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/clear@mybot")
        self.assertEqual(await mw(handler, msg, {}), "ok")
        self.assertIsNone(await mw(handler, msg, {}))
        handler.assert_awaited_once()

    async def test_handler_exception_does_not_burn_cooldown(self) -> None:
        # SQLITE_BUSY during /clear must not lock the user out for the window.
        mw = self._make_middleware()
        failing = AsyncMock(side_effect=RuntimeError("database is locked"))
        msg = self._make_cmd_message("/clear")
        with self.assertRaises(RuntimeError):
            await mw(failing, msg, {})
        handler = AsyncMock(return_value="ok")
        self.assertEqual(await mw(handler, msg, {}), "ok")

    async def test_no_from_user_passes(self) -> None:
        mw = self._make_middleware()
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/pivo")
        msg.from_user = None
        result = await mw(handler, msg, {})
        self.assertEqual(result, "ok")

    async def test_throttled_command_in_notify_set_replies_with_cooldown_message(
        self,
    ) -> None:
        mw = self._make_middleware(notify_on_throttle={"clear"})
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/clear")
        await mw(handler, msg, {})  # primes the cooldown
        result = await mw(handler, msg, {})
        self.assertIsNone(result)
        handler.assert_awaited_once()
        msg.reply.assert_awaited_once()
        reply_text = msg.reply.await_args.args[0]
        self.assertIn("Слишком часто", reply_text)
        self.assertIn("сек", reply_text)

    async def test_throttled_command_not_in_notify_set_stays_silent(self) -> None:
        mw = self._make_middleware(notify_on_throttle={"clear"})
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/pivo")
        await mw(handler, msg, {})  # primes the cooldown
        result = await mw(handler, msg, {})
        self.assertIsNone(result)
        msg.reply.assert_not_awaited()

    async def test_clear_confirm_throttled_also_notifies(self) -> None:
        mw = self._make_middleware(notify_on_throttle={"clear"})
        handler = AsyncMock(return_value="ok")
        msg_confirm = self._make_cmd_message("/clear confirm")
        await mw(handler, msg_confirm, {})  # primes
        result = await mw(handler, msg_confirm, {})
        self.assertIsNone(result)
        msg_confirm.reply.assert_awaited_once()

    async def test_repeated_throttled_attempts_notify_only_once_per_window(self) -> None:
        # N6: the throttle notification must not amplify a hammering user —
        # inside notify_cooldown_sec only the first throttled attempt replies.
        mw = self._make_middleware(notify_on_throttle={"clear"})
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/clear")
        await mw(handler, msg, {})  # primes the cooldown
        for _ in range(5):
            result = await mw(handler, msg, {})
            self.assertIsNone(result)
        handler.assert_awaited_once()
        msg.reply.assert_awaited_once()

    async def test_notify_fires_again_after_notify_cooldown(self) -> None:
        from app.middlewares import ThrottlingMiddleware

        mw = ThrottlingMiddleware(
            limits={"clear": 100.0},
            notify_on_throttle={"clear"},
            notify_cooldown_sec=10.0,
        )
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/clear")
        with patch("app.middlewares.throttling.time.monotonic", return_value=100.0):
            await mw(handler, msg, {})  # primes the cooldown
        with patch("app.middlewares.throttling.time.monotonic", return_value=101.0):
            await mw(handler, msg, {})  # notifies
        with patch("app.middlewares.throttling.time.monotonic", return_value=105.0):
            await mw(handler, msg, {})  # silent (inside notify cooldown)
        with patch("app.middlewares.throttling.time.monotonic", return_value=112.0):
            await mw(handler, msg, {})  # notifies again
        self.assertEqual(msg.reply.await_count, 2)

    async def test_stale_throttle_entries_expire_by_ttl(self) -> None:
        from app.middlewares import ThrottlingMiddleware

        mw = ThrottlingMiddleware(limits={"pivo": 30.0}, state_ttl_sec=10, state_max_keys=8)
        handler = AsyncMock(return_value="ok")
        msg = self._make_cmd_message("/pivo")

        with patch("app.middlewares.throttling.time.monotonic", return_value=100.0):
            await mw(handler, msg, {})
        with patch("app.middlewares.throttling.time.monotonic", return_value=111.0):
            mw._prune_state(111.0)
            result = await mw(handler, msg, {})

        self.assertEqual(result, "ok")
        self.assertEqual(handler.await_count, 2)

    async def test_oldest_throttle_entries_are_trimmed_when_capacity_exceeded(self) -> None:
        from app.middlewares import ThrottlingMiddleware

        mw = ThrottlingMiddleware(limits={"pivo": 30.0}, state_ttl_sec=1000, state_max_keys=2)
        handler = AsyncMock(return_value="ok")

        with patch("app.middlewares.throttling.time.monotonic", return_value=31.0):
            await mw(handler, self._make_cmd_message("/pivo", user_id=1), {})
        with patch("app.middlewares.throttling.time.monotonic", return_value=32.0):
            await mw(handler, self._make_cmd_message("/pivo", user_id=2), {})
        with patch("app.middlewares.throttling.time.monotonic", return_value=33.0):
            await mw(handler, self._make_cmd_message("/pivo", user_id=3), {})

        self.assertEqual(len(mw._last_used), 2)
        self.assertNotIn((100, 1, "pivo"), mw._last_used)
