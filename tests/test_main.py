from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from aiogram import Dispatcher


class TestMainWiring(unittest.TestCase):
    def test_command_cooldowns(self) -> None:
        from main import COMMAND_COOLDOWNS_SECONDS

        self.assertEqual(COMMAND_COOLDOWNS_SECONDS["clear"], 3600.0)
        # Deliberate since 282051f: the middleware drops a call before the
        # handler, so a throttled /pivo replied with silence instead of the
        # "daily quota exhausted" message. The quota is its rate limit.
        self.assertNotIn("pivo", COMMAND_COOLDOWNS_SECONDS)

    def test_cheap_and_writing_commands_are_throttled(self) -> None:
        from main import COMMAND_COOLDOWNS_SECONDS

        # /config is the sharp one: no AdminOrOwner, no GroupOnly, so anyone
        # could loop it and make the bot answer indefinitely.
        for command in ("ping", "help", "config", "pivo_privacy"):
            with self.subTest(command=command):
                self.assertEqual(COMMAND_COOLDOWNS_SECONDS[command], 15.0)
        self.assertEqual(COMMAND_COOLDOWNS_SECONDS["stats"], 60.0)
        # These two write to SQLite on every call.
        for command in ("pivo_on", "pivo_off"):
            with self.subTest(command=command):
                self.assertEqual(COMMAND_COOLDOWNS_SECONDS[command], 30.0)

    def test_every_advertised_command_has_a_window_or_a_stated_exemption(
        self,
    ) -> None:
        # Guard for the next command: it must land in one class or the other,
        # never fall through the gap the way these did.
        from app.presentation.bot_messages import TELEGRAM_COMMANDS
        from main import COMMAND_COOLDOWNS_SECONDS, COOLDOWN_EXEMPT_COMMANDS

        uncovered = {
            command
            for command, _description in TELEGRAM_COMMANDS
            if command not in COMMAND_COOLDOWNS_SECONDS
            and command not in COOLDOWN_EXEMPT_COMMANDS
        }
        self.assertEqual(uncovered, set())

    def test_exemptions_and_windows_do_not_overlap(self) -> None:
        from main import COMMAND_COOLDOWNS_SECONDS, COOLDOWN_EXEMPT_COMMANDS

        self.assertEqual(
            set(COMMAND_COOLDOWNS_SECONDS) & COOLDOWN_EXEMPT_COMMANDS, set()
        )

    def test_commands_asking_for_confirmation_notify_on_throttle(self) -> None:
        # Silence in reply to "add me to the list" reads as a broken bot.
        from main import COMMANDS_NOTIFIED_ON_THROTTLE

        self.assertEqual(
            COMMANDS_NOTIFIED_ON_THROTTLE, {"clear", "pivo_on", "pivo_off"}
        )

    def test_configure_dispatcher_registers_expected_data_routers_and_middleware(self) -> None:
        from app.middlewares import ChatSettingsMiddleware, ThrottlingMiddleware
        from main import configure_dispatcher

        dependencies = {
            "db": MagicMock(),
            "generator": MagicMock(),
            "pivo_service": MagicMock(),
            "learning_service": MagicMock(),
            "runtime_state": MagicMock(),
            "settings": MagicMock(),
            "bot_username": "pepebot",
            "bot_id": 12345,
        }
        dependencies["settings"].bot_text_aliases = frozenset({"pepe"})
        dependencies["settings"].throttle_state_ttl_sec = 111
        dependencies["settings"].throttle_state_max_keys = 222
        dependencies["settings"].text_cache_max_messages = 333

        dp = configure_dispatcher(Dispatcher(), **dependencies)

        for key, value in dependencies.items():
            self.assertIs(dp[key], value)
        # bot_text_aliases is derived from settings, not passed as a separate
        # configure_dispatcher kwarg.
        self.assertIs(
            dp["bot_text_aliases"], dependencies["settings"].bot_text_aliases
        )
        self.assertEqual(
            [router.name for router in dp.sub_routers],
            ["common", "admin", "pivo", "learning", "errors"],
        )
        middlewares = dp.message.middleware._middlewares
        self.assertEqual(len(middlewares), 2)
        self.assertIsInstance(middlewares[0], ThrottlingMiddleware)
        # Order matters: a throttled update must not pay for building the
        # per-chat settings view.
        self.assertIsInstance(middlewares[1], ChatSettingsMiddleware)
        self.assertEqual(middlewares[0]._state_ttl_sec, 111.0)
        self.assertEqual(middlewares[0]._state_max_keys, 222)
        self.assertEqual(
            dp["learning_service"], dependencies["learning_service"]
        )


if __name__ == "__main__":
    unittest.main()
