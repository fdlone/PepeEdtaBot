from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from aiogram import Dispatcher


class TestMainWiring(unittest.TestCase):
    def test_command_cooldowns(self) -> None:
        from main import COMMAND_COOLDOWNS_SECONDS

        self.assertEqual(COMMAND_COOLDOWNS_SECONDS["clear"], 3600.0)
        self.assertNotIn("pivo", COMMAND_COOLDOWNS_SECONDS)

    def test_configure_dispatcher_registers_expected_data_routers_and_middleware(self) -> None:
        from app.middlewares import ThrottlingMiddleware
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
            ["common", "admin", "pivo", "learning"],
        )
        middlewares = dp.message.middleware._middlewares
        self.assertEqual(len(middlewares), 1)
        self.assertIsInstance(middlewares[0], ThrottlingMiddleware)
        self.assertEqual(middlewares[0]._state_ttl_sec, 111.0)
        self.assertEqual(middlewares[0]._state_max_keys, 222)


if __name__ == "__main__":
    unittest.main()
