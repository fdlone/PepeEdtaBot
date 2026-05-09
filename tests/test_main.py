from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from aiogram import Dispatcher


class TestMainWiring(unittest.TestCase):
    def test_command_cooldowns_are_one_hour_for_pivo_and_clear(self) -> None:
        from main import COMMAND_COOLDOWNS_SECONDS

        self.assertEqual(COMMAND_COOLDOWNS_SECONDS, {"pivo": 3600.0, "clear": 3600.0})

    def test_configure_dispatcher_registers_expected_data_routers_and_middleware(self) -> None:
        from app.middlewares import ThrottlingMiddleware
        from main import configure_dispatcher

        dependencies = {
            "db": MagicMock(),
            "generator": MagicMock(),
            "pivo_service": MagicMock(),
            "learning_service": MagicMock(),
            "state": MagicMock(),
            "settings": MagicMock(),
            "bot_username": "pepebot",
            "bot_id": 12345,
        }

        dp = configure_dispatcher(Dispatcher(), **dependencies)

        for key, value in dependencies.items():
            self.assertIs(dp[key], value)
        self.assertEqual(
            [router.name for router in dp.sub_routers],
            ["common", "admin", "pivo", "learning"],
        )
        middlewares = dp.message.middleware._middlewares
        self.assertEqual(len(middlewares), 1)
        self.assertIsInstance(middlewares[0], ThrottlingMiddleware)


if __name__ == "__main__":
    unittest.main()
