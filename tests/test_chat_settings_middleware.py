"""Tests for the middleware that hands handlers their chat's view of settings."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from aiogram.types import Message

from app.middlewares import BASE_STATE_KEY, ChatSettingsMiddleware
from tests.test_runtime_state import make_runtime_state


def _message(chat_id: int = 100) -> MagicMock:
    msg = MagicMock()
    msg.__class__ = Message  # isinstance(msg, Message) == True
    msg.chat.id = chat_id
    return msg


class TestChatSettingsMiddleware(unittest.IsolatedAsyncioTestCase):
    async def test_chat_without_overrides_gets_the_base_object(self) -> None:
        state = make_runtime_state()
        data: dict[str, object] = {"runtime_state": state}
        handler = AsyncMock(return_value="ok")

        await ChatSettingsMiddleware()(handler, _message(), data)

        self.assertIs(data["runtime_state"], state)

    async def test_chat_with_overrides_gets_its_own_view(self) -> None:
        state = make_runtime_state(reply_probability=0.08)
        state.set_override(100, "reply_probability", 0.5)
        data: dict[str, object] = {"runtime_state": state}
        handler = AsyncMock(return_value="ok")

        await ChatSettingsMiddleware()(handler, _message(100), data)

        view = data["runtime_state"]
        self.assertEqual(view.reply_probability, 0.5)
        self.assertEqual(state.reply_probability, 0.08)

    async def test_neighbouring_chat_is_untouched(self) -> None:
        state = make_runtime_state(reply_probability=0.08)
        state.set_override(100, "reply_probability", 0.5)
        data: dict[str, object] = {"runtime_state": state}

        await ChatSettingsMiddleware()(AsyncMock(), _message(200), data)

        self.assertEqual(data["runtime_state"].reply_probability, 0.08)

    async def test_base_state_stays_reachable_for_writers(self) -> None:
        state = make_runtime_state()
        state.set_override(100, "reply_probability", 0.5)
        data: dict[str, object] = {"runtime_state": state}

        await ChatSettingsMiddleware()(AsyncMock(), _message(100), data)

        self.assertIs(data[BASE_STATE_KEY], state)

    async def test_handler_still_runs_and_its_result_is_returned(self) -> None:
        data: dict[str, object] = {"runtime_state": make_runtime_state()}
        handler = AsyncMock(return_value="ok")

        result = await ChatSettingsMiddleware()(handler, _message(), data)

        self.assertEqual(result, "ok")
        handler.assert_awaited_once()


class TestSettingWritesGoThroughOneDoor(unittest.TestCase):
    """Guard for the trap in this design.

    A handler that assigns a setting field directly would write into the
    per-chat *view*, answer "done", and lose the value on the next update.
    Writes must go through the shared apply path instead — the only place
    that knows which scope the value belongs to.
    """

    SETTING_FIELDS = frozenset(
        {
            "reply_probability",
            "min_cooldown_sec",
            "typing_min_ms",
            "typing_max_ms",
            "markov_order",
            "use_reply_context",
            "reply_context_max_tokens",
        }
    )

    def test_no_handler_assigns_a_setting_field_directly(self) -> None:
        offenders: list[str] = []
        for path in Path("app").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr in self.SETTING_FIELDS
                        and isinstance(target.value, ast.Name)
                        and "state" in target.value.id
                    ):
                        offenders.append(f"{path}:{node.lineno} {target.attr}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
