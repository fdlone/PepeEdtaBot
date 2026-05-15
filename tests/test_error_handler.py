"""Tests for the centralised Telegram API error handler."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from aiogram.exceptions import TelegramAPIError

from app.handlers.errors import handle_error


def _fake_error_event(exc: Exception) -> MagicMock:
    event = MagicMock()
    event.exception = exc
    event.update = MagicMock()
    return event


class TestErrorHandler(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_api_error_is_logged(self) -> None:
        exc = TelegramAPIError(method=AsyncMock(), message="rate limit")
        event = _fake_error_event(exc)

        with patch("app.handlers.errors.logger") as mock_logger:
            await handle_error(event)

        mock_logger.error.assert_called_once()
        args = mock_logger.error.call_args[0]
        self.assertIn("Telegram API error", args[0])

    async def test_generic_exception_is_logged_with_exc_info(self) -> None:
        exc = ValueError("something went wrong")
        event = _fake_error_event(exc)

        with patch("app.handlers.errors.logger") as mock_logger:
            await handle_error(event)

        mock_logger.error.assert_called_once()
        kwargs = mock_logger.error.call_args[1]
        self.assertIs(kwargs.get("exc_info"), exc)
