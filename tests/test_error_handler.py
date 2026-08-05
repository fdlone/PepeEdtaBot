"""Tests for the centralised Telegram API error handler."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from aiogram.exceptions import TelegramAPIError

from app import log_masking
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

    async def test_chat_id_inside_error_text_is_masked(self) -> None:
        # aiogram bakes the raw chat_id into the message of TelegramRetryAfter
        # and friends, so masking the arguments is not enough.
        exc = TelegramAPIError(
            method=AsyncMock(),
            message=(
                "Too Many Requests: retry after 30 (Flood control exceeded "
                "on method 'SendMessage' in chat -1001147461458)"
            ),
        )
        event = _fake_error_event(exc)

        log_masking.reset_masking()
        log_masking.init_masking("secret-for-error-handler")
        self.addCleanup(log_masking.reset_masking)

        with patch("app.handlers.errors.logger") as mock_logger:
            await handle_error(event)

        logged = " ".join(str(arg) for arg in mock_logger.error.call_args[0])
        self.assertNotIn("-1001147461458", logged)
        self.assertIn(log_masking.mask_chat_id(-1001147461458), logged)
        # Диагностика остаётся: тип ошибки и время ожидания не тронуты.
        self.assertIn("retry after 30", logged)

    async def test_error_is_still_logged_when_masking_is_uninitialised(self) -> None:
        exc = TelegramAPIError(
            method=AsyncMock(),
            message="Flood control exceeded in chat -1001147461458",
        )
        event = _fake_error_event(exc)

        log_masking.reset_masking()

        with patch("app.handlers.errors.logger") as mock_logger:
            await handle_error(event)

        mock_logger.error.assert_called_once()
        logged = " ".join(str(arg) for arg in mock_logger.error.call_args[0])
        self.assertNotIn("-1001147461458", logged)

    async def test_generic_exception_is_logged_with_exc_info(self) -> None:
        exc = ValueError("something went wrong")
        event = _fake_error_event(exc)

        with patch("app.handlers.errors.logger") as mock_logger:
            await handle_error(event)

        mock_logger.error.assert_called_once()
        kwargs = mock_logger.error.call_args[1]
        self.assertIs(kwargs.get("exc_info"), exc)
