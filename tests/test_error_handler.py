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
        self.assertIn("Exception in handler", args[0])
        self.assertIn("rate limit", " ".join(str(a) for a in args))

    async def test_chat_id_inside_error_text_is_masked(self) -> None:
        # aiogram bakes the raw chat_id into the message of TelegramRetryAfter
        # and friends, so masking the arguments is not enough.
        exc = TelegramAPIError(
            method=AsyncMock(),
            message=(
                "Too Many Requests: retry after 30 (Flood control exceeded "
                "on method 'SendMessage' in chat -1001234567890)"
            ),
        )
        event = _fake_error_event(exc)

        log_masking.reset_masking()
        log_masking.init_masking("secret-for-error-handler")
        self.addCleanup(log_masking.reset_masking)

        with patch("app.handlers.errors.logger") as mock_logger:
            await handle_error(event)

        logged = " ".join(str(arg) for arg in mock_logger.error.call_args[0])
        self.assertNotIn("-1001234567890", logged)
        self.assertIn(log_masking.mask_chat_id(-1001234567890), logged)
        # Диагностика остаётся: тип ошибки и время ожидания не тронуты.
        self.assertIn("retry after 30", logged)

    async def test_error_is_still_logged_when_masking_is_uninitialised(self) -> None:
        exc = TelegramAPIError(
            method=AsyncMock(),
            message="Flood control exceeded in chat -1001234567890",
        )
        event = _fake_error_event(exc)

        log_masking.reset_masking()

        with patch("app.handlers.errors.logger") as mock_logger:
            await handle_error(event)

        mock_logger.error.assert_called_once()
        logged = " ".join(str(arg) for arg in mock_logger.error.call_args[0])
        self.assertNotIn("-1001234567890", logged)

    async def test_generic_exception_keeps_its_diagnostics(self) -> None:
        """Обычное исключение печатается трассировкой, а не `exc_info`.

        Прежняя форма выбирала ветку по типу: `TelegramAPIError` шёл через
        маскирование, остальное — через `exc_info`. Это оказалось хрупким по
        построению: обёртка над отправкой (`PartialDeliveryError`) подменила
        тип, и исключение с сырым chat_id внутри `__cause__` ушло в
        необработанную ветку. Теперь маскируется всё и всегда, поэтому
        трассировка попадает в сообщение — диагностика сохраняется.
        """
        exc = ValueError("something went wrong")
        event = _fake_error_event(exc)

        with patch("app.handlers.errors.logger") as mock_logger:
            await handle_error(event)

        mock_logger.error.assert_called_once()
        logged = " ".join(str(arg) for arg in mock_logger.error.call_args[0])
        self.assertIn("something went wrong", logged)
        self.assertIn("ValueError", logged)

    async def test_wrapped_exception_still_gets_masked(self) -> None:
        """Chat_id внутри `__cause__` тоже маскируется.

        Ровно тот случай, на котором прежняя форма ломалась: отправка
        оборвалась на второй части ответа, обёртка `PartialDeliveryError`
        подменила тип, а traceback печатает и причину. Инвариант §4 не должен
        зависеть от того, не завернул ли кто-то исключение по дороге.
        """
        from app.services.reply_pipeline import PartialDeliveryError

        try:
            try:
                raise TelegramAPIError(
                    method=AsyncMock(),
                    message="Flood control exceeded in chat -1001234567890",
                )
            except TelegramAPIError as cause:
                raise PartialDeliveryError(1) from cause
        except PartialDeliveryError as wrapped:
            event = _fake_error_event(wrapped)

        log_masking.reset_masking()
        log_masking.init_masking("secret-for-error-handler")
        self.addCleanup(log_masking.reset_masking)

        with patch("app.handlers.errors.logger") as mock_logger:
            await handle_error(event)

        logged = " ".join(str(arg) for arg in mock_logger.error.call_args[0])
        self.assertNotIn("-1001234567890", logged, "сырой chat_id из причины попал в лог")
        self.assertIn("Flood control", logged, "причина потерялась")
