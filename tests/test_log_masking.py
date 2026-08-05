"""Tests for app.log_masking — chat_id masking helper."""
from __future__ import annotations

import unittest

from app import log_masking


class TestLogMasking(unittest.TestCase):
    def setUp(self) -> None:
        log_masking.reset_masking()

    def tearDown(self) -> None:
        log_masking.reset_masking()

    def test_calling_before_init_raises(self) -> None:
        with self.assertRaises(log_masking.LogMaskingNotInitialized):
            log_masking.mask_chat_id(123)

    def test_mask_is_eight_hex_chars(self) -> None:
        log_masking.init_masking("any-secret-value")
        mask = log_masking.mask_chat_id(123)
        self.assertEqual(len(mask), 8)
        self.assertTrue(all(c in "0123456789abcdef" for c in mask))

    def test_mask_is_deterministic_within_same_secret(self) -> None:
        log_masking.init_masking("secret-a")
        self.assertEqual(
            log_masking.mask_chat_id(-1003736119498),
            log_masking.mask_chat_id(-1003736119498),
        )

    def test_different_chat_ids_yield_different_masks(self) -> None:
        log_masking.init_masking("secret-a")
        self.assertNotEqual(
            log_masking.mask_chat_id(100), log_masking.mask_chat_id(101)
        )

    def test_rotating_secret_changes_mask(self) -> None:
        log_masking.init_masking("secret-a")
        first = log_masking.mask_chat_id(42)
        log_masking.init_masking("secret-b")
        second = log_masking.mask_chat_id(42)
        self.assertNotEqual(first, second)

    def test_raw_id_not_contained_in_mask(self) -> None:
        log_masking.init_masking("secret-x")
        chat_id = 987654321
        mask = log_masking.mask_chat_id(chat_id)
        self.assertNotIn(str(chat_id), mask)


class TestMaskChatIdsInText(unittest.TestCase):
    """Identifiers that arrive inside text we did not format ourselves."""

    def setUp(self) -> None:
        log_masking.reset_masking()
        log_masking.init_masking("secret-for-text-masking")

    def tearDown(self) -> None:
        log_masking.reset_masking()

    def test_masks_flood_control_chat_id(self) -> None:
        text = (
            "Telegram server says - Too Many Requests: retry after 30 "
            "(Flood control exceeded on method 'SendMessage' in chat "
            "-1001147461458)"
        )

        masked = log_masking.mask_chat_ids_in_text(text)

        self.assertNotIn("-1001147461458", masked)
        self.assertIn(log_masking.mask_chat_id(-1001147461458), masked)

    def test_masks_migrate_to_chat_id(self) -> None:
        text = "Bad Request: group chat was upgraded to a supergroup chat -1002233445566"

        masked = log_masking.mask_chat_ids_in_text(text)

        self.assertNotIn("-1002233445566", masked)
        self.assertIn(log_masking.mask_chat_id(-1002233445566), masked)

    def test_keeps_diagnostic_numbers_intact(self) -> None:
        # Retry delay, error code and message id are not chat ids and must
        # survive untouched — the record has to stay useful.
        text = "Too Many Requests: retry after 30, error_code=429, message_id=12345"

        masked = log_masking.mask_chat_ids_in_text(text)

        self.assertEqual(masked, text)

    def test_keeps_short_negative_numbers_intact(self) -> None:
        text = "offset -42 and delta -1000 are not chat ids"

        self.assertEqual(log_masking.mask_chat_ids_in_text(text), text)

    def test_same_chat_masks_identically_across_texts(self) -> None:
        first = log_masking.mask_chat_ids_in_text("flood in chat -1001147461458")
        second = log_masking.mask_chat_ids_in_text("migrate to chat -1001147461458")

        self.assertEqual(
            first.removeprefix("flood in chat "),
            second.removeprefix("migrate to chat "),
        )

    def test_masks_every_identifier_in_one_text(self) -> None:
        text = "moved from -1001147461458 to -1002233445566"

        masked = log_masking.mask_chat_ids_in_text(text)

        self.assertNotIn("-1001147461458", masked)
        self.assertNotIn("-1002233445566", masked)

    def test_uninitialised_masking_yields_placeholder_without_raising(self) -> None:
        log_masking.reset_masking()

        masked = log_masking.mask_chat_ids_in_text("flood in chat -1001147461458")

        self.assertNotIn("-1001147461458", masked)
        self.assertIn(log_masking.UNMASKED_PLACEHOLDER, masked)


if __name__ == "__main__":
    unittest.main()
