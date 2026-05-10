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


if __name__ == "__main__":
    unittest.main()
