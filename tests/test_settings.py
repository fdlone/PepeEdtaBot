from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from settings import load_settings


def minimal_env(db_path: str = "test_settings.db") -> dict[str, str]:
    return {
        "BOT_TOKEN": "123:token",
        "DB_PATH": db_path,
        "PIVO_HMAC_SECRET": "test-pivo-hmac-secret",
        "PIVO_ENCRYPTION_SECRET": "test-pivo-encryption-secret",
    }


class TestSettings(unittest.TestCase):
    def test_load_settings_accepts_minimal_valid_env(self) -> None:
        with patch.dict(os.environ, minimal_env(), clear=True):
            settings = load_settings(load_env=False)

        self.assertEqual(settings.bot_token, "123:token")
        self.assertEqual(settings.reply_probability, 0.08)
        self.assertEqual(settings.max_reply_tokens, 45)
        self.assertEqual(settings.runtime_state_ttl_sec, 86400)
        self.assertEqual(settings.runtime_state_max_chats, 2048)
        self.assertEqual(settings.throttle_state_ttl_sec, 21600)
        self.assertEqual(settings.throttle_state_max_keys, 4096)
        self.assertEqual(settings.text_cache_max_messages, 500)

    def test_load_settings_rejects_missing_bot_token(self) -> None:
        env = minimal_env()
        env.pop("BOT_TOKEN")
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "BOT_TOKEN"):
                load_settings(load_env=False)

    def test_load_settings_rejects_missing_pivo_hmac_secret(self) -> None:
        env = minimal_env()
        env.pop("PIVO_HMAC_SECRET")
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "PIVO_HMAC_SECRET"):
                load_settings(load_env=False)

    def test_load_settings_rejects_missing_pivo_encryption_secret(self) -> None:
        env = minimal_env()
        env.pop("PIVO_ENCRYPTION_SECRET")
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "PIVO_ENCRYPTION_SECRET"):
                load_settings(load_env=False)

    def test_load_settings_rejects_invalid_bool(self) -> None:
        env = minimal_env()
        env["ENABLE_BACKOFF"] = "maybe"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "Invalid boolean"):
                load_settings(load_env=False)

    def test_load_settings_rejects_negative_min_cooldown(self) -> None:
        env = minimal_env()
        env["MIN_COOLDOWN_SEC"] = "-1"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "MIN_COOLDOWN_SEC"):
                load_settings(load_env=False)

    def test_load_settings_rejects_empty_db_path(self) -> None:
        env = minimal_env("")
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "DB_PATH"):
                load_settings(load_env=False)

    def test_load_settings_rejects_invalid_owner_id(self) -> None:
        env = minimal_env()
        env["OWNER_ID"] = "abc"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "OWNER_ID"):
                load_settings(load_env=False)

    def test_load_settings_default_log_level_is_info(self) -> None:
        with patch.dict(os.environ, minimal_env(), clear=True):
            settings = load_settings(load_env=False)
        self.assertEqual(settings.log_level, "INFO")

    def test_load_settings_accepts_lowercase_log_level(self) -> None:
        env = minimal_env()
        env["LOG_LEVEL"] = "debug"
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings(load_env=False)
        self.assertEqual(settings.log_level, "DEBUG")

    def test_load_settings_rejects_invalid_log_level(self) -> None:
        env = minimal_env()
        env["LOG_LEVEL"] = "verbose"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "LOG_LEVEL"):
                load_settings(load_env=False)

    def test_load_settings_rejects_invalid_runtime_state_ttl(self) -> None:
        env = minimal_env()
        env["RUNTIME_STATE_TTL_SEC"] = "0"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "RUNTIME_STATE_TTL_SEC"):
                load_settings(load_env=False)

    def test_load_settings_rejects_invalid_throttle_state_max_keys(self) -> None:
        env = minimal_env()
        env["THROTTLE_STATE_MAX_KEYS"] = "abc"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "THROTTLE_STATE_MAX_KEYS"):
                load_settings(load_env=False)

    def test_load_settings_rejects_invalid_text_cache_max_messages(self) -> None:
        env = minimal_env()
        env["TEXT_CACHE_MAX_MESSAGES"] = "0"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "TEXT_CACHE_MAX_MESSAGES"):
                load_settings(load_env=False)

    def test_load_settings_default_bot_text_aliases(self) -> None:
        with patch.dict(os.environ, minimal_env(), clear=True):
            settings = load_settings(load_env=False)
        self.assertEqual(settings.bot_text_aliases, frozenset({"pepe", "пепе"}))

    def test_load_settings_empty_bot_text_aliases_falls_back_to_defaults(self) -> None:
        env = minimal_env()
        env["BOT_TEXT_ALIASES"] = "  ,  ,"
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings(load_env=False)
        self.assertEqual(settings.bot_text_aliases, frozenset({"pepe", "пепе"}))

    def test_load_settings_overrides_bot_text_aliases(self) -> None:
        env = minimal_env()
        env["BOT_TEXT_ALIASES"] = "Bim,БИМ ,bim"
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings(load_env=False)
        self.assertEqual(settings.bot_text_aliases, frozenset({"bim", "бим"}))

    def test_load_settings_rejects_conflicting_backoff(self) -> None:
        env = minimal_env()
        env["MARKOV_ORDER"] = "2"
        env["BACKOFF_MIN_ORDER"] = "2"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "BACKOFF_MIN_ORDER"):
                load_settings(load_env=False)


if __name__ == "__main__":
    unittest.main()
