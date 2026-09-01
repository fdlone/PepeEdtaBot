from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config.registry import RUNTIME_FIELDS
from app.config.settings import load_settings

# Env vars parsed directly in load_settings (not via the registry), excluding
# deployment identity (BOT_TOKEN, OWNER_ID, DB_PATH, secrets, aliases). Their
# .env.example values must match the code defaults just like registry fields.
SETTINGS_ONLY_ENV_VARS: tuple[str, ...] = (
    "GEN_TRACE_LOG",
    "LOG_LEVEL",
    "MESSAGES_RETENTION_PER_CHAT",
    "SQLITE_BUSY_TIMEOUT_MS",
    "SQLITE_WAL_AUTOCHECKPOINT_PAGES",
    "PIVO_EXPLICIT_MENTIONS_LIMIT",
    "PIVO_SUBSCRIBER_FANOUT_LIMIT",
    "RUNTIME_STATE_TTL_SEC",
    "RUNTIME_STATE_MAX_CHATS",
    "THROTTLE_STATE_TTL_SEC",
    "THROTTLE_STATE_MAX_KEYS",
    "TEXT_CACHE_MAX_MESSAGES",
    "MARKOV_CACHE_MAX_ENTRIES",
)

# Идентичность деплоя: значения у каждого своя, сверять их с дефолтами нечего,
# но присутствовать в .env.example они обязаны — с них начинается настройка.
DEPLOYMENT_ENV_VARS: tuple[str, ...] = (
    "BOT_TOKEN",
    "OWNER_ID",
    "DB_PATH",
    "PIVO_HMAC_SECRET",
    "PIVO_ENCRYPTION_SECRET",
)

# Ключи, которые .env.example показывает закомментированными: пустое значение
# у них означает не «не настроено», а рабочий встроенный дефолт, и раскомментить
# строку — уже осознанный шаг.
OPTIONAL_ENV_VARS: tuple[str, ...] = ("BOT_TEXT_ALIASES", "CHAT_TIMEZONE")


def minimal_env(db_path: str = "test_settings.db") -> dict[str, str]:
    return {
        "BOT_TOKEN": "123:token",
        "DB_PATH": db_path,
        "PIVO_HMAC_SECRET": "test-pivo-hmac-secret-long-enough-32",
        "PIVO_ENCRYPTION_SECRET": "test-pivo-encryption-secret-long-32",
    }


def env_example_values() -> dict[str, str]:
    """KEY=value pairs from .env.example; a duplicated key is a bug there."""
    env_text = (Path(__file__).parents[1] / ".env.example").read_text(
        encoding="utf-8"
    )
    pairs = re.findall(r"^([A-Z_0-9]+)=(.*)$", env_text, re.MULTILINE)
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise AssertionError(f".env.example defines keys twice: {duplicates}")
    return dict(pairs)


class TestSettings(unittest.TestCase):
    def test_load_settings_accepts_minimal_valid_env(self) -> None:
        with patch.dict(os.environ, minimal_env(), clear=True):
            settings = load_settings(load_env=False)

        self.assertEqual(settings.bot_token, "123:token")
        self.assertEqual(settings.reply_probability, 0.08)
        self.assertEqual(settings.max_reply_tokens, 45)
        self.assertFalse(settings.auto_capitalize_replies)
        self.assertTrue(settings.fuzzy_context_casefold)
        self.assertEqual(settings.typing_per_char_ms, 12)
        self.assertEqual(settings.runtime_state_ttl_sec, 86400)
        self.assertEqual(settings.runtime_state_max_chats, 2048)
        self.assertEqual(settings.throttle_state_ttl_sec, 21600)
        self.assertEqual(settings.throttle_state_max_keys, 4096)
        self.assertEqual(settings.text_cache_max_messages, 1000)
        self.assertEqual(settings.messages_retention_per_chat, 1000)
        self.assertEqual(settings.sqlite_busy_timeout_ms, 5000)
        self.assertEqual(settings.sqlite_wal_autocheckpoint_pages, 1000)

    def test_env_example_covers_runtime_fields_with_registry_defaults(self) -> None:
        # "git clone and go": a fresh deployment copies .env.example, so every
        # runtime knob must be present there and its value must match the
        # registry default -- otherwise prod silently runs a stale config.
        env_values = env_example_values()

        missing = [
            spec.env_var for spec in RUNTIME_FIELDS
            if spec.env_var not in env_values
        ]
        self.assertEqual(missing, [], f".env.example misses: {missing}")

        drifted = {}
        for spec in RUNTIME_FIELDS:
            raw = env_values[spec.env_var].strip()
            try:
                parsed = spec.parse(raw)
            except ValueError as error:
                self.fail(
                    f".env.example {spec.env_var}={raw!r} is unparseable: {error}"
                )
            if parsed != spec.parse(spec.default):
                drifted[spec.env_var] = (raw, spec.default)
        self.assertEqual(
            drifted, {},
            f".env.example drifted from registry (env, registry): {drifted}",
        )

    def test_env_example_has_no_keys_the_code_does_not_read(self) -> None:
        # Обратная сторона проверки выше. Ключ, который код не читает, — это
        # либо опечатка, либо след удалённой настройки; и то и другое выглядит
        # как рабочая ручка, пока кто-нибудь не попробует ею воспользоваться.
        known = (
            {spec.env_var for spec in RUNTIME_FIELDS}
            | set(SETTINGS_ONLY_ENV_VARS)
            | set(DEPLOYMENT_ENV_VARS)
            | set(OPTIONAL_ENV_VARS)
        )
        unknown = sorted(set(env_example_values()) - known)
        self.assertEqual(
            unknown, [], f".env.example описывает ключи, которых нет в коде: {unknown}"
        )

    def test_env_example_documents_the_deployment_keys(self) -> None:
        env_values = env_example_values()
        missing = [var for var in DEPLOYMENT_ENV_VARS if var not in env_values]
        self.assertEqual(missing, [], f".env.example misses: {missing}")

    def test_env_example_matches_code_defaults_for_settings_only_knobs(self) -> None:
        # The registry drift check above cannot see knobs parsed directly in
        # load_settings (GEN_TRACE_LOG, LOG_LEVEL, retention/SQLite/state
        # limits). Guard them by loading settings twice -- with bare defaults
        # and with the .env.example values -- and comparing the results.
        env_values = env_example_values()

        missing = [
            var for var in SETTINGS_ONLY_ENV_VARS if var not in env_values
        ]
        self.assertEqual(missing, [], f".env.example misses: {missing}")

        with patch.dict(os.environ, minimal_env(), clear=True):
            from_defaults = load_settings(load_env=False)
        overrides = {var: env_values[var] for var in SETTINGS_ONLY_ENV_VARS}
        with patch.dict(os.environ, minimal_env() | overrides, clear=True):
            from_example = load_settings(load_env=False)

        drifted = {
            var: (
                getattr(from_example, var.lower()),
                getattr(from_defaults, var.lower()),
            )
            for var in SETTINGS_ONLY_ENV_VARS
            if getattr(from_example, var.lower())
            != getattr(from_defaults, var.lower())
        }
        self.assertEqual(
            drifted, {},
            f".env.example drifted from code defaults (env, code): {drifted}",
        )

    def test_gen_trace_log_defaults_off(self) -> None:
        with patch.dict(os.environ, minimal_env(), clear=True):
            settings = load_settings(load_env=False)
        self.assertFalse(settings.gen_trace_log)

        with patch.dict(
            os.environ, minimal_env() | {"GEN_TRACE_LOG": "true"}, clear=True
        ):
            settings = load_settings(load_env=False)
        self.assertTrue(settings.gen_trace_log)

    def test_load_settings_rejects_invalid_gen_trace_log(self) -> None:
        env = minimal_env()
        env["GEN_TRACE_LOG"] = "enabled"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "GEN_TRACE_LOG"):
                load_settings(load_env=False)

    def test_chat_timezone_defaults_to_utc(self) -> None:
        # Дефолт обязан быть байт-идентичен поведению до появления настройки.
        from zoneinfo import ZoneInfo

        with patch.dict(os.environ, minimal_env(), clear=True):
            settings = load_settings(load_env=False)
        self.assertEqual(settings.chat_timezone, ZoneInfo("UTC"))

    def test_chat_timezone_accepts_iana_name(self) -> None:
        from zoneinfo import ZoneInfo

        env = minimal_env() | {"CHAT_TIMEZONE": "Europe/Moscow"}
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings(load_env=False)
        self.assertEqual(settings.chat_timezone, ZoneInfo("Europe/Moscow"))

    def test_chat_timezone_reaches_runtime_state_but_not_the_registry(self) -> None:
        # env-only поле: доезжает до RuntimeState по образцу runtime_state_ttl_sec,
        # но не становится ручкой /set — часовой пояс есть свойство деплоя,
        # и вывод /config (порождаемый из реестра) его не показывает.
        from zoneinfo import ZoneInfo

        from app.config.runtime_state import runtime_state_from_settings

        env = minimal_env() | {"CHAT_TIMEZONE": "Europe/Moscow"}
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings(load_env=False)
        state = runtime_state_from_settings(settings)
        self.assertEqual(state.chat_timezone, ZoneInfo("Europe/Moscow"))
        self.assertNotIn(
            "chat_timezone", {spec.name for spec in RUNTIME_FIELDS}
        )

    def test_load_settings_rejects_unknown_chat_timezone(self) -> None:
        # Fail fast: молчаливый фолбэк в UTC воспроизводил бы O12 — снаружи
        # неотличим от случайного шаблона, а логов у владельца нет.
        env = minimal_env() | {"CHAT_TIMEZONE": "Europe/Moscw"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(
                ValueError, r"CHAT_TIMEZONE.*Europe/Moscw"
            ):
                load_settings(load_env=False)

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

    def test_load_settings_rejects_placeholder_pivo_hmac_secret(self) -> None:
        env = minimal_env()
        env["PIVO_HMAC_SECRET"] = "change_me_to_a_long_random_hmac_secret"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "PIVO_HMAC_SECRET.*placeholder"):
                load_settings(load_env=False)

    def test_load_settings_rejects_placeholder_pivo_encryption_secret(self) -> None:
        env = minimal_env()
        env["PIVO_ENCRYPTION_SECRET"] = "change_me_to_a_long_random_encryption_secret"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "PIVO_ENCRYPTION_SECRET.*placeholder"):
                load_settings(load_env=False)

    def test_load_settings_rejects_short_pivo_hmac_secret(self) -> None:
        """Слабый секрет останавливает старт, а не проходит молча.

        Функция вывода ключа защищает ровно настолько, насколько непредсказуем
        сам секрет: парольная фраза в 16 символов перебирается независимо от
        схемы вывода.
        """
        env = minimal_env()
        env["PIVO_HMAC_SECRET"] = "short-hmac-16chr"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "PIVO_HMAC_SECRET.*32"):
                load_settings(load_env=False)

    def test_load_settings_rejects_short_pivo_encryption_secret(self) -> None:
        env = minimal_env()
        env["PIVO_ENCRYPTION_SECRET"] = "short-encryption"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "PIVO_ENCRYPTION_SECRET.*32"):
                load_settings(load_env=False)

    def test_load_settings_accepts_secret_at_the_length_floor(self) -> None:
        from app.config.settings import MIN_SECRET_LENGTH

        env = minimal_env()
        env["PIVO_HMAC_SECRET"] = "h" * MIN_SECRET_LENGTH
        env["PIVO_ENCRYPTION_SECRET"] = "e" * MIN_SECRET_LENGTH
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings(load_env=False)

        self.assertEqual(len(settings.pivo_hmac_secret), MIN_SECRET_LENGTH)

    def test_load_settings_rejects_identical_pivo_secrets(self) -> None:
        env = minimal_env()
        env["PIVO_HMAC_SECRET"] = "shared-secret-value-long-enough-32ch"
        env["PIVO_ENCRYPTION_SECRET"] = "shared-secret-value-long-enough-32ch"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "must be different"):
                load_settings(load_env=False)

    def test_load_settings_rejects_invalid_bool(self) -> None:
        env = minimal_env()
        env["ENABLE_BACKOFF"] = "maybe"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "Invalid boolean"):
                load_settings(load_env=False)

    def test_load_settings_enables_auto_capitalize_replies(self) -> None:
        env = minimal_env()
        env["AUTO_CAPITALIZE_REPLIES"] = "true"
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings(load_env=False)

        self.assertTrue(settings.auto_capitalize_replies)

    def test_load_settings_enables_fuzzy_context_casefold(self) -> None:
        env = minimal_env()
        env["FUZZY_CONTEXT_CASEFOLD"] = "true"
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings(load_env=False)

        self.assertTrue(settings.fuzzy_context_casefold)

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

    def test_load_settings_rejects_retention_below_text_cache_window(self) -> None:
        env = minimal_env()
        env["MESSAGES_RETENTION_PER_CHAT"] = "499"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(
                ValueError,
                "MESSAGES_RETENTION_PER_CHAT.*TEXT_CACHE_MAX_MESSAGES",
            ):
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

if __name__ == "__main__":
    unittest.main()
