from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pivo import (
    PIVO_FALLBACK_MENTIONS,
    PIVO_PRIVACY_MESSAGE,
    PivoMember,
    PivoSecurity,
    build_pivo_mention,
    build_pivo_mentions,
    collect_pivo_mentions,
    display_name_from_user,
    get_random_pivo_message,
    normalize_username,
)
from pivo_templates import PIVO_TEMPLATES


def make_security() -> PivoSecurity:
    return PivoSecurity(
        hmac_secret="test-hmac-secret-value",
        encryption_secret="test-encryption-secret-value",
    )


def make_member(
    security: PivoSecurity,
    *,
    user_id: int,
    username: str = "",
    display_name: str = "Тестовый Пользователь",
    is_bot: bool = False,
) -> PivoMember:
    return PivoMember(
        encrypted_user_id=security.encrypt_value(user_id),
        encrypted_username=security.encrypt_value(username),
        encrypted_display_name=security.encrypt_value(display_name),
        is_bot=is_bot,
    )


class TestPivo(unittest.TestCase):
    def test_normalize_username(self) -> None:
        self.assertEqual(normalize_username("pepe"), "@pepe")
        self.assertEqual(normalize_username("@pepe"), "@pepe")
        self.assertEqual(normalize_username("  pepe  "), "@pepe")
        self.assertEqual(normalize_username(""), "")
        self.assertEqual(normalize_username(None), "")

    def test_security_encrypts_values_and_uses_stable_hmac(self) -> None:
        security = make_security()
        encrypted = security.encrypt_value("12345")

        self.assertNotIn("12345", encrypted)
        self.assertEqual(security.decrypt_value(encrypted), "12345")
        self.assertEqual(security.hmac_value("12345"), security.hmac_value("12345"))
        self.assertNotEqual(security.hmac_value("12345"), "12345")

    def test_build_pivo_mention_uses_username(self) -> None:
        security = make_security()
        member = make_member(security, user_id=100, username="PepeUser")

        self.assertEqual(build_pivo_mention(member, security), "@PepeUser")

    def test_build_pivo_mention_uses_inline_mention_without_username(self) -> None:
        security = make_security()
        member = make_member(
            security,
            user_id=101,
            username="",
            display_name="Pepe <Admin>",
        )

        mention = build_pivo_mention(member, security)

        self.assertEqual(
            mention,
            '<a href="tg://user?id=101">Pepe &lt;Admin&gt;</a>',
        )

    def test_build_pivo_mention_skips_bots(self) -> None:
        security = make_security()
        member = make_member(security, user_id=102, username="bot", is_bot=True)

        self.assertEqual(build_pivo_mention(member, security), "")

    def test_build_pivo_mentions_excludes_caller(self) -> None:
        security = make_security()
        members = [
            make_member(security, user_id=201, username="caller"),
            make_member(security, user_id=202, username="friend"),
        ]

        mentions = build_pivo_mentions(members, caller_user_id=201, security=security)

        self.assertEqual(mentions, "@friend")

    def test_build_pivo_mentions_fallback_when_empty(self) -> None:
        security = make_security()
        members = [make_member(security, user_id=301, username="caller")]

        mentions = build_pivo_mentions(members, caller_user_id=301, security=security)

        self.assertEqual(mentions, PIVO_FALLBACK_MENTIONS)

    def test_collect_pivo_mentions_excludes_bots(self) -> None:
        security = make_security()
        members = [
            make_member(security, user_id=401, username="friend"),
            make_member(security, user_id=402, username="bot", is_bot=True),
        ]

        mentions = collect_pivo_mentions(
            members,
            caller_user_id=999,
            security=security,
        )

        self.assertEqual(mentions, ["@friend"])

    def test_display_name_from_user_supports_fallback(self) -> None:
        self.assertEqual(
            display_name_from_user(SimpleNamespace(full_name="Pepe Tester")),
            "Pepe Tester",
        )
        self.assertEqual(
            display_name_from_user(
                SimpleNamespace(full_name="", first_name="Pepe", last_name="Tester")
            ),
            "Pepe Tester",
        )
        self.assertEqual(display_name_from_user(SimpleNamespace()), "участник")

    def test_get_random_pivo_message_uses_whole_template(self) -> None:
        with patch("random.choice", return_value=PIVO_TEMPLATES[0]):
            text = get_random_pivo_message("@pepe")

        self.assertIn("@pepe", text)
        self.assertIn("Внимание, заслуженные дегенераты", text)

    def test_pivo_privacy_message_has_no_technical_details(self) -> None:
        self.assertIn("/pivo_on", PIVO_PRIVACY_MESSAGE)
        self.assertIn("/pivo_off", PIVO_PRIVACY_MESSAGE)
        self.assertNotIn("HMAC", PIVO_PRIVACY_MESSAGE)
        self.assertNotIn("таблиц", PIVO_PRIVACY_MESSAGE.lower())


if __name__ == "__main__":
    unittest.main()
