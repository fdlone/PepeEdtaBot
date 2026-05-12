from __future__ import annotations

import unittest
import uuid
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from db import Database
from pivo import (
    PIVO_FALLBACK_MENTIONS,
    PIVO_PRIVACY_MESSAGE,
    PivoMember,
    PivoSecurity,
    build_pivo_mention,
    build_pivo_mentions,
    display_name_from_user,
    get_random_pivo_message,
    normalize_username,
)
from pivo_templates import (
    PIVO_TARGET_BODY_PARTS,
    PIVO_TARGET_BOTTOM_PARTS,
    PIVO_TARGET_INTROS,
    PIVO_TARGET_TOP_PARTS,
)

RANDOM_CHOICE_PATH = "app.services.pivo_message_builder.random.choice"


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
) -> PivoMember:
    return PivoMember(
        encrypted_user_id=security.encrypt_value(user_id),
        encrypted_username=security.encrypt_value(username),
        encrypted_display_name=security.encrypt_value(display_name),
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
        text = get_random_pivo_message("@pepe")

        self.assertIn("@pepe", text)
        self.assertIn("Discord", text)

    def test_pivo_privacy_message_has_no_technical_details(self) -> None:
        self.assertIn("/pivo_on", PIVO_PRIVACY_MESSAGE)
        self.assertIn("/pivo_off", PIVO_PRIVACY_MESSAGE)
        self.assertNotIn("HMAC", PIVO_PRIVACY_MESSAGE)
        self.assertNotIn("таблиц", PIVO_PRIVACY_MESSAGE.lower())


class TestPivoServiceQuota(unittest.IsolatedAsyncioTestCase):
    async def test_regular_user_has_three_daily_calls(self) -> None:
        from app.services.pivo_service import PIVO_DAILY_LIMIT_USER, PivoService

        security = make_security()
        db = AsyncMock()
        db.consume_pivo_daily_call = AsyncMock(return_value=(True, 1))
        service = PivoService(db=db, security=security)

        result = await service.consume_daily_call_quota(
            chat_id=100,
            user_id=200,
            is_admin_or_owner=False,
            today=date(2026, 5, 8),
        )

        self.assertTrue(result.allowed)
        self.assertEqual(result.limit, PIVO_DAILY_LIMIT_USER)
        self.assertEqual(result.usage_day, "2026-05-08")
        db.consume_pivo_daily_call.assert_awaited_once_with(
            chat_hash=security.hmac_value(100),
            user_hash=security.hmac_value(200),
            usage_day="2026-05-08",
            limit=PIVO_DAILY_LIMIT_USER,
        )

    async def test_admin_has_five_daily_calls(self) -> None:
        from app.services.pivo_service import PIVO_DAILY_LIMIT_ADMIN, PivoService

        security = make_security()
        db = AsyncMock()
        db.consume_pivo_daily_call = AsyncMock(return_value=(True, 5))
        service = PivoService(db=db, security=security)

        result = await service.consume_daily_call_quota(
            chat_id=100,
            user_id=200,
            is_admin_or_owner=True,
            today=date(2026, 5, 8),
        )

        self.assertTrue(result.allowed)
        self.assertEqual(result.used_count, 5)
        self.assertEqual(result.limit, PIVO_DAILY_LIMIT_ADMIN)

    async def test_refund_daily_call_uses_same_hashes_and_day(self) -> None:
        from app.services.pivo_service import PivoService

        security = make_security()
        db = AsyncMock()
        db.refund_pivo_daily_call = AsyncMock()
        service = PivoService(db=db, security=security)

        await service.refund_daily_call_quota(
            chat_id=100,
            user_id=200,
            usage_day="2026-05-08",
        )

        db.refund_pivo_daily_call.assert_awaited_once_with(
            chat_hash=security.hmac_value(100),
            user_hash=security.hmac_value(200),
            usage_day="2026-05-08",
        )

    async def test_build_call_message_uses_explicit_mentions_without_db_lookup(self) -> None:
        from app.services.pivo_service import PivoService

        security = make_security()
        db = AsyncMock()
        db.get_chat_members = AsyncMock()
        service = PivoService(db=db, security=security)

        choices = iter(
            [
                PIVO_TARGET_INTROS[0],
                PIVO_TARGET_TOP_PARTS[1],
                PIVO_TARGET_BODY_PARTS[0],
                PIVO_TARGET_BOTTOM_PARTS[0],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text, mentions_count = await service.build_call_message(
                chat_id=100,
                caller_user_id=200,
                planned_time="20:00",
                target="watch movie",
                explicit_mentions=("@friend",),
            )

        db.get_chat_members.assert_not_called()
        self.assertEqual(mentions_count, 1)
        self.assertIn("общий сбор в 20:00 в Discord.", text)
        self.assertIn("watch movie", text)
        self.assertIn("@friend", text)

    async def test_build_call_message_builds_contextual_body(self) -> None:
        from app.services.pivo_service import PivoService

        security = make_security()
        db = AsyncMock()
        db.get_chat_members = AsyncMock(return_value=[])
        service = PivoService(db=db, security=security)

        choices = iter(
            [
                PIVO_TARGET_INTROS[0],
                "местные дегенераты",
                PIVO_TARGET_TOP_PARTS[1],
                PIVO_TARGET_BODY_PARTS[2],
                PIVO_TARGET_BOTTOM_PARTS[0],
            ]
        )
        with patch(RANDOM_CHOICE_PATH, side_effect=lambda _: next(choices)):
            text, mentions_count = await service.build_call_message(
                chat_id=100,
                caller_user_id=200,
                planned_time="<20:00>",
                target="movie & chat",
            )

        self.assertEqual(mentions_count, 0)
        self.assertIn("&lt;20:00&gt;", text)
        self.assertIn("movie &amp; chat", text)


class TestPivoServiceFlow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from app.services.pivo_service import PivoService

        self.db_path = Path(f"test_pivo_flow_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        self.security = make_security()
        self.service = PivoService(db=self.db, security=self.security)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def test_subscribe_call_quota_and_unsubscribe_flow(self) -> None:
        caller = SimpleNamespace(
            id=100,
            username="caller",
            full_name="Caller User",
            is_bot=False,
        )
        friend = SimpleNamespace(
            id=200,
            username="friend",
            full_name="Friend User",
            is_bot=False,
        )

        await self.service.subscribe(chat_id=500, user=caller)
        await self.service.subscribe(chat_id=500, user=friend)

        text, mentions_count = await self.service.build_call_message(
            chat_id=500,
            caller_user_id=caller.id,
        )
        self.assertEqual(mentions_count, 1)
        self.assertIn("@friend", text)
        self.assertNotIn("@caller", text)

        quota = await self.service.consume_daily_call_quota(
            chat_id=500,
            user_id=caller.id,
            is_admin_or_owner=False,
            today=date(2026, 5, 9),
        )
        self.assertTrue(quota.allowed)
        self.assertEqual(quota.used_count, 1)
        self.assertEqual(quota.limit, 3)

        quota = await self.service.consume_daily_call_quota(
            chat_id=500,
            user_id=caller.id,
            is_admin_or_owner=False,
            today=date(2026, 5, 9),
        )
        self.assertTrue(quota.allowed)
        self.assertEqual(quota.used_count, 2)

        quota = await self.service.consume_daily_call_quota(
            chat_id=500,
            user_id=caller.id,
            is_admin_or_owner=False,
            today=date(2026, 5, 9),
        )
        self.assertTrue(quota.allowed)
        self.assertEqual(quota.used_count, 3)

        quota = await self.service.consume_daily_call_quota(
            chat_id=500,
            user_id=caller.id,
            is_admin_or_owner=False,
            today=date(2026, 5, 9),
        )
        self.assertFalse(quota.allowed)
        self.assertEqual(quota.used_count, 3)

        await self.service.unsubscribe(chat_id=500, user_id=friend.id)
        text, mentions_count = await self.service.build_call_message(
            chat_id=500,
            caller_user_id=caller.id,
        )
        self.assertEqual(mentions_count, 0)
        self.assertNotIn(PIVO_FALLBACK_MENTIONS, text)


if __name__ == "__main__":
    unittest.main()
