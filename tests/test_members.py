"""Tests for MembersRepo, MembersService, and key_derivation."""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

import aiosqlite

from app.infrastructure import migrator
from app.repositories.members_repo import MembersRepo
from app.security.key_derivation import derive_fernet_key, derive_key
from app.services.members_service import MembersService

_HMAC_SECRET = "test-hmac-secret-32-chars-minimum!"
_ENC_SECRET = "test-encryption-secret-32-chars!!"


def _fake_user(user_id: int = 42, username: str = "tester") -> MagicMock:
    u = MagicMock()
    u.id = user_id
    u.username = username
    u.is_bot = False
    return u


async def _make_conn() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await migrator.run(conn)
    return conn


class TestKeyDerivation(unittest.TestCase):
    def test_derive_key_returns_32_bytes(self) -> None:
        key = derive_key(_HMAC_SECRET, "members:hmac")
        self.assertEqual(len(key), 32)
        self.assertIsInstance(key, bytes)

    def test_derive_fernet_key_is_valid_base64(self) -> None:
        import base64
        key = derive_fernet_key(_ENC_SECRET, "members:encryption")
        decoded = base64.urlsafe_b64decode(key)
        self.assertEqual(len(decoded), 32)

    def test_different_domains_produce_different_keys(self) -> None:
        k1 = derive_key(_HMAC_SECRET, "members:hmac")
        k2 = derive_key(_HMAC_SECRET, "pivo:hmac")
        self.assertNotEqual(k1, k2)

    def test_members_key_differs_from_raw_secret(self) -> None:
        k = derive_key(_HMAC_SECRET, "members:hmac")
        self.assertNotEqual(k, _HMAC_SECRET.encode("utf-8").ljust(32)[:32])

    def test_same_inputs_produce_same_key(self) -> None:
        k1 = derive_key(_HMAC_SECRET, "members:hmac")
        k2 = derive_key(_HMAC_SECRET, "members:hmac")
        self.assertEqual(k1, k2)

    def test_pivo_raw_hmac_differs_from_derived(self) -> None:
        """Убеждаемся что HKDF-ключ для members не совпадает с raw-секретом pivo."""
        import hashlib
        import hmac as hmac_mod
        pivo_hmac = hmac_mod.new(
            _HMAC_SECRET.encode(), b"12345", hashlib.sha256
        ).digest()
        members_key = derive_key(_HMAC_SECRET, "members:hmac")
        members_hmac = hmac_mod.new(members_key, b"12345", hashlib.sha256).digest()
        self.assertNotEqual(pivo_hmac, members_hmac)


class TestMembersRepo(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.conn = await _make_conn()
        self.repo = MembersRepo(self._get_conn, asyncio.Lock())

    async def asyncTearDown(self) -> None:
        await self.conn.close()

    async def _get_conn(self) -> aiosqlite.Connection:
        return self.conn

    async def test_upsert_and_get(self) -> None:
        await self.repo.upsert(
            chat_hash="ch1", user_hash="uh1",
            encrypted_user_id="euid", encrypted_payload="epay",
            key_version=1, consent_version=1,
        )
        row = await self.repo.get("ch1", "uh1")
        self.assertIsNotNone(row)
        self.assertEqual(row["encrypted_user_id"], "euid")
        self.assertEqual(row["encrypted_payload"], "epay")
        self.assertEqual(row["key_version"], 1)
        self.assertEqual(row["consent_version"], 1)

    async def test_get_missing_returns_none(self) -> None:
        self.assertIsNone(await self.repo.get("no", "such"))

    async def test_upsert_updates_existing(self) -> None:
        await self.repo.upsert(
            chat_hash="ch1", user_hash="uh1",
            encrypted_user_id="old", encrypted_payload="old_pay",
            key_version=1, consent_version=1,
        )
        await self.repo.upsert(
            chat_hash="ch1", user_hash="uh1",
            encrypted_user_id="new", encrypted_payload="new_pay",
            key_version=1, consent_version=2,
        )
        row = await self.repo.get("ch1", "uh1")
        self.assertEqual(row["encrypted_user_id"], "new")
        self.assertEqual(row["consent_version"], 2)

    async def test_remove(self) -> None:
        await self.repo.upsert(
            chat_hash="ch1", user_hash="uh1",
            encrypted_user_id="e", encrypted_payload="p",
            key_version=1, consent_version=1,
        )
        await self.repo.remove("ch1", "uh1")
        self.assertIsNone(await self.repo.get("ch1", "uh1"))

    async def test_remove_nonexistent_is_silent(self) -> None:
        await self.repo.remove("no", "such")  # не должно бросать

    async def test_list_members_returns_all_for_chat(self) -> None:
        for i in range(3):
            await self.repo.upsert(
                chat_hash="chat", user_hash=f"u{i}",
                encrypted_user_id=f"uid{i}", encrypted_payload="p",
                key_version=1, consent_version=1,
            )
        members = await self.repo.list_members("chat")
        self.assertEqual(len(members), 3)

    async def test_list_members_isolated_by_chat(self) -> None:
        await self.repo.upsert(
            chat_hash="chatA", user_hash="u1",
            encrypted_user_id="e", encrypted_payload="p",
            key_version=1, consent_version=1,
        )
        await self.repo.upsert(
            chat_hash="chatB", user_hash="u2",
            encrypted_user_id="e", encrypted_payload="p",
            key_version=1, consent_version=1,
        )
        self.assertEqual(len(await self.repo.list_members("chatA")), 1)
        self.assertEqual(len(await self.repo.list_members("chatB")), 1)


class TestMembersService(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.conn = await _make_conn()
        repo = MembersRepo(self._get_conn, asyncio.Lock())
        self.svc = MembersService(repo, _HMAC_SECRET, _ENC_SECRET)

    async def asyncTearDown(self) -> None:
        await self.conn.close()

    async def _get_conn(self) -> aiosqlite.Connection:
        return self.conn

    async def test_record_and_get_profile(self) -> None:
        user = _fake_user(user_id=100)
        payload = {"display_name": "Тестер", "locale": "ru"}
        await self.svc.record_consent(chat_id=1, user=user, payload=payload, consent_version=1)
        result = await self.svc.get_profile(chat_id=1, user_id=100)
        self.assertEqual(result, payload)

    async def test_get_profile_missing_returns_none(self) -> None:
        self.assertIsNone(await self.svc.get_profile(chat_id=99, user_id=99))

    async def test_revoke_removes_profile(self) -> None:
        user = _fake_user(user_id=200)
        await self.svc.record_consent(chat_id=1, user=user, payload={}, consent_version=1)
        await self.svc.revoke(chat_id=1, user_id=200)
        self.assertIsNone(await self.svc.get_profile(chat_id=1, user_id=200))

    async def test_update_consent(self) -> None:
        user = _fake_user(user_id=300)
        await self.svc.record_consent(chat_id=1, user=user, payload={"v": 1}, consent_version=1)
        await self.svc.record_consent(chat_id=1, user=user, payload={"v": 2}, consent_version=2)
        result = await self.svc.get_profile(chat_id=1, user_id=300)
        self.assertEqual(result["v"], 2)

    async def test_different_chats_isolated(self) -> None:
        user = _fake_user(user_id=400)
        await self.svc.record_consent(chat_id=1, user=user, payload={"chat": 1}, consent_version=1)
        await self.svc.record_consent(chat_id=2, user=user, payload={"chat": 2}, consent_version=1)
        r1 = await self.svc.get_profile(chat_id=1, user_id=400)
        r2 = await self.svc.get_profile(chat_id=2, user_id=400)
        self.assertEqual(r1["chat"], 1)
        self.assertEqual(r2["chat"], 2)

    async def test_payload_is_encrypted_at_rest(self) -> None:
        """Данные в БД должны быть зашифрованы — не читаться как открытый текст."""
        user = _fake_user(user_id=500)
        secret_payload = {"secret": "не должно быть видно"}
        await self.svc.record_consent(
            chat_id=1, user=user, payload=secret_payload, consent_version=1
        )

        cur = await self.conn.execute("SELECT encrypted_payload FROM chat_member_profiles")
        raw = (await cur.fetchone())[0]
        self.assertNotIn("не должно быть видно", raw)

    async def test_members_keys_differ_from_pivo_keys(self) -> None:
        """HKDF-ключи для members не совпадают с raw pivo-секретами."""
        import hashlib
        import hmac as hmac_mod

        from app.security.key_derivation import derive_key

        members_key = derive_key(_HMAC_SECRET, "members:hmac")
        pivo_raw = _HMAC_SECRET.encode("utf-8")
        self.assertNotEqual(members_key, pivo_raw[:32].ljust(32, b"\x00"))

        members_hmac = hmac_mod.new(members_key, b"42", hashlib.sha256).hexdigest()
        pivo_hmac = hmac_mod.new(pivo_raw, b"42", hashlib.sha256).hexdigest()
        self.assertNotEqual(members_hmac, pivo_hmac)

    async def test_migrator_creates_table(self) -> None:
        cur = await self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_member_profiles'"
        )
        self.assertIsNotNone(await cur.fetchone())
