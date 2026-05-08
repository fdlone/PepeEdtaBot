from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import aiosqlite

ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]


class MembersRepo:
    """Доступ к таблице chat_member_profiles."""

    def __init__(self, conn_provider: ConnProvider, lock: asyncio.Lock) -> None:
        self._conn_provider = conn_provider
        self._lock = lock

    async def upsert(
        self,
        *,
        chat_hash: str,
        user_hash: str,
        encrypted_user_id: str,
        encrypted_payload: str,
        key_version: int,
        consent_version: int,
    ) -> None:
        async with self._lock:
            db = await self._conn_provider()
            await db.execute(
                """
                INSERT INTO chat_member_profiles(
                    chat_hash, user_hash, encrypted_user_id, encrypted_payload,
                    key_version, consent_version, consented_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(chat_hash, user_hash) DO UPDATE SET
                    encrypted_user_id = excluded.encrypted_user_id,
                    encrypted_payload = excluded.encrypted_payload,
                    key_version       = excluded.key_version,
                    consent_version   = excluded.consent_version,
                    updated_at        = datetime('now')
                """,
                (
                    chat_hash, user_hash, encrypted_user_id, encrypted_payload,
                    key_version, consent_version,
                ),
            )
            await db.commit()

    async def get(self, chat_hash: str, user_hash: str) -> dict[str, object] | None:
        async with self._lock:
            db = await self._conn_provider()
            cur = await db.execute(
                """
                SELECT encrypted_user_id, encrypted_payload, key_version,
                       consent_version, consented_at, updated_at
                FROM chat_member_profiles
                WHERE chat_hash = ? AND user_hash = ?
                """,
                (chat_hash, user_hash),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "encrypted_user_id": str(row[0]),
            "encrypted_payload": str(row[1]),
            "key_version": int(row[2]),
            "consent_version": int(row[3]),
            "consented_at": str(row[4]),
            "updated_at": str(row[5]),
        }

    async def remove(self, chat_hash: str, user_hash: str) -> None:
        async with self._lock:
            db = await self._conn_provider()
            await db.execute(
                "DELETE FROM chat_member_profiles WHERE chat_hash = ? AND user_hash = ?",
                (chat_hash, user_hash),
            )
            await db.commit()

    async def list_members(self, chat_hash: str) -> list[dict[str, object]]:
        async with self._lock:
            db = await self._conn_provider()
            cur = await db.execute(
                """
                SELECT user_hash, encrypted_user_id, encrypted_payload,
                       key_version, consent_version, consented_at, updated_at
                FROM chat_member_profiles
                WHERE chat_hash = ?
                ORDER BY consented_at, user_hash
                """,
                (chat_hash,),
            )
            rows = await cur.fetchall()
        return [
            {
                "user_hash": str(r[0]),
                "encrypted_user_id": str(r[1]),
                "encrypted_payload": str(r[2]),
                "key_version": int(r[3]),
                "consent_version": int(r[4]),
                "consented_at": str(r[5]),
                "updated_at": str(r[6]),
            }
            for r in rows
        ]
