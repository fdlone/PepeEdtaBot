from __future__ import annotations

import unittest
import uuid
from datetime import date
from pathlib import Path

import aiosqlite

from db import PIVO_DAILY_USAGE_RETENTION_DAYS, Database


class TestDatabaseLogic(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_db_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def test_save_raw_message_and_update_model(self) -> None:
        raw_text = "Привеееет   https://example.com"
        tokens = ["Привеет"]

        volume = await self.db.save_message_and_update_model(
            chat_id=1001,
            raw_text=raw_text,
            tokens=tokens,
        )
        self.assertEqual(volume, 0)

        stats = await self.db.get_stats(1001)
        self.assertEqual(stats["messages"], 1)
        self.assertEqual(stats["starts2"], 0)
        self.assertEqual(stats["starts3"], 0)
        self.assertEqual(stats["transitions2"], 0)
        self.assertEqual(stats["transitions3"], 0)
        self.assertEqual(stats["transitions1"], 0)
        self.assertEqual(stats["volume"], 0)

    async def test_save_message_does_not_store_real_author_id(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=1002,
            raw_text="privacy first",
            tokens=["privacy", "first"],
        )

        async with aiosqlite.connect(str(self.db_path)) as conn:
            row = await (
                await conn.execute(
                    "SELECT author_id FROM messages WHERE chat_id = ?",
                    (1002,),
                )
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 0)

    async def test_transitions_and_starts_are_counted_for_3_2_1(self) -> None:
        tokens = ["Я", "очень", "люблю", "чат", "!"]
        volume = await self.db.save_message_and_update_model(
            chat_id=2002,
            raw_text="Я очень люблю чат!",
            tokens=tokens,
        )
        self.assertEqual(volume, 2)  # two trigram transitions

        starts2 = await self.db.get_starts(2002)
        self.assertEqual(starts2, [("Я", "очень", 1)])
        starts3 = await self.db.get_starts3(2002)
        self.assertEqual(starts3, [("Я", "очень", "люблю", 1)])

        transitions2 = await self.db.get_transitions(2002, "Я", "очень")
        self.assertEqual(transitions2, [("люблю", 1)])
        transitions3 = await self.db.get_transitions3(2002, "Я", "очень", "люблю")
        self.assertEqual(transitions3, [("чат", 1)])
        transitions1 = await self.db.get_transitions1(2002, "чат")
        self.assertEqual(transitions1, [("!", 1)])

        stats = await self.db.get_stats(2002)
        self.assertEqual(stats["starts2"], 1)
        self.assertEqual(stats["starts3"], 1)
        self.assertEqual(stats["transitions2"], 3)
        self.assertEqual(stats["transitions3"], 2)
        self.assertEqual(stats["transitions1"], 4)
        self.assertEqual(stats["volume3"], 2)

    async def test_clear_chat(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=3003,
            raw_text="a b c d",
            tokens=["a", "b", "c", "d"],
        )
        await self.db.clear_chat(3003)
        stats = await self.db.get_stats(3003)
        self.assertEqual(
            stats["messages"],
            0,
        )
        self.assertEqual(stats["starts2"], 0)
        self.assertEqual(stats["starts3"], 0)
        self.assertEqual(stats["transitions1"], 0)
        self.assertEqual(stats["transitions2"], 0)
        self.assertEqual(stats["transitions3"], 0)

    async def test_message_exists(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=4004,
            raw_text="hello world",
            tokens=["hello", "world"],
        )
        self.assertTrue(await self.db.message_exists(4004, "hello world"))
        self.assertFalse(await self.db.message_exists(4004, "hello"))

    async def test_message_exists_uses_normalized_text(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=4104,
            raw_text="Привеееет   @PepeBot https://example.com",
            tokens=["Привеет"],
        )

        self.assertTrue(await self.db.message_exists(4104, "Привеет"))
        self.assertTrue(await self.db.message_exists(4104, "Привеееет @AnotherBot"))

    async def test_init_migrates_legacy_messages_without_normalized_text(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute(
                """
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    author_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                """
            )
            await conn.execute(
                """
                INSERT INTO messages(chat_id, author_id, text)
                VALUES (?, ?, ?)
                """,
                (4204, 13, "Стаааарый   текст @bot https://example.com"),
            )
            await conn.commit()

        self.db = Database(str(self.db_path))
        await self.db.init()

        self.assertTrue(await self.db.message_exists(4204, "Стаарый текст"))
        async with aiosqlite.connect(str(self.db_path)) as conn:
            row = await (
                await conn.execute(
                    "SELECT author_id FROM messages WHERE chat_id = ?",
                    (4204,),
                )
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 0)

    async def test_reopen_existing_database_preserves_chat_data(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=5005,
            raw_text="кофе утром бодрит",
            tokens=["кофе", "утром", "бодрит"],
        )
        before = await self.db.get_stats(5005)
        await self.db.close()

        reopened = Database(str(self.db_path))
        await reopened.init()
        try:
            after = await reopened.get_stats(5005)
            self.assertEqual(after, before)
        finally:
            await reopened.close()
            self.db = reopened

    async def test_chat_member_upsert_get_and_remove(self) -> None:
        await self.db.upsert_chat_member(
            chat_hash="chat-hash",
            user_hash="user-hash",
            encrypted_user_id="encrypted-user-id",
            encrypted_username="encrypted-username",
            encrypted_display_name="encrypted-display-name",
        )

        members = await self.db.get_chat_members("chat-hash")
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]["encrypted_user_id"], "encrypted-user-id")
        self.assertEqual(members[0]["encrypted_username"], "encrypted-username")
        self.assertEqual(
            members[0]["encrypted_display_name"], "encrypted-display-name"
        )

        await self.db.upsert_chat_member(
            chat_hash="chat-hash",
            user_hash="user-hash",
            encrypted_user_id="encrypted-user-id-2",
            encrypted_username="encrypted-username-2",
            encrypted_display_name="encrypted-display-name-2",
        )
        members = await self.db.get_chat_members("chat-hash")
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]["encrypted_user_id"], "encrypted-user-id-2")

        await self.db.remove_chat_member("chat-hash", "user-hash")
        self.assertEqual(await self.db.get_chat_members("chat-hash"), [])

    async def test_schema_contains_expected_tables(self) -> None:
        await self.db.close()
        async with aiosqlite.connect(str(self.db_path)) as conn:
            cursor = await conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
            tables = [row[0] for row in await cursor.fetchall()]

        self.assertEqual(
            tables,
            [
                "chat_members",
                "messages",
                "pivo_daily_usage",
                "schema_migrations",
                "starts",
                "starts3",
                "transitions",
                "transitions1",
                "transitions3",
            ],
        )

        self.db = Database(str(self.db_path))
        await self.db.init()

    async def test_pivo_daily_usage_limit(self) -> None:
        result = await self.db.consume_pivo_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
            limit=1,
        )
        self.assertEqual(result, (True, 1))

        result = await self.db.consume_pivo_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
            limit=1,
        )
        self.assertEqual(result, (False, 1))

        result = await self.db.consume_pivo_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-09",
            limit=1,
        )
        self.assertEqual(result, (True, 1))

    async def test_pivo_daily_usage_refund_restores_quota(self) -> None:
        result = await self.db.consume_pivo_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
            limit=1,
        )
        self.assertEqual(result, (True, 1))

        await self.db.refund_pivo_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
        )

        result = await self.db.consume_pivo_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
            limit=1,
        )
        self.assertEqual(result, (True, 1))

    async def test_pivo_daily_usage_cleanup_deletes_only_rows_older_than_retention(self) -> None:
        rows = [
            ("chat-hash", "old-user", "2026-05-01"),
            ("chat-hash", "cutoff-user", "2026-05-02"),
            ("chat-hash", "recent-user", "2026-05-08"),
        ]
        for chat_hash, user_hash, usage_day in rows:
            await self.db.consume_pivo_daily_call(
                chat_hash=chat_hash,
                user_hash=user_hash,
                usage_day=usage_day,
                limit=3,
            )

        deleted_count = await self.db.cleanup_pivo_daily_usage(
            retention_days=PIVO_DAILY_USAGE_RETENTION_DAYS,
            today=date(2026, 5, 9),
        )

        self.assertEqual(deleted_count, 1)
        async with aiosqlite.connect(str(self.db_path)) as conn:
            cursor = await conn.execute(
                "SELECT user_hash FROM pivo_daily_usage ORDER BY user_hash"
            )
            users = [row[0] for row in await cursor.fetchall()]

        self.assertEqual(users, ["cutoff-user", "recent-user"])
