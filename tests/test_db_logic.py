from __future__ import annotations

import unittest
import uuid
from pathlib import Path

import aiosqlite

from db import Database


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
            author_id=77,
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

    async def test_transitions_and_starts_are_counted_for_3_2_1(self) -> None:
        tokens = ["Я", "очень", "люблю", "чат", "!"]
        volume = await self.db.save_message_and_update_model(
            chat_id=2002,
            author_id=88,
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
            author_id=99,
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
            author_id=11,
            raw_text="hello world",
            tokens=["hello", "world"],
        )
        self.assertTrue(await self.db.message_exists(4004, "hello world"))
        self.assertFalse(await self.db.message_exists(4004, "hello"))

    async def test_reopen_existing_database_preserves_chat_data(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=5005,
            author_id=21,
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

    async def test_schema_stays_compatible_without_new_tables(self) -> None:
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
            ["messages", "starts", "starts3", "transitions", "transitions1", "transitions3"],
        )

        self.db = Database(str(self.db_path))
        await self.db.init()
