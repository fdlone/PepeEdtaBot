"""Tests for the cumulative verbatim 4-gram index (O4 fix, migration 016)."""
from __future__ import annotations

import importlib.util
import unittest
import uuid
from pathlib import Path

import aiosqlite

from app.infrastructure.database import Database

_MIGRATION_PATH = (
    Path(__file__).parent.parent / "app" / "migrations" / "016_verbatim_ngrams.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("m016", _MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCumulativeVerbatimIndex(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_vng_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def test_learning_records_content_windows(self) -> None:
        tokens = ["кот", "опять", "уронил", "ёлку", "на", "пол"]
        await self.db.save_message_and_update_model(
            chat_id=1, raw_text="кот опять уронил ёлку на пол", tokens=tokens
        )
        ngrams = set(await self.db.get_verbatim_ngrams(1))
        self.assertIn(("кот", "опять", "уронил", "ёлку"), ngrams)
        self.assertIn(("уронил", "ёлку", "на", "пол"), ngrams)
        self.assertEqual(len(ngrams), 3)

    async def test_windows_cross_punctuation_and_casefold(self) -> None:
        tokens = ["Кот", ",", "опять", "уронил", "ёлку"]
        await self.db.save_message_and_update_model(
            chat_id=1, raw_text="Кот, опять уронил ёлку", tokens=tokens
        )
        ngrams = set(await self.db.get_verbatim_ngrams(1))
        self.assertEqual(ngrams, {("кот", "опять", "уронил", "ёлку")})

    async def test_index_survives_message_retention(self) -> None:
        db = Database(str(self.db_path) + ".retention", messages_retention_per_chat=1)
        await db.init()
        try:
            await db.save_message_and_update_model(
                chat_id=1,
                raw_text="первое сообщение про новогоднюю ёлку",
                tokens=["первое", "сообщение", "про", "новогоднюю", "ёлку"],
            )
            await db.save_message_and_update_model(
                chat_id=1,
                raw_text="совсем другое",
                tokens=["совсем", "другое"],
            )
            texts = await db.get_recent_normalized_messages(1, 10)
            self.assertEqual(len(texts), 1)  # retention trimmed the first text
            ngrams = set(await db.get_verbatim_ngrams(1))
            self.assertIn(
                ("первое", "сообщение", "про", "новогоднюю"), ngrams
            )  # ...but its 4-grams survive
        finally:
            await db.close()
            Path(str(self.db_path) + ".retention").unlink(missing_ok=True)

    async def test_clear_chat_wipes_index(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=1,
            raw_text="кот опять уронил ёлку",
            tokens=["кот", "опять", "уронил", "ёлку"],
        )
        await self.db.clear_chat(1)
        self.assertEqual(await self.db.get_verbatim_ngrams(1), [])

    async def test_migration_backfills_both_sources(self) -> None:
        path = Path(f"test_vng_mig_{uuid.uuid4().hex}.sqlite")
        try:
            async with aiosqlite.connect(path) as conn:
                await conn.execute(
                    "CREATE TABLE transitions3 (chat_id INTEGER, w1 TEXT, w2 TEXT,"
                    " w3 TEXT, w4 TEXT, cnt INTEGER)"
                )
                await conn.execute(
                    "CREATE TABLE messages (chat_id INTEGER, normalized_text TEXT)"
                )
                # All-time quadruple older than any message + one with
                # punctuation (must be skipped) + a window-crossing message.
                await conn.executemany(
                    "INSERT INTO transitions3 VALUES (1, ?, ?, ?, ?, 1)",
                    [
                        ("старая", "фраза", "из", "истории"),
                        ("хвост", ",", "с", "пунктуацией"),
                    ],
                )
                await conn.execute(
                    "INSERT INTO messages VALUES (1, 'кот, опять уронил ёлку')"
                )
                migration = _load_migration()
                await migration.apply(conn)
                cursor = await conn.execute(
                    "SELECT w1, w2, w3, w4 FROM chat_verbatim_ngrams"
                )
                rows = {tuple(r) for r in await cursor.fetchall()}
            self.assertIn(("старая", "фраза", "из", "истории"), rows)
            self.assertIn(("кот", "опять", "уронил", "ёлку"), rows)
            self.assertEqual(len(rows), 2)  # punctuation quadruple skipped
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
