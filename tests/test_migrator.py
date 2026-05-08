"""Tests for app.infrastructure.migrator."""
from __future__ import annotations

import unittest

import aiosqlite

from app.infrastructure import migrator


EXPECTED_MIGRATIONS = [
    "001_initial",
    "002_normalize_messages_text_column",
    "003_anonymize_authors",
]

EXPECTED_TABLES = {
    "messages",
    "starts",
    "starts3",
    "transitions",
    "transitions3",
    "transitions1",
    "pivo_chat_members",
    "schema_migrations",
}


async def _applied_names(conn: aiosqlite.Connection) -> list[str]:
    cursor = await conn.execute("SELECT name FROM schema_migrations ORDER BY id")
    return [row[0] for row in await cursor.fetchall()]


async def _table_names(conn: aiosqlite.Connection) -> set[str]:
    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    return {row[0] for row in await cursor.fetchall()}


class TestMigratorFreshDb(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.conn = await aiosqlite.connect(":memory:")

    async def asyncTearDown(self) -> None:
        await self.conn.close()

    async def test_all_migrations_applied(self) -> None:
        await migrator.run(self.conn)
        self.assertEqual(await _applied_names(self.conn), EXPECTED_MIGRATIONS)

    async def test_all_tables_created(self) -> None:
        await migrator.run(self.conn)
        tables = await _table_names(self.conn)
        self.assertTrue(EXPECTED_TABLES.issubset(tables))

    async def test_second_run_is_idempotent(self) -> None:
        await migrator.run(self.conn)
        await migrator.run(self.conn)
        cursor = await self.conn.execute("SELECT COUNT(*) FROM schema_migrations")
        count = (await cursor.fetchone())[0]
        self.assertEqual(count, len(EXPECTED_MIGRATIONS))

    async def test_partial_migration_resumes(self) -> None:
        """If only 001 is recorded, subsequent run applies only 002 and 003."""
        await migrator._ensure_table(self.conn)
        # Apply 001 manually so migrator thinks it's done
        sql_path = migrator._MIGRATIONS_DIR / "001_initial.sql"
        await migrator._apply(self.conn, sql_path)
        await self.conn.execute(
            "INSERT INTO schema_migrations(name) VALUES ('001_initial')"
        )
        await self.conn.commit()

        await migrator.run(self.conn)
        names = await _applied_names(self.conn)
        self.assertEqual(names, EXPECTED_MIGRATIONS)


class TestMigratorExistingDb(unittest.IsolatedAsyncioTestCase):
    """Simulates a pre-existing database created by the old inline-DDL init()."""

    async def asyncSetUp(self) -> None:
        self.conn = await aiosqlite.connect(":memory:")
        # Reproduce what the old Database.init() would create
        await self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                normalized_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS starts (
                chat_id INTEGER NOT NULL,
                w1 TEXT NOT NULL, w2 TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(chat_id, w1, w2)
            );
            CREATE TABLE IF NOT EXISTS transitions (
                chat_id INTEGER NOT NULL,
                w1 TEXT NOT NULL, w2 TEXT NOT NULL, w3 TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(chat_id, w1, w2, w3)
            );
            CREATE TABLE IF NOT EXISTS starts3 (
                chat_id INTEGER NOT NULL,
                w1 TEXT NOT NULL, w2 TEXT NOT NULL, w3 TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(chat_id, w1, w2, w3)
            );
            CREATE TABLE IF NOT EXISTS transitions3 (
                chat_id INTEGER NOT NULL,
                w1 TEXT NOT NULL, w2 TEXT NOT NULL, w3 TEXT NOT NULL, w4 TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(chat_id, w1, w2, w3, w4)
            );
            CREATE TABLE IF NOT EXISTS transitions1 (
                chat_id INTEGER NOT NULL,
                w1 TEXT NOT NULL, w2 TEXT NOT NULL,
                cnt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(chat_id, w1, w2)
            );
            CREATE TABLE IF NOT EXISTS pivo_chat_members (
                chat_hash TEXT NOT NULL,
                user_hash TEXT NOT NULL,
                encrypted_user_id TEXT NOT NULL,
                encrypted_username TEXT NOT NULL DEFAULT '',
                encrypted_display_name TEXT NOT NULL DEFAULT '',
                is_bot INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY(chat_hash, user_hash)
            );
            """
        )

    async def asyncTearDown(self) -> None:
        await self.conn.close()

    async def test_migrator_runs_without_error(self) -> None:
        await migrator.run(self.conn)

    async def test_all_migrations_recorded(self) -> None:
        await migrator.run(self.conn)
        self.assertEqual(await _applied_names(self.conn), EXPECTED_MIGRATIONS)

    async def test_existing_data_preserved(self) -> None:
        await self.conn.execute(
            "INSERT INTO messages(chat_id, author_id, text, normalized_text) VALUES (1, 5, 'hello', '')"
        )
        await self.conn.commit()

        await migrator.run(self.conn)

        cursor = await self.conn.execute(
            "SELECT author_id, normalized_text FROM messages WHERE chat_id = 1"
        )
        row = await cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 0, "author_id should be anonymized")
        self.assertEqual(row[1], "hello", "normalized_text should be backfilled")
