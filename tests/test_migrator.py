"""Tests for app.infrastructure.migrator."""
from __future__ import annotations

import unittest
from pathlib import Path

import aiosqlite

from app.infrastructure import migrator

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


EXPECTED_MIGRATIONS = [
    "001_initial",
    "002_normalize_messages_text_column",
    "003_anonymize_authors",
    "004_chat_member_profiles",
    "005_drop_messages_text",
]

EXPECTED_TABLES = {
    "messages",
    "starts",
    "starts3",
    "transitions",
    "transitions3",
    "transitions1",
    "pivo_chat_members",
    "chat_member_profiles",
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


_LEGACY_DDL = """
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
    CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
    CREATE INDEX IF NOT EXISTS idx_messages_normalized_lookup ON messages(chat_id, normalized_text);
    CREATE INDEX IF NOT EXISTS idx_starts_lookup ON starts(chat_id, w1, w2);
    CREATE INDEX IF NOT EXISTS idx_transitions_lookup ON transitions(chat_id, w1, w2);
    CREATE INDEX IF NOT EXISTS idx_starts3_chat_id ON starts3(chat_id);
    CREATE INDEX IF NOT EXISTS idx_transitions3_lookup ON transitions3(chat_id, w1, w2, w3);
    CREATE INDEX IF NOT EXISTS idx_transitions1_lookup ON transitions1(chat_id, w1);
    CREATE INDEX IF NOT EXISTS idx_pivo_chat_members_chat_hash ON pivo_chat_members(chat_hash);
"""


class TestMigratorExistingDb(unittest.IsolatedAsyncioTestCase):
    """Simulates a pre-existing database created by the old inline-DDL init()."""

    async def asyncSetUp(self) -> None:
        self.conn = await aiosqlite.connect(":memory:")
        await self.conn.executescript(_LEGACY_DDL)

    async def asyncTearDown(self) -> None:
        await self.conn.close()

    async def test_migrator_runs_without_error(self) -> None:
        await migrator.run(self.conn)

    async def test_all_migrations_recorded(self) -> None:
        await migrator.run(self.conn)
        self.assertEqual(await _applied_names(self.conn), EXPECTED_MIGRATIONS)

    async def test_existing_data_preserved(self) -> None:
        await self.conn.execute(
            "INSERT INTO messages(chat_id, author_id, text, normalized_text)"
            " VALUES (1, 5, 'hello', '')"
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


class TestMigratorFullCompatibility(unittest.IsolatedAsyncioTestCase):
    """
    Soak test: seeds ALL tables with realistic data, runs migrator on the legacy
    schema (including the old idx_messages_normalized_lookup index), then asserts
    every row in every table is intact.
    """

    async def asyncSetUp(self) -> None:
        self.conn = await aiosqlite.connect(":memory:")
        await self.conn.executescript(_LEGACY_DDL)

        # messages: one with already-filled normalized_text, one with empty (needs backfill),
        # one with non-zero author_id (needs anonymization).
        await self.conn.executemany(
            "INSERT INTO messages(chat_id, author_id, text, normalized_text) VALUES (?,?,?,?)",
            [
                (100, 0, "кофе утром бодрит", "кофе утром бодрит"),   # already normalized
                (100, 0, "Привет @bot !", ""),                         # needs backfill
                (100, 42, "старый автор", "старый автор"),             # needs anonymization
            ],
        )

        # Markov n=2
        await self.conn.executemany(
            "INSERT INTO starts(chat_id, w1, w2, cnt) VALUES (?,?,?,?)",
            [(100, "кофе", "утром", 3), (100, "привет", "мир", 1)],
        )
        await self.conn.executemany(
            "INSERT INTO transitions(chat_id, w1, w2, w3, cnt) VALUES (?,?,?,?,?)",
            [(100, "кофе", "утром", "бодрит", 3)],
        )

        # Markov n=3
        await self.conn.executemany(
            "INSERT INTO starts3(chat_id, w1, w2, w3, cnt) VALUES (?,?,?,?,?)",
            [(100, "кофе", "утром", "бодрит", 2)],
        )
        await self.conn.executemany(
            "INSERT INTO transitions3(chat_id, w1, w2, w3, w4, cnt) VALUES (?,?,?,?,?,?)",
            [(100, "а", "кофе", "утром", "бодрит", 1)],
        )

        # Markov n=1
        await self.conn.executemany(
            "INSERT INTO transitions1(chat_id, w1, w2, cnt) VALUES (?,?,?,?)",
            [(100, "кофе", "утром", 5), (100, "утром", "бодрит", 5)],
        )

        # pivo members
        await self.conn.executemany(
            "INSERT INTO pivo_chat_members"
            "(chat_hash, user_hash, encrypted_user_id,"
            " encrypted_username, encrypted_display_name, is_bot)"
            " VALUES (?,?,?,?,?,?)",
            [
                ("chash1", "uhash1", "enc_uid_1", "enc_uname_1", "enc_dname_1", 0),
                ("chash1", "uhash2", "enc_uid_2", "", "enc_dname_2", 0),
            ],
        )

        await self.conn.commit()

    async def asyncTearDown(self) -> None:
        await self.conn.close()

    async def _count(self, table: str) -> int:
        cur = await self.conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
        return (await cur.fetchone())[0]

    async def test_row_counts_unchanged_in_all_tables(self) -> None:
        counts_before = {
            t: await self._count(t)
            for t in ("messages", "starts", "transitions", "starts3",
                      "transitions3", "transitions1", "pivo_chat_members")
        }
        await migrator.run(self.conn)
        for table, before in counts_before.items():
            after = await self._count(table)
            self.assertEqual(after, before, f"{table}: row count changed after migration")

    async def test_markov_data_intact(self) -> None:
        await migrator.run(self.conn)

        cur = await self.conn.execute(
            "SELECT cnt FROM starts WHERE chat_id=100 AND w1='кофе' AND w2='утром'"
        )
        self.assertEqual((await cur.fetchone())[0], 3)

        cur = await self.conn.execute(
            "SELECT cnt FROM transitions"
            " WHERE chat_id=100 AND w1='кофе' AND w2='утром' AND w3='бодрит'"
        )
        self.assertEqual((await cur.fetchone())[0], 3)

        cur = await self.conn.execute(
            "SELECT cnt FROM starts3 WHERE chat_id=100 AND w1='кофе' AND w2='утром' AND w3='бодрит'"
        )
        self.assertEqual((await cur.fetchone())[0], 2)

        cur = await self.conn.execute(
            "SELECT cnt FROM transitions3"
            " WHERE chat_id=100 AND w1='а' AND w2='кофе' AND w3='утром' AND w4='бодрит'"
        )
        self.assertEqual((await cur.fetchone())[0], 1)

        cur = await self.conn.execute(
            "SELECT SUM(cnt) FROM transitions1 WHERE chat_id=100"
        )
        self.assertEqual((await cur.fetchone())[0], 10)

    async def test_pivo_members_intact(self) -> None:
        await migrator.run(self.conn)

        cur = await self.conn.execute(
            "SELECT encrypted_user_id, encrypted_display_name"
            " FROM pivo_chat_members WHERE chat_hash='chash1' ORDER BY user_hash"
        )
        rows = await cur.fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "enc_uid_1")
        self.assertEqual(rows[1][1], "enc_dname_2")

    async def test_text_column_removed(self) -> None:
        await migrator.run(self.conn)

        cur = await self.conn.execute("PRAGMA table_info(messages)")
        columns = {row[1] for row in await cur.fetchall()}
        self.assertNotIn("text", columns, "text column must be removed by migration 005")
        self.assertIn("normalized_text", columns, "normalized_text must remain")

    async def test_already_normalized_text_not_overwritten(self) -> None:
        await migrator.run(self.conn)

        # First inserted row had normalized_text already set to "кофе утром бодрит"
        cur = await self.conn.execute(
            "SELECT normalized_text FROM messages WHERE chat_id=100 ORDER BY id LIMIT 1"
        )
        self.assertEqual((await cur.fetchone())[0], "кофе утром бодрит")

    async def test_empty_normalized_text_backfilled(self) -> None:
        await migrator.run(self.conn)

        # Second inserted row had normalized_text='' and needed backfill
        cur = await self.conn.execute(
            "SELECT normalized_text FROM messages WHERE chat_id=100 ORDER BY id LIMIT 1 OFFSET 1"
        )
        result = (await cur.fetchone())[0]
        self.assertNotEqual(result, "", "normalized_text must be backfilled")

    async def test_author_ids_anonymized(self) -> None:
        await migrator.run(self.conn)

        cur = await self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE author_id != 0"
        )
        self.assertEqual((await cur.fetchone())[0], 0)

    async def test_existing_index_not_dropped(self) -> None:
        """The old init created idx_messages_normalized_lookup; migrator must not remove it."""
        await migrator.run(self.conn)

        cur = await self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
            " AND name='idx_messages_normalized_lookup'"
        )
        self.assertIsNotNone(await cur.fetchone(), "index must still exist after migration")

    async def test_very_old_db_without_normalized_text_column(self) -> None:
        """
        Edge case: DB created before normalized_text was added (no column at all).
        Migration 002 must add the column, backfill it, and leave the rest untouched.
        """
        conn = await aiosqlite.connect(":memory:")
        try:
            await conn.executescript(
                """
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    author_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE starts (
                    chat_id INTEGER NOT NULL,
                    w1 TEXT NOT NULL, w2 TEXT NOT NULL,
                    cnt INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(chat_id, w1, w2)
                );
                INSERT INTO messages(chat_id, author_id, text) VALUES (200, 7, 'старый текст');
                INSERT INTO starts(chat_id, w1, w2, cnt) VALUES (200, 'старый', 'текст', 1);
                """
            )
            await migrator.run(conn)

            cur = await conn.execute(
                "SELECT normalized_text, author_id FROM messages WHERE chat_id=200"
            )
            row = await cur.fetchone()
            self.assertIsNotNone(row)
            self.assertNotEqual(row[0], "", "normalized_text must be backfilled")
            self.assertEqual(row[1], 0, "author_id must be anonymized")

            cur = await conn.execute(
                "SELECT cnt FROM starts WHERE chat_id=200 AND w1='старый' AND w2='текст'"
            )
            self.assertEqual((await cur.fetchone())[0], 1, "starts data must be preserved")
        finally:
            await conn.close()


class TestMigratorRealSchemaFixture(unittest.IsolatedAsyncioTestCase):
    """
    Loads tests/fixtures/legacy_real_schema.sql — an exact replica of the schema
    found in a real production markov.db (no normalized_text column, no
    pivo_chat_members table, idx_starts_chat_id index instead of the current set).
    Verifies that migrator handles this without data loss.
    """

    async def asyncSetUp(self) -> None:
        self.conn = await aiosqlite.connect(":memory:")
        ddl = (_FIXTURES_DIR / "legacy_real_schema.sql").read_text(encoding="utf-8")
        await self.conn.executescript(ddl)

    async def asyncTearDown(self) -> None:
        await self.conn.close()

    async def _count(self, table: str) -> int:
        cur = await self.conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
        return (await cur.fetchone())[0]

    async def test_migrator_runs_without_error(self) -> None:
        await migrator.run(self.conn)

    async def test_all_migrations_recorded(self) -> None:
        await migrator.run(self.conn)
        self.assertEqual(await _applied_names(self.conn), EXPECTED_MIGRATIONS)

    async def test_message_count_unchanged(self) -> None:
        before = await self._count("messages")
        await migrator.run(self.conn)
        self.assertEqual(await self._count("messages"), before)

    async def test_markov_row_counts_unchanged(self) -> None:
        counts = {
            t: await self._count(t)
            for t in ("starts", "starts3", "transitions", "transitions1", "transitions3")
        }
        await migrator.run(self.conn)
        for table, before in counts.items():
            self.assertEqual(await self._count(table), before, f"{table} count changed")

    async def test_normalized_text_backfilled_for_all_messages(self) -> None:
        await migrator.run(self.conn)
        cur = await self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE normalized_text = ''"
        )
        self.assertEqual((await cur.fetchone())[0], 0)

    async def test_all_author_ids_anonymized(self) -> None:
        # Fixture has non-zero author_ids (396273827, 249369775, 111222333)
        before = await self._count("messages")
        self.assertGreater(before, 0)
        await migrator.run(self.conn)
        cur = await self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE author_id != 0"
        )
        self.assertEqual((await cur.fetchone())[0], 0)

    async def test_markov_chain_values_intact(self) -> None:
        await migrator.run(self.conn)
        cur = await self.conn.execute(
            "SELECT cnt FROM starts WHERE chat_id=-1003736119498 AND w1='пойдём' AND w2='пить'"
        )
        self.assertEqual((await cur.fetchone())[0], 1)

        cur = await self.conn.execute(
            "SELECT cnt FROM transitions WHERE chat_id=-1003736119498"
            " AND w1='пойдём' AND w2='пить' AND w3='кофе'"
        )
        self.assertEqual((await cur.fetchone())[0], 1)

        cur = await self.conn.execute(
            "SELECT SUM(cnt) FROM transitions1 WHERE chat_id=-1003736119498"
        )
        self.assertEqual((await cur.fetchone())[0], 5)

    async def test_text_column_removed(self) -> None:
        await migrator.run(self.conn)
        cur = await self.conn.execute("PRAGMA table_info(messages)")
        columns = {row[1] for row in await cur.fetchall()}
        self.assertNotIn("text", columns, "text column must be removed by migration 005")
        self.assertIn("normalized_text", columns, "normalized_text must remain")

    async def test_pivo_chat_members_created_empty(self) -> None:
        await migrator.run(self.conn)
        cur = await self.conn.execute("SELECT COUNT(*) FROM pivo_chat_members")
        self.assertEqual((await cur.fetchone())[0], 0)

    async def test_legacy_index_preserved(self) -> None:
        """idx_starts_chat_id existed in old DB and must survive migration."""
        await migrator.run(self.conn)
        cur = await self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_starts_chat_id'"
        )
        self.assertIsNotNone(await cur.fetchone())

    async def test_multi_chat_isolation(self) -> None:
        """Data from two different chat_ids must remain separate after migration."""
        await migrator.run(self.conn)
        cur = await self.conn.execute(
            "SELECT COUNT(DISTINCT chat_id) FROM messages"
        )
        self.assertEqual((await cur.fetchone())[0], 2)
