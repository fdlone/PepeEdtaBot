"""Тесты доставки снимка БД (спека db-snapshot-delivery).

Все БД — временные, с синтетическими ID (§4 CLAUDE.md).
"""
from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services import db_snapshot
from app.services.db_snapshot import (
    make_snapshot,
    send_packed_snapshot,
    send_startup_snapshot,
    startup_snapshot,
)


def _make_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (chat_id INTEGER, word TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'пепе')")
    conn.commit()
    conn.close()


class SnapshotTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="test_db_snapshot_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db_path = os.path.join(self.tmp, "markov.db")


class TestMakeSnapshot(SnapshotTestCase):
    async def test_snapshot_roundtrips_the_data(self) -> None:
        _make_db(self.db_path)

        packed = await make_snapshot(self.db_path)

        assert packed is not None
        self.addCleanup(shutil.rmtree, os.path.dirname(packed), True)
        self.assertRegex(
            os.path.basename(packed), r"^markov-\d{8}-\d{6}\.db\.gz$"
        )
        restored = os.path.join(self.tmp, "restored.db")
        with gzip.open(packed, "rb") as src, open(restored, "wb") as dst:
            shutil.copyfileobj(src, dst)
        conn = sqlite3.connect(restored)
        try:
            rows = conn.execute("SELECT chat_id, word FROM t").fetchall()
        finally:
            conn.close()
        self.assertEqual(rows, [(1, "пепе")])

    async def test_missing_database_file_yields_none(self) -> None:
        # Первый запуск — не ошибка: снимать нечего.
        self.assertIsNone(await make_snapshot(self.db_path))


class TestSendPackedSnapshot(SnapshotTestCase):
    def _packed(self) -> str:
        packed_dir = tempfile.mkdtemp(prefix="pepe_db_snapshot_")
        packed = os.path.join(packed_dir, "markov-stamp.db.gz")
        with open(packed, "wb") as fh:
            fh.write(b"gzip-bytes")
        return packed

    async def test_document_is_sent_and_temp_dir_removed(self) -> None:
        packed = self._packed()
        bot = AsyncMock()

        await send_packed_snapshot(bot, 5, packed)

        bot.send_document.assert_awaited_once()
        self.assertEqual(bot.send_document.await_args.args[0], 5)
        self.assertFalse(os.path.exists(os.path.dirname(packed)))

    async def test_oversize_file_becomes_a_text_warning(self) -> None:
        packed = self._packed()
        bot = AsyncMock()

        with patch.object(db_snapshot, "TELEGRAM_DOCUMENT_LIMIT_BYTES", 1):
            await send_packed_snapshot(bot, 5, packed)

        bot.send_document.assert_not_awaited()
        bot.send_message.assert_awaited_once()
        self.assertFalse(os.path.exists(os.path.dirname(packed)))

    async def test_temp_dir_removed_even_when_sending_fails(self) -> None:
        packed = self._packed()
        bot = AsyncMock()
        bot.send_document.side_effect = RuntimeError("chat not found")

        with self.assertRaises(RuntimeError):
            await send_packed_snapshot(bot, 5, packed)

        self.assertFalse(os.path.exists(os.path.dirname(packed)))


class TestStartupSnapshot(SnapshotTestCase):
    def _settings(self, **overrides: object) -> SimpleNamespace:
        base: dict[str, object] = {
            "db_snapshot_to_owner": True,
            "owner_id": 5,
            "db_path": self.db_path,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    async def test_enabled_with_database_returns_a_snapshot(self) -> None:
        _make_db(self.db_path)

        packed = await startup_snapshot(self._settings())

        assert packed is not None
        self.addCleanup(shutil.rmtree, os.path.dirname(packed), True)
        self.assertTrue(os.path.exists(packed))

    async def test_disabled_knob_skips_silently(self) -> None:
        _make_db(self.db_path)
        self.assertIsNone(
            await startup_snapshot(self._settings(db_snapshot_to_owner=False))
        )

    async def test_missing_owner_skips_silently(self) -> None:
        _make_db(self.db_path)
        self.assertIsNone(await startup_snapshot(self._settings(owner_id=None)))

    async def test_first_start_without_database_skips_silently(self) -> None:
        self.assertIsNone(await startup_snapshot(self._settings()))

    async def test_snapshot_error_is_a_warning_not_a_crash(self) -> None:
        # Отказ бэкапа не имеет права стоить запуска (спека, D3).
        with patch(
            "app.services.db_snapshot.make_snapshot",
            side_effect=RuntimeError("db is locked"),
        ):
            with self.assertLogs("chat_markov", level="WARNING"):
                result = await startup_snapshot(self._settings())
        self.assertIsNone(result)


class TestSendStartupSnapshot(SnapshotTestCase):
    async def test_delivery_failure_is_swallowed_with_a_warning(self) -> None:
        # Типичный случай — владелец не начинал диалог с ботом.
        packed_dir = tempfile.mkdtemp(prefix="pepe_db_snapshot_")
        packed = os.path.join(packed_dir, "markov-stamp.db.gz")
        with open(packed, "wb") as fh:
            fh.write(b"gzip-bytes")
        bot = AsyncMock()
        bot.send_document.side_effect = RuntimeError("chat not found")

        with self.assertLogs("chat_markov", level="WARNING"):
            await send_startup_snapshot(bot, 5, packed)

        self.assertFalse(os.path.exists(packed_dir))


if __name__ == "__main__":
    unittest.main()
