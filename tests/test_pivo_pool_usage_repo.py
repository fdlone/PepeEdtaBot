from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from app.infrastructure.database import Database
from app.repositories.pivo_pool_usage_repo import _parse_indices


class TestParseIndices(unittest.TestCase):
    def test_parses_csv_and_skips_blanks_and_garbage(self) -> None:
        self.assertEqual(_parse_indices("1,2,3"), (1, 2, 3))
        self.assertEqual(_parse_indices(""), ())
        self.assertEqual(_parse_indices(" 4 , , 5 "), (4, 5))
        self.assertEqual(_parse_indices("1,x,2"), (1, 2))


class TestPivoPoolUsageRepo(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_pool_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def test_empty_chat_returns_no_history(self) -> None:
        self.assertEqual(await self.db.get_pivo_pool_usage("chatA"), {})

    async def test_record_appends_and_reads_back_per_pool(self) -> None:
        await self.db.record_pivo_pool_usage(
            "chatA", {"default_top": 3, "default_body": 7}, keep=5
        )
        await self.db.record_pivo_pool_usage("chatA", {"default_top": 4}, keep=5)

        recent = await self.db.get_pivo_pool_usage("chatA")
        self.assertEqual(recent["default_top"], (3, 4))
        self.assertEqual(recent["default_body"], (7,))

    async def test_history_is_trimmed_to_keep_most_recent(self) -> None:
        for index in range(6):
            await self.db.record_pivo_pool_usage(
                "chatA", {"default_top": index}, keep=3
            )
        recent = await self.db.get_pivo_pool_usage("chatA")
        self.assertEqual(recent["default_top"], (3, 4, 5))

    async def test_keep_zero_is_noop(self) -> None:
        await self.db.record_pivo_pool_usage("chatA", {"default_top": 1}, keep=0)
        self.assertEqual(await self.db.get_pivo_pool_usage("chatA"), {})

    async def test_histories_are_isolated_per_chat(self) -> None:
        await self.db.record_pivo_pool_usage("chatA", {"default_top": 1}, keep=5)
        await self.db.record_pivo_pool_usage("chatB", {"default_top": 9}, keep=5)
        self.assertEqual((await self.db.get_pivo_pool_usage("chatA"))["default_top"], (1,))
        self.assertEqual((await self.db.get_pivo_pool_usage("chatB"))["default_top"], (9,))


if __name__ == "__main__":
    unittest.main()
