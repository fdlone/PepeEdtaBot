"""Phase 5 data layer (M2R-400, TZ §9.2/§9.3, ADR-012).

The property this change ships under is negative: reverse lookups and the df
aggregate exist, agree with the forward chain by construction, and change
nothing about what the bot says. The neutrality half is pinned globally by
``tools/generation_hash.py``; here the tests pin the per-mechanism halves.
"""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

from app.core.markov import tokenize
from app.infrastructure.database import Database
from app.repositories.markov_repo import (
    REVERSE_BRANCH_SQL,
    REVERSE_TRANSITIONS_SQL,
)

CHAT = 4242


class _DatabaseCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_reverse_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)
        for suffix in ("-shm", "-wal"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)

    async def _learn(self, *texts: str) -> None:
        for text in texts:
            await self.db.save_message_and_update_model(
                chat_id=CHAT, raw_text=text, tokens=tokenize(text)
            )


class TestReverseLookups(_DatabaseCase):
    """D1: the reverse view IS the forward rows — agreement by construction,
    but the test is what keeps that true if the design ever changes."""

    async def test_reverse_rows_agree_with_forward_rows(self) -> None:
        await self._learn("а б в г", "х б в д", "у б в е")
        reverse = await self.db.markov.get_reverse_transitions(CHAT, "б", "в")
        self.assertEqual([row[0] for row in reverse], ["а", "у", "х"])
        conn = await self.db._get_conn()
        cursor = await conn.execute(
            "SELECT w1, cnt, s_value, s_updated_at FROM transitions "
            "WHERE chat_id = ? AND w2 = ? AND w3 = ? ORDER BY w1",
            (CHAT, "б", "в"),
        )
        forward = [
            (str(r[0]), int(r[1]), float(r[2]), int(r[3]))
            for r in await cursor.fetchall()
        ]
        self.assertEqual(reverse, forward)

    async def test_reverse_lookup_rides_the_index_not_a_scan(self) -> None:
        """План проверяется у запроса, который репозиторий реально выполняет.

        Прежняя редакция теста строила EXPLAIN по **рукописной копии** SQL и
        потому ничего не охраняла: на тестовой базе из одного сообщения
        планировщик выбирает индекс сам, а в проде — нет. Замер на прод-копии
        2026-08-26: без принуждения оба реверсных запроса шли сканом чата
        (1.42 и 1.16 мс), потому что `sqlite_stat1` в базе нет — `ANALYZE` не
        выполняется нигде в проекте, — и без статистики SQLite предпочитает
        PK-индекс, который бесплатно даёт `ORDER BY w1`. Утверждение
        докстринга «served by idx_transitions_reverse, never a scan» прожило
        так с миграции 020.

        Здесь тест и репозиторий читают **одну константу** с текстом запроса,
        поэтому копия разойтись не может: EXPLAIN строится ровно по тому SQL,
        который уйдёт в базу.
        """
        await self._learn("а б в г")
        conn = await self.db._get_conn()

        for sql, params in (
            (REVERSE_TRANSITIONS_SQL, (CHAT, "б", "в")),
            (REVERSE_BRANCH_SQL, (CHAT, "б")),
        ):
            cursor = await conn.execute("EXPLAIN QUERY PLAN " + sql, params)
            plan = " | ".join(str(row[3]) for row in await cursor.fetchall())
            self.assertIn(
                "idx_transitions_reverse", plan, f"скан вместо индекса: {plan}"
            )
            self.assertNotIn("SCAN transitions", plan)

    async def test_wiped_with_the_chat(self) -> None:
        await self._learn("а б в г")
        await self.db.clear_chat(CHAT)
        self.assertEqual(
            await self.db.markov.get_reverse_transitions(CHAT, "б", "в"), []
        )


class TestTokenDf(_DatabaseCase):
    """TZ §9.3: +1 per unique token per message, immune to retention."""

    async def test_unique_per_message_counting(self) -> None:
        await self._learn("кек кек кек")
        self.assertEqual(await self.db.markov.get_token_df(CHAT, "кек"), 1)
        self.assertEqual(await self.db.markov.get_n_docs(CHAT), 1)
        await self._learn("кек снова тут")
        self.assertEqual(await self.db.markov.get_token_df(CHAT, "кек"), 2)
        self.assertEqual(await self.db.markov.get_n_docs(CHAT), 2)
        self.assertEqual(await self.db.markov.get_token_df(CHAT, "нет"), 0)

    async def test_retention_trim_leaves_df_untouched(self) -> None:
        db_path = Path(f"test_reverse_{uuid.uuid4().hex}.sqlite")
        db = Database(str(db_path), messages_retention_per_chat=2)
        await db.init()
        try:
            for index in range(5):
                await db.save_message_and_update_model(
                    chat_id=CHAT,
                    raw_text=f"маркер номер {index}",
                    tokens=tokenize(f"маркер номер {index}"),
                )
            conn = await db._get_conn()
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id = ?", (CHAT,)
            )
            self.assertEqual((await cursor.fetchone())[0], 2)
            self.assertEqual(await db.markov.get_token_df(CHAT, "маркер"), 5)
            self.assertEqual(await db.markov.get_n_docs(CHAT), 5)
        finally:
            await db.close()
            db_path.unlink(missing_ok=True)

    async def test_clear_removes_df_and_message_count(self) -> None:
        await self._learn("кек лол тут")
        await self.db.clear_chat(CHAT)
        self.assertEqual(await self.db.markov.get_token_df(CHAT, "кек"), 0)
        self.assertEqual(await self.db.markov.get_n_docs(CHAT), 0)
        conn = await self.db._get_conn()
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM markov_token_df WHERE chat_id = ?", (CHAT,)
        )
        self.assertEqual((await cursor.fetchone())[0], 0)


class TestLearningAtomicity(_DatabaseCase):
    """One transaction for everything: a failed learn is durable for nothing.

    The uncommitted transaction dies with the connection, so the assertion is
    made on a fresh one — durability is the contract, not what the failed
    connection happened to see before rollback.
    """

    async def test_failed_learning_leaves_nothing_durable(self) -> None:
        with self.assertRaises(Exception):
            # None cannot bind to a NOT NULL column: learning blows up after
            # some statements already ran inside the open transaction.
            await self.db.save_message_and_update_model(
                chat_id=CHAT,
                raw_text="сломанное сообщение",
                tokens=["сломанное", "сообщение", "тут", None],  # type: ignore[list-item]
            )
        await self.db.close()
        self.db = Database(str(self.db_path))
        await self.db.init()
        conn = await self.db._get_conn()
        for table in ("messages", "transitions", "markov_token_df"):
            cursor = await conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE chat_id = ?",  # nosec B608
                (CHAT,),
            )
            self.assertEqual((await cursor.fetchone())[0], 0, table)
        self.assertEqual(await self.db.markov.get_n_docs(CHAT), 0)


class TestNothingReadsReverseInGeneration(_DatabaseCase):
    """The spec's negative requirement: no reply path touches the new data."""

    async def test_reply_pipeline_issues_no_reverse_or_df_queries(self) -> None:
        import random
        from types import SimpleNamespace

        from app import log_masking
        from app.config.registry import RUNTIME_FIELDS
        from app.core.markov import MarkovGenerator
        from app.core.response_generator import (
            GenerationRequest,
            ResponseGenerator,
        )
        from app.services.learning_service import LearningService

        log_masking.init_masking("reverse-index-test")
        await self._learn(
            "первое пробное сообщение чата",
            "второе пробное сообщение чата",
            "третье пробное сообщение чата",
        )
        for name in ("get_reverse_transitions", "get_token_df", "get_n_docs"):
            setattr(self.db.markov, name, AsyncMock(side_effect=AssertionError))
        generator = MarkovGenerator(self.db.markov)
        state = SimpleNamespace(
            **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS}
        )
        state.recent_short_replies = {}
        state.recent_replies = {}
        pipeline = ResponseGenerator(
            generator=generator,
            learning_service=LearningService(self.db, generator),
            runtime_state=state,
        )
        result = await pipeline.generate(
            GenerationRequest(
                chat_id=CHAT,
                context_tokens=["пробное", "сообщение"],
                seed=None,
                current_message_normalized="совсем другой текст",
            ),
            rng=random.Random(7),
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
