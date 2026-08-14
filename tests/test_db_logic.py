from __future__ import annotations

import asyncio
import unittest
import uuid
from datetime import date
from pathlib import Path
from unittest.mock import patch

import aiosqlite

from app.infrastructure.database import PIVO_DAILY_USAGE_RETENTION_DAYS, Database


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
        self.assertEqual(stats["volume"], 0)

    async def _sum_volume(self, table: str, chat_id: int) -> int:
        async with self.db._lock:
            conn = await self.db._get_conn()
            cur = await conn.execute(
                f"SELECT COALESCE(SUM(cnt), 0) FROM {table} WHERE chat_id = ?",
                (chat_id,),
            )
            return int((await cur.fetchone())[0] or 0)

    async def test_model_volume_matches_sum_after_many_saves(self) -> None:
        """D2: the incremental chat_model_volume stays exactly equal to SUM(cnt)."""
        chat_id = 4242
        messages = [
            ["привет", "как", "дела", "сегодня"],
            ["как", "дела", "у", "тебя"],
            ["привет", "друг"],
            ["сегодня", "хорошая", "погода", "правда"],
        ]
        last_volume = 0
        for tokens in messages:
            last_volume = await self.db.save_message_and_update_model(
                chat_id=chat_id, raw_text=" ".join(tokens), tokens=tokens
            )

        sum2 = await self._sum_volume("transitions", chat_id)
        sum3 = await self._sum_volume("transitions3", chat_id)
        stats = await self.db.get_stats(chat_id)
        self.assertEqual(stats["volume2"], sum2)
        self.assertEqual(stats["volume3"], sum3)
        # Returned volume mirrors get_stats' "volume" (prefers order-3).
        self.assertEqual(last_volume, sum3 if sum3 > 0 else sum2)

    async def test_transition_rows_are_ordered(self) -> None:
        """Предусловие горячего пути: строки приходят упорядоченными.

        Функции взвешенного выбора больше не сортируют вход — на живом чате
        это 4.5 тысячи стартов, пересортируемых на каждой попытке генерации.
        Порядок обеспечивает источник, и это должно быть проверено здесь, а не
        подразумеваться в комментарии.
        """
        chat_id = 4747
        for tokens in (
            ["яблоко", "груша", "слива", "вишня"],
            ["апельсин", "банан", "манго", "киви"],
            ["яблоко", "груша", "персик", "нектарин"],
        ):
            await self.db.save_message_and_update_model(
                chat_id=chat_id, raw_text=" ".join(tokens), tokens=tokens
            )

        starts2 = await self.db.markov.get_starts(chat_id)
        self.assertEqual(
            [(w1, w2) for w1, w2, _ in starts2],
            sorted((w1, w2) for w1, w2, _ in starts2),
        )
        starts3 = await self.db.markov.get_starts3(chat_id)
        self.assertEqual(
            [(w1, w2, w3) for w1, w2, w3, _ in starts3],
            sorted((w1, w2, w3) for w1, w2, w3, _ in starts3),
        )
        transitions = await self.db.markov.get_transitions(chat_id, "яблоко", "груша")
        self.assertEqual(
            [row[0] for row in transitions],
            sorted(row[0] for row in transitions),
        )
        self.assertGreater(len(transitions), 1)

    async def test_word_frequency_rows_are_ordered(self) -> None:
        """Словарь частот приходит упорядоченным, а не в порядке хранилища.

        Он питает розыгрыш слот-мутаций, а всякое чтение, питающее розыгрыш,
        обязано иметь тотальный порядок. Слова здесь выучены в намеренно
        неалфавитном порядке.

        Оговорка про силу теста: сегодня он прошёл бы и без ``ORDER BY`` —
        SQLite отдаёт сгруппированные строки в порядке ключа группировки при
        обоих планах, которые получает этот запрос. Тест сторожит **свойство**,
        а не конкретную клаузу: он упадёт, если запрос перепишут так, что
        порядок потеряется (другой ключ группировки, UNION, постобработка в
        Python).
        """
        chat_id = 4848
        for tokens in (
            ["раз", "два", "яблоко", "груша"],
            ["раз", "два", "апельсин", "банан"],
            ["раз", "два", "манго", "вишня"],
        ):
            await self.db.save_message_and_update_model(
                chat_id=chat_id, raw_text=" ".join(tokens), tokens=tokens
            )

        frequencies = await self.db.markov.get_word_frequencies(
            chat_id, min_word_len=3
        )
        words = list(frequencies)
        self.assertEqual(words, sorted(words))
        # Иначе тест проходил бы на пустом словаре и ничего не сторожил.
        self.assertGreater(len(words), 1)
        self.assertNotEqual(words, ["яблоко", "груша", "апельсин", "банан"][: len(words)])

    async def test_token_volume_reads_the_counter_not_the_aggregate(self) -> None:
        """Горячий путь чтения объёма не сканирует таблицы переходов.

        Вызов происходит на каждое входящее сообщение, а полный SUM(cnt) — это
        скан, растущий вместе с корпусом.
        """
        chat_id = 4444
        tokens = ["кофе", "утром", "бодрит", "правда"]
        await self.db.save_message_and_update_model(
            chat_id=chat_id, raw_text=" ".join(tokens), tokens=tokens
        )
        expected = await self._sum_volume("transitions3", chat_id)

        markov = self.db.markov
        assert markov is not None
        with patch.object(
            markov, "sum_model_volume", side_effect=AssertionError("aggregated")
        ):
            volume = await self.db.get_chat_token_volume(chat_id)

        self.assertEqual(volume, expected)

    async def test_token_volume_backfills_a_missing_counter_row(self) -> None:
        """Чат без строки счётчика получает значение агрегатом — один раз."""
        chat_id = 4545
        tokens = ["чай", "вечером", "успокаивает", "тоже"]
        await self.db.save_message_and_update_model(
            chat_id=chat_id, raw_text=" ".join(tokens), tokens=tokens
        )
        expected = await self._sum_volume("transitions3", chat_id)
        async with self.db._lock:
            conn = await self.db._get_conn()
            await conn.execute(
                "DELETE FROM chat_model_volume WHERE chat_id = ?", (chat_id,)
            )
            await conn.commit()

        first = await self.db.get_chat_token_volume(chat_id)

        self.assertEqual(first, expected)
        markov = self.db.markov
        assert markov is not None
        with patch.object(
            markov, "sum_model_volume", side_effect=AssertionError("aggregated")
        ):
            self.assertEqual(await self.db.get_chat_token_volume(chat_id), expected)

    async def test_stats_does_not_write_a_counter_row(self) -> None:
        """Диагностика остаётся read-only: побочная запись портила бы состояние."""
        chat_id = 4646
        stats = await self.db.get_stats(chat_id)

        self.assertEqual(stats["volume"], 0)
        async with self.db._lock:
            conn = await self.db._get_conn()
            cur = await conn.execute(
                "SELECT COUNT(*) FROM chat_model_volume WHERE chat_id = ?", (chat_id,)
            )
            self.assertEqual((await cur.fetchone())[0], 0)

    async def test_clear_chat_resets_model_volume(self) -> None:
        """D2: clear_chat drops the chat_model_volume row so volume restarts at 0."""
        chat_id = 4343
        tokens = ["a", "b", "c", "d", "e"]
        await self.db.save_message_and_update_model(
            chat_id=chat_id, raw_text=" ".join(tokens), tokens=tokens
        )
        self.assertGreater((await self.db.get_stats(chat_id))["volume"], 0)

        await self.db.clear_chat(chat_id)

        async with self.db._lock:
            conn = await self.db._get_conn()
            cur = await conn.execute(
                "SELECT COUNT(*) FROM chat_model_volume WHERE chat_id = ?", (chat_id,)
            )
            self.assertEqual((await cur.fetchone())[0], 0)

        # A fresh message after clear starts the running total over.
        new_volume = await self.db.save_message_and_update_model(
            chat_id=chat_id, raw_text="x y z w", tokens=["x", "y", "z", "w"]
        )
        self.assertEqual(new_volume, await self._sum_volume("transitions3", chat_id))

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

    async def test_transitions_and_starts_are_counted_for_3_2(self) -> None:
        tokens = ["Я", "очень", "люблю", "чат", "!"]
        volume = await self.db.save_message_and_update_model(
            chat_id=2002,
            raw_text="Я очень люблю чат!",
            tokens=tokens,
        )
        self.assertEqual(volume, 2)  # two trigram transitions

        starts2 = await self.db.markov.get_starts(2002)
        self.assertEqual(starts2, [("Я", "очень", 1)])
        starts3 = await self.db.markov.get_starts3(2002)
        self.assertEqual(starts3, [("Я", "очень", "люблю", 1)])

        # Rows carry both layers since M2R-200: (token, count, s_value,
        # s_updated_at). The short pair is asserted in the Phase 3 suite; here
        # only the long count matters.
        transitions2 = await self.db.markov.get_transitions(2002, "Я", "очень")
        self.assertEqual([(row[0], row[1]) for row in transitions2], [("люблю", 1)])
        transitions3 = await self.db.markov.get_transitions3(2002, "Я", "очень", "люблю")
        self.assertEqual([(row[0], row[1]) for row in transitions3], [("чат", 1)])

        stats = await self.db.get_stats(2002)
        self.assertEqual(stats["starts2"], 1)
        self.assertEqual(stats["starts3"], 1)
        self.assertEqual(stats["transitions2"], 3)
        self.assertEqual(stats["transitions3"], 2)
        self.assertEqual(stats["volume3"], 2)

    async def test_markov_query_results_have_stable_lexical_order(self) -> None:
        messages = [
            ["zeta", "start", "shared", "omega"],
            ["alpha", "start", "shared", "beta"],
            ["zeta", "start", "shared", "alpha"],
        ]
        for tokens in messages:
            await self.db.save_message_and_update_model(
                chat_id=2003,
                raw_text=" ".join(tokens),
                tokens=tokens,
            )

        self.assertEqual(
            await self.db.markov.get_starts(2003),
            [
                ("alpha", "start", 1),
                ("zeta", "start", 2),
            ],
        )
        self.assertEqual(
            await self.db.markov.get_starts3(2003),
            [
                ("alpha", "start", "shared", 1),
                ("zeta", "start", "shared", 2),
            ],
        )
        rows = await self.db.markov.get_transitions3(2003, "zeta", "start", "shared")
        self.assertEqual(
            [(row[0], row[1]) for row in rows],
            [("alpha", 1), ("omega", 1)],
        )

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
        self.assertEqual(stats["transitions2"], 0)
        self.assertEqual(stats["transitions3"], 0)

        # M2R-200: the temporal record lives on those same rows, so a wipe must
        # not leave a short layer or an observation time behind. Asserted
        # directly rather than inferred from the counts above — a future
        # temporal side table would pass the counts and fail this.
        async with self.db._lock:
            conn = await self.db._get_conn()
            for table in ("starts", "starts3", "transitions", "transitions3"):
                cur = await conn.execute(
                    f"SELECT COUNT(*) FROM {table} "  # nosec B608
                    "WHERE chat_id = ? AND (s_updated_at IS NOT NULL OR s_value > 0)",
                    (3003,),
                )
                self.assertEqual((await cur.fetchone())[0], 0, table)

    async def test_stored_text_is_normalized_on_write(self) -> None:
        """Нормализация применяется при записи, а не при чтении.

        Раньше это проверялось через `message_exists`, которая нормализовала и
        аргумент тоже; проверка сместилась на то, что реально хранится.
        """
        await self.db.save_message_and_update_model(
            chat_id=4104,
            raw_text="Привеееет   @PepeBot https://example.com",
            tokens=["Привеет"],
        )

        self.assertEqual(
            await self.db.markov.get_recent_normalized(4104, 10),
            ["Привеет"],
        )

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

        self.assertEqual(
            await self.db.markov.get_recent_normalized(4204, 10),
            ["Стаарый текст"],
        )
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
        await self.db.chat_members.upsert(
            chat_hash="chat-hash",
            user_hash="user-hash",
            encrypted_user_id="encrypted-user-id",
            encrypted_username="encrypted-username",
            encrypted_display_name="encrypted-display-name",
        )

        members = await self.db.chat_members.list_members("chat-hash")
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]["encrypted_user_id"], "encrypted-user-id")
        self.assertEqual(members[0]["encrypted_username"], "encrypted-username")
        self.assertEqual(
            members[0]["encrypted_display_name"], "encrypted-display-name"
        )

        await self.db.chat_members.upsert(
            chat_hash="chat-hash",
            user_hash="user-hash",
            encrypted_user_id="encrypted-user-id-2",
            encrypted_username="encrypted-username-2",
            encrypted_display_name="encrypted-display-name-2",
        )
        members = await self.db.chat_members.list_members("chat-hash")
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]["encrypted_user_id"], "encrypted-user-id-2")

        await self.db.chat_members.remove("chat-hash", "user-hash")
        self.assertEqual(await self.db.chat_members.list_members("chat-hash"), [])

    async def test_refresh_chat_member_updates_profile_only(self) -> None:
        await self.db.chat_members.upsert(
            chat_hash="chat-hash",
            user_hash="user-hash",
            encrypted_user_id="encrypted-user-id",
            encrypted_username="old-username",
            encrypted_display_name="old-display-name",
        )

        await self.db.chat_members.refresh_profile(
            chat_hash="chat-hash",
            user_hash="user-hash",
            encrypted_username="new-username",
            encrypted_display_name="new-display-name",
        )

        members = await self.db.chat_members.list_members("chat-hash")
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]["encrypted_username"], "new-username")
        self.assertEqual(members[0]["encrypted_display_name"], "new-display-name")
        self.assertEqual(members[0]["encrypted_user_id"], "encrypted-user-id")

    async def test_refresh_chat_member_does_not_subscribe_anyone(self) -> None:
        await self.db.chat_members.refresh_profile(
            chat_hash="chat-hash",
            user_hash="stranger-hash",
            encrypted_username="username",
            encrypted_display_name="display-name",
        )

        self.assertEqual(await self.db.chat_members.list_members("chat-hash"), [])

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
                "chat_emoji_stats",
                "chat_hot_ngrams",
                "chat_members",
                "chat_model_volume",
                "chat_user_interactions",
                "chat_verbatim_ngrams",
                "markov_collocations",
                "markov_token_df",
                "messages",
                "pivo_daily_usage",
                "pivo_pool_usage",
                "schema_migrations",
                "starts",
                "starts3",
                "transitions",
                "transitions3",
            ],
        )

        self.db = Database(str(self.db_path))
        await self.db.init()

    async def test_pivo_daily_usage_limit(self) -> None:
        result = await self.db.pivo_usage.consume_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
            limit=1,
        )
        self.assertEqual(result, (True, 1))

        result = await self.db.pivo_usage.consume_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
            limit=1,
        )
        self.assertEqual(result, (False, 1))

        result = await self.db.pivo_usage.consume_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-09",
            limit=1,
        )
        self.assertEqual(result, (True, 1))

    async def test_pivo_daily_usage_refund_restores_quota(self) -> None:
        result = await self.db.pivo_usage.consume_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
            limit=1,
        )
        self.assertEqual(result, (True, 1))

        await self.db.pivo_usage.refund_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
        )

        result = await self.db.pivo_usage.consume_daily_call(
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
            await self.db.pivo_usage.consume_daily_call(
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

    async def test_pivo_daily_usage_zero_limit_denies_first_call(self) -> None:
        """Лимит действует и на первый вызов дня: limit <= 0 — отказ без записи."""
        result = await self.db.pivo_usage.consume_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
            limit=0,
        )
        self.assertEqual(result, (False, 0))

        # Строка квоты не завелась: с нормальным лимитом вызов проходит первым.
        result = await self.db.pivo_usage.consume_daily_call(
            chat_hash="chat-hash",
            user_hash="user-hash",
            usage_day="2026-05-08",
            limit=1,
        )
        self.assertEqual(result, (True, 1))

    async def test_write_transaction_rolls_back_on_error(self) -> None:
        """Ошибка в мультистейтментной записи откатывает её целиком.

        Без rollback незакоммиченный «осколок» уехал бы в базу вместе с
        коммитом следующего вызова на общем соединении.
        """
        with self.assertRaises(RuntimeError):
            async with self.db._write_transaction() as conn:
                await conn.execute(
                    "INSERT INTO messages(chat_id, author_id, normalized_text)"
                    " VALUES (?, 0, 'осколок')",
                    (6006,),
                )
                raise RuntimeError("boom")

        # Следующая успешная запись не должна прихватить осколок.
        await self.db.save_message_and_update_model(
            chat_id=6007, raw_text="ok", tokens=["ok"]
        )
        self.assertEqual((await self.db.get_stats(6006))["messages"], 0)

    async def test_repo_transaction_rolls_back_on_error(self) -> None:
        """BaseRepo._transaction: тот же контракт отката для репозиториев."""
        repo = self.db.pivo_usage
        with self.assertRaises(RuntimeError):
            async with repo._transaction() as conn:
                await conn.execute(
                    "INSERT INTO pivo_daily_usage"
                    "(chat_hash, user_hash, usage_day, used_count)"
                    " VALUES ('c', 'u', '2026-05-08', 5)"
                )
                raise RuntimeError("boom")

        # Чужой коммит не должен материализовать фантомную квоту.
        await repo.consume_daily_call(
            chat_hash="c2", user_hash="u2", usage_day="2026-05-08", limit=1
        )
        result = await repo.consume_daily_call(
            chat_hash="c", user_hash="u", usage_day="2026-05-08", limit=3
        )
        self.assertEqual(result, (True, 1))

    async def test_concurrent_init_opens_single_connection(self) -> None:
        """Параллельный двойной init() не открывает второе (утекающее) соединение."""
        db_path = Path(f"test_db_{uuid.uuid4().hex}.sqlite")
        db = Database(str(db_path))
        real_connect = aiosqlite.connect
        calls = 0

        def counting_connect(*args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            return real_connect(*args, **kwargs)

        try:
            with patch(
                "app.infrastructure.database.aiosqlite.connect",
                side_effect=counting_connect,
            ):
                await asyncio.gather(db.init(), db.init())
            self.assertEqual(calls, 1)
        finally:
            await db.close()
            db_path.unlink(missing_ok=True)


class TestDatabaseMessageRetention(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_retention_{uuid.uuid4().hex}.sqlite")
        self.db = Database(
            str(self.db_path),
            messages_retention_per_chat=3,
            busy_timeout_ms=4321,
            wal_autocheckpoint_pages=321,
        )
        await self.db.init()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def _record(self, chat_id: int, text: str) -> None:
        await self.db.save_message_and_update_model(
            chat_id=chat_id,
            raw_text=text,
            tokens=[],
        )

    async def test_retention_keeps_exactly_most_recent_rows(self) -> None:
        for text in ("first", "second", "third", "fourth"):
            await self._record(1, text)

        self.assertEqual(
            await self.db.markov.get_recent_normalized(1, 10),
            ["second", "third", "fourth"],
        )
        self.assertEqual((await self.db.get_stats(1))["messages"], 3)

    async def test_retention_is_scoped_to_chat(self) -> None:
        await self._record(2, "other chat message")
        for text in ("a1", "a2", "a3", "a4"):
            await self._record(1, text)

        self.assertEqual(
            await self.db.markov.get_recent_normalized(2, 10),
            ["other chat message"],
        )

    async def test_configured_pragmas_are_applied(self) -> None:
        conn = await self.db._get_conn()
        busy_timeout = await (await conn.execute("PRAGMA busy_timeout;")).fetchone()
        wal_autocheckpoint = await (
            await conn.execute("PRAGMA wal_autocheckpoint;")
        ).fetchone()

        self.assertEqual(busy_timeout, (4321,))
        self.assertEqual(wal_autocheckpoint, (321,))
