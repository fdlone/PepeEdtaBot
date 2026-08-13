from __future__ import annotations

import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.infrastructure.database import Database

CHAT = 4242
BIGRAM = ("крутой", "бобёр")
TRIGRAM = ("крутой", "бобёр", "пришёл")


class TestChatHotNgramsRepo(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_hot_ngrams_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def _seed_long_term_counts(self, chat_id: int, cnt: int) -> None:
        """Plant all-time model counts the hotness ratio compares against.

        A single transitions row seeds both denominators: the trigram uses it
        directly, the bigram aggregates SUM(cnt) over its (w1, w2) prefix.
        """
        conn = await self.db._get_conn()
        await conn.execute(
            "INSERT INTO transitions(chat_id, w1, w2, w3, cnt) VALUES (?, ?, ?, ?, ?)",
            (chat_id, *TRIGRAM, cnt),
        )
        await conn.commit()

    async def test_empty_chat_has_no_hot_ngrams(self) -> None:
        hot = await self.db.chat_hot_ngrams.get_hot(CHAT, min_count=1, recency_share=0.0)
        self.assertEqual(hot, [])

    async def test_bump_accumulates_and_high_share_is_hot(self) -> None:
        await self._seed_long_term_counts(CHAT, 10)
        # Window count 8 of 10 all-time -> share 0.8, hot at threshold 0.5.
        for _ in range(8):
            await self.db.chat_hot_ngrams.bump(CHAT, [BIGRAM, TRIGRAM])
        hot = await self.db.chat_hot_ngrams.get_hot(CHAT, min_count=3, recency_share=0.5)
        self.assertIn(BIGRAM, hot)
        self.assertIn(TRIGRAM, hot)

    async def test_low_share_is_not_hot(self) -> None:
        await self._seed_long_term_counts(CHAT, 10)
        # Window count 3 of 10 all-time -> share 0.3 < 0.5.
        await self.db.chat_hot_ngrams.bump(CHAT, [BIGRAM] * 3)
        hot = await self.db.chat_hot_ngrams.get_hot(CHAT, min_count=3, recency_share=0.5)
        self.assertEqual(hot, [])

    async def test_below_min_count_is_not_hot(self) -> None:
        await self.db.chat_hot_ngrams.bump(CHAT, [BIGRAM] * 2)
        hot = await self.db.chat_hot_ngrams.get_hot(CHAT, min_count=3, recency_share=0.5)
        self.assertEqual(hot, [])

    async def test_new_ngram_without_long_term_row_qualifies(self) -> None:
        # No transitions row at all -> share falls back to 1.0 (brand-new meme).
        await self.db.chat_hot_ngrams.bump(CHAT, [TRIGRAM] * 4)
        hot = await self.db.chat_hot_ngrams.get_hot(CHAT, min_count=3, recency_share=0.5)
        self.assertEqual(hot, [TRIGRAM])

    async def test_hot_ngrams_are_per_chat(self) -> None:
        await self.db.chat_hot_ngrams.bump(CHAT, [BIGRAM] * 5)
        hot_other = await self.db.chat_hot_ngrams.get_hot(999, min_count=1, recency_share=0.0)
        self.assertEqual(hot_other, [])

    async def test_bump_ignores_empty_and_wrong_sizes(self) -> None:
        await self.db.chat_hot_ngrams.bump(CHAT, [])
        await self.db.chat_hot_ngrams.bump(CHAT, [("одно",), ("а", "б", "в", "г")])
        hot = await self.db.chat_hot_ngrams.get_hot(CHAT, min_count=1, recency_share=0.0)
        self.assertEqual(hot, [])

    async def test_clear_chat_wipes_hot_ngrams(self) -> None:
        await self.db.chat_hot_ngrams.bump(CHAT, [BIGRAM] * 4)
        await self.db.clear_chat(CHAT)
        hot = await self.db.chat_hot_ngrams.get_hot(CHAT, min_count=1, recency_share=0.0)
        self.assertEqual(hot, [])

    async def test_decay_halves_stale_rows_and_drops_zeros(self) -> None:
        await self.db.chat_hot_ngrams.bump(CHAT, [BIGRAM] * 4)
        await self.db.chat_hot_ngrams.bump(CHAT, [TRIGRAM])
        # Decay as if 30 days have passed: 4→2 (kept), 1→0 (purged).
        future = datetime.now(UTC) + timedelta(days=30)
        deleted = await self.db.decay_chat_hot_ngrams(now=future)
        self.assertEqual(deleted, 1)
        hot = await self.db.chat_hot_ngrams.get_hot(CHAT, min_count=1, recency_share=0.0)
        self.assertEqual(hot, [BIGRAM])

    async def test_decay_leaves_fresh_rows_untouched(self) -> None:
        await self.db.chat_hot_ngrams.bump(CHAT, [BIGRAM] * 4)
        await self.db.decay_chat_hot_ngrams()
        hot = await self.db.chat_hot_ngrams.get_hot(CHAT, min_count=4, recency_share=0.0)
        self.assertEqual(hot, [BIGRAM])

    async def test_get_hot_uses_indexes_not_full_scans(self) -> None:
        """Perf guard: every lookup in get_hot must ride a PK/index search."""
        await self.db.chat_hot_ngrams.bump(CHAT, [BIGRAM, TRIGRAM])
        conn = await self.db._get_conn()
        cursor = await conn.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT h.w1, h.w2, h.w3, h.cnt
            FROM chat_hot_ngrams h
            LEFT JOIN transitions t
              ON t.chat_id = h.chat_id
             AND t.w1 = h.w1 AND t.w2 = h.w2 AND t.w3 = h.w3
            WHERE h.chat_id = ? AND h.w3 <> '' AND h.cnt >= ?
            """,
            (CHAT, 1),
        )
        plan_rows = await cursor.fetchall()
        plan = " | ".join(str(row[3]) for row in plan_rows)
        self.assertNotIn("SCAN transitions", plan)


class TestHotNgramTieOrder(unittest.IsolatedAsyncioTestCase):
    """Равные счётчики упорядочены по самой n-грамме, а не как выйдет.

    Результат ``get_hot`` идёт в ``random.choice`` на пути ответа, поэтому
    неопределённый тай-брейк — это неопределённый розыгрыш. На замороженном
    снимке при дефолтных порогах чтение возвращает пусто, так что данными это
    не проверяется — только синтетикой с намеренно равными счётчиками.

    Оговорка про силу теста: сегодня он проходит и без тай-брейка в запросе —
    план идёт по первичному ключу, а его порядок совпадает с порядком
    n-граммы. Тест закрепляет наблюдаемое свойство, но не отличает «держится
    клаузой» от «держится планом»; форсировать второй случай из теста нечем.
    Ценность в том, что смена плана (новый индекс, другой набор колонок) или
    потеря клаузы при переписывании запроса станет видна здесь.
    """

    PAIRS = [("ямка", "яма"), ("арка", "аре"), ("мель", "мели")]

    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_hot_tie_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        # Вставляются в неалфавитном порядке и с одинаковым счётчиком: любой
        # порядок, кроме заданного тай-брейком, был бы порядком вставки.
        for pair in self.PAIRS:
            await self.db.chat_hot_ngrams.bump(CHAT, [pair] * 4)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def test_equal_counts_are_ordered_by_the_ngram(self) -> None:
        hot = await self.db.chat_hot_ngrams.get_hot(
            CHAT, min_count=3, recency_share=0.0
        )
        self.assertEqual(len(hot), len(self.PAIRS))
        self.assertEqual(hot, sorted(hot))

    async def test_meme_ordering_path_breaks_ties_the_same_way(self) -> None:
        hot = await self.db.chat_hot_ngrams.get_hot(
            CHAT, min_count=3, recency_share=0.0, meme_ordering=True
        )
        # Реестр коллокаций пуст, значит score одинаков у всех и решает
        # тот же тай-брейк, что на дефолтном пути.
        self.assertEqual(hot, sorted(hot))


class TestMemeOrdering(unittest.IsolatedAsyncioTestCase):
    """M2R-310 / design D5: an ordering behind a knob, not a replacement."""

    LOUD = ("просто", "частая")
    MEME = ("крутой", "бобёр")

    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_hot_meme_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        # The frequent pair outcounts the meme in the window; the registry
        # holds only the meme as active.
        await self.db.chat_hot_ngrams.bump(CHAT, [self.LOUD] * 9)
        await self.db.chat_hot_ngrams.bump(CHAT, [self.MEME] * 4)
        await self.db.collocations.replace_scored(
            CHAT, [(*self.MEME, 4, 0.93, 0.85)]
        )

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def _hot(self, *, meme_ordering: bool) -> list[tuple[str, ...]]:
        return await self.db.chat_hot_ngrams.get_hot(
            CHAT, min_count=3, recency_share=0.0, meme_ordering=meme_ordering
        )

    async def test_default_path_still_orders_by_count(self) -> None:
        self.assertEqual(await self._hot(meme_ordering=False), [self.LOUD, self.MEME])

    async def test_meme_ordering_puts_the_registry_pair_first(self) -> None:
        self.assertEqual(await self._hot(meme_ordering=True), [self.MEME, self.LOUD])

    async def test_retired_entry_stops_influencing_the_order(self) -> None:
        await self.db.collocations.set_status(CHAT, *self.MEME, "retired")
        self.assertEqual(await self._hot(meme_ordering=True), [self.LOUD, self.MEME])


class TestPlantedSpikeEndToEnd(unittest.IsolatedAsyncioTestCase):
    """L1 acceptance check from the action plan: a phrase the chat suddenly
    picked up becomes hot and the seed API opens a reply with it."""

    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_hot_spike_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def test_planted_spike_becomes_hot_and_seeds_generation(self) -> None:
        from app.core.hot_ngrams import extract_content_ngrams
        from app.core.markov import MarkovGenerator, tokenize

        filler = [
            "сегодня отличная погода на улице",
            "пойдём вечером гулять в парк",
            "кто смотрел вчера новый фильм",
            "надо доделать проект до пятницы",
            "заказали пиццу на всю компанию",
            "опять понедельник и куча работы",
        ]
        planted = "крутой бобёр пришёл в чат"
        corpus = filler * 5 + [planted] * 6

        # Learn the corpus exactly the way the handler does: model update plus
        # hot-ngram window per message.
        for text in corpus:
            tokens = tokenize(text)
            await self.db.save_message_and_update_model(
                chat_id=CHAT, raw_text=text, tokens=tokens
            )
            ngrams = extract_content_ngrams(tokens)
            if ngrams:
                await self.db.chat_hot_ngrams.bump(CHAT, ngrams)

        hot = await self.db.chat_hot_ngrams.get_hot(CHAT, min_count=3, recency_share=0.5)
        planted_hot = [ng for ng in hot if ng[:2] == ("крутой", "бобёр")]
        self.assertTrue(
            planted_hot,
            f"planted n-gram not detected as hot; hot={hot!r}",
        )

        generator = MarkovGenerator(self.db.markov)
        text = await generator.generate_text(
            chat_id=CHAT,
            max_chars=280,
            max_tokens=45,
            seed_tokens=list(planted_hot[0]),
        )
        self.assertTrue(text)
        self.assertTrue(
            text.lower().startswith("крутой бобёр"),
            f"seeded reply does not open with the planted phrase: {text!r}",
        )


if __name__ == "__main__":
    unittest.main()
