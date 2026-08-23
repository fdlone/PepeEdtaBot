"""Фразовый индекс: производное представление цепи (derive-phrase-index).

Свойство, которое здесь закрепляется, — индекс нельзя отличить от прямого
отбора фраз по `transitions`. Всё остальное (порог, порядок, стирание) —
следствия того же: у индекса нет собственной истории, только пересчёт.
"""

from __future__ import annotations

import unittest
import uuid
from collections import Counter
from pathlib import Path

from app.core.hot_ngrams import is_content_ngram
from app.infrastructure.database import Database
from app.repositories import ChatPhraseNgramsRepo

# Синтетические ID: инвариант tests/test_no_real_chat_ids.
CHAT = 4242
OTHER_CHAT = 999


class PhraseIndexTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_phrases_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        self.repo: ChatPhraseNgramsRepo = self.db.chat_phrase_ngrams

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def _chain(self, chat_id: int, rows: list[tuple[str, str, str, int]]) -> None:
        """Положить строки прямо в `transitions` — цепь как данность."""
        async with self.db._lock:
            conn = await self.db._get_conn()
            await conn.executemany(
                "INSERT INTO transitions(chat_id, w1, w2, w3, cnt)"
                " VALUES (?, ?, ?, ?, ?)",
                [(chat_id, *row) for row in rows],
            )
            await conn.commit()

    async def _wipe_chain(self, chat_id: int) -> None:
        async with self.db._lock:
            conn = await self.db._get_conn()
            await conn.execute("DELETE FROM transitions WHERE chat_id = ?", (chat_id,))
            await conn.commit()

    async def _stored(self, chat_id: int) -> dict[tuple[str, ...], int]:
        return dict(await self.repo.get_phrases(chat_id, min_count=1))


class TestRebuildFromTheChain(PhraseIndexTestCase):
    async def test_trigram_is_a_chain_row_and_bigram_is_its_group(self) -> None:
        await self._chain(
            CHAT,
            [
                ("бобёр", "сломал", "сборку", 2),
                ("бобёр", "сломал", "прогон", 3),
                ("бобёр", "сломал", ".", 1),
            ],
        )

        rows = await self.repo.rebuild_chat(CHAT)

        # Триграмма с пунктуацией отсеяна, но её вхождения — часть биграммы:
        # пара встретилась шесть раз, чем бы она ни кончалась.
        self.assertEqual(
            await self._stored(CHAT),
            {
                ("бобёр", "сломал", "сборку"): 2,
                ("бобёр", "сломал", "прогон"): 3,
                ("бобёр", "сломал"): 6,
            },
        )
        self.assertEqual(rows, 3)

    async def test_rebuild_is_idempotent(self) -> None:
        await self._chain(CHAT, [("бобёр", "сломал", "сборку", 2)])

        await self.repo.rebuild_chat(CHAT)
        first = await self._stored(CHAT)
        await self.repo.rebuild_chat(CHAT)

        self.assertEqual(await self._stored(CHAT), first)

    async def test_rebuild_replaces_rather_than_adds(self) -> None:
        await self._chain(CHAT, [("бобёр", "сломал", "сборку", 2)])
        await self.repo.rebuild_chat(CHAT)
        await self._wipe_chain(CHAT)
        await self._chain(CHAT, [("бобёр", "починил", "сборку", 1)])

        await self.repo.rebuild_chat(CHAT)

        stored = await self._stored(CHAT)
        self.assertNotIn(("бобёр", "сломал", "сборку"), stored)
        self.assertIn(("бобёр", "починил", "сборку"), stored)

    async def test_index_matches_a_direct_scan_of_the_chain(self) -> None:
        """Согласованность с цепью — главное свойство индекса."""
        chain = [
            ("бобёр", "опять", "сломал", 4),
            ("опять", "сломал", "сборку", 3),
            ("сломал", "сборку", ".", 2),
            ("он", "же", "не", 5),
            ("сборку", "а", "не", 1),
        ]
        await self._chain(CHAT, chain)

        await self.repo.rebuild_chat(CHAT)

        expected: Counter[tuple[str, ...]] = Counter()
        for w1, w2, w3, cnt in chain:
            if is_content_ngram((w1, w2, w3)):
                expected[(w1, w2, w3)] += cnt
            if is_content_ngram((w1, w2)):
                expected[(w1, w2)] += cnt
        self.assertEqual(await self._stored(CHAT), dict(expected))

    async def test_a_chat_without_a_chain_gets_an_empty_index(self) -> None:
        self.assertEqual(await self.repo.rebuild_chat(CHAT), 0)
        self.assertEqual(await self._stored(CHAT), {})

    async def test_rebuild_touches_only_its_own_chat(self) -> None:
        await self._chain(CHAT, [("бобёр", "сломал", "сборку", 1)])
        await self._chain(OTHER_CHAT, [("другой", "чат", "тут", 1)])

        await self.repo.rebuild_chat(CHAT)

        self.assertEqual(await self._stored(OTHER_CHAT), {})


class TestReadingByThreshold(PhraseIndexTestCase):
    async def test_threshold_is_checked_in_the_point(self) -> None:
        await self._chain(
            CHAT,
            [
                ("бобёр", "сломал", "сборку", 3),
                ("ёжик", "починил", "прогон", 2),
            ],
        )
        await self.repo.rebuild_chat(CHAT)

        phrases = dict(await self.repo.get_phrases(CHAT, min_count=3))

        # Порог берётся включительно: счётчик ровно 3 проходит, 2 — нет.
        self.assertIn(("бобёр", "сломал", "сборку"), phrases)
        self.assertNotIn(("ёжик", "починил", "прогон"), phrases)

    async def test_ties_are_ordered_reproducibly(self) -> None:
        """Порядок пойдёт в розыгрыш маршрута — он обязан быть определён.

        Честная оговорка о силе этой проверки: снятие тай-брейка
        (`ORDER BY cnt DESC` без ключей) её **не роняет**, и это проверено
        мутацией, а не предположено. Причина — та же, что у
        `get_word_frequencies` в пакете W1-C: план запроса идёт
        `SEARCH ... USING PRIMARY KEY`, ключ таблицы и есть `(w1, w2, w3)`, и
        сортировка по `cnt DESC` поверх уже упорядоченного входа отдаёт тот же
        порядок. То есть тай-брейк сегодня тождественен.

        Он всё равно записан явно: стабильность сортировщика SQLite не
        гарантирована контрактом, план может смениться от индекса или версии, а
        цена ошибки — недетерминированный розыгрыш у будущего маршрута. Тест
        закрепляет наблюдаемое следствие (порядок полностью определён и
        воспроизводим), но отличить его от случайного совпадения с планом не
        может — как не мог и гард `generation_hash` в своё время.
        """
        await self._chain(
            CHAT,
            [
                ("яблоко", "красное", "спелое", 5),
                ("банан", "жёлтый", "спелый", 5),
                ("вишня", "тёмная", "спелая", 5),
            ],
        )
        await self.repo.rebuild_chat(CHAT)

        first = await self.repo.get_phrases(CHAT, min_count=1)
        second = await self.repo.get_phrases(CHAT, min_count=1)

        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first, key=lambda item: (-item[1], item[0])))

    async def test_bigrams_and_trigrams_come_back_with_their_own_shape(self) -> None:
        await self._chain(CHAT, [("бобёр", "сломал", "сборку", 1)])
        await self.repo.rebuild_chat(CHAT)

        phrases = dict(await self.repo.get_phrases(CHAT, min_count=1))

        self.assertEqual(phrases[("бобёр", "сломал")], 1)
        self.assertEqual(phrases[("бобёр", "сломал", "сборку")], 1)


class TestTheLearnPathDoesNotWriteTheIndex(PhraseIndexTestCase):
    """Единственная точка записи — фоновая пересборка (design Р2).

    Проверяется наблюдаемым результатом, а не подсчётом вызовов: после
    обучения индекс обязан остаться пустым, хотя цепь уже пополнилась.
    Считать вызовы значило бы закрепить устройство; здесь закрепляется
    свойство.
    """

    async def test_learning_a_message_leaves_the_index_empty(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from app.core.markov import MarkovGenerator
        from app.services.learning_service import LearningService

        generator = MagicMock(spec=MarkovGenerator)
        generator.invalidate_chat_cache = MagicMock()
        generator.generate_text = AsyncMock(return_value="")
        service = LearningService(self.db, generator)

        await service.record_message(
            CHAT, "бобёр опять сломал сборку", ["бобёр", "опять", "сломал", "сборку"]
        )

        # Цепь пополнилась — значит обучение действительно прошло...
        async with self.db._lock:
            conn = await self.db._get_conn()
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM transitions WHERE chat_id = ?", (CHAT,)
            )
            self.assertGreater((await cursor.fetchone())[0], 0)
        # ...а индекс остался пустым: его пишет только пересборка.
        self.assertEqual(await self._stored(CHAT), {})


class TestClearWipesTheIndex(PhraseIndexTestCase):
    async def test_clear_chat_removes_the_phrases_of_that_chat(self) -> None:
        """Пережившая очистку фраза — это удержание удалённых данных."""
        await self._chain(CHAT, [("бобёр", "сломал", "сборку", 1)])
        await self._chain(OTHER_CHAT, [("другой", "чат", "тут", 1)])
        await self.repo.rebuild_chat(CHAT)
        await self.repo.rebuild_chat(OTHER_CHAT)

        await self.db.clear_chat(CHAT)

        self.assertEqual(await self._stored(CHAT), {})
        self.assertNotEqual(await self._stored(OTHER_CHAT), {})


if __name__ == "__main__":
    unittest.main()
