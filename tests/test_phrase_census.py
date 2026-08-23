"""Замер фраз в цепи не должен разойтись с извлечением, которым его считают.

Весь смысл `tools/eval/phrase_census.py` — в том, что число, попадающее в
роадмап, снято тем же фильтром, каким фразовый индекс будет отбирать n-граммы.
Если фильтры разойдутся, замер станет числом ни о чём, и заметить это будет
нечем: обе стороны по отдельности выглядят правдоподобно.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app.core.hot_ngrams import extract_content_ngrams, is_content_ngram
from tools.eval.phrase_census import _counts, main

# Синтетические ID: инвариант tests/test_no_real_chat_ids.
CHAT = 4321


class TestFilterMatchesExtraction(unittest.TestCase):
    """`is_content_ngram` — тот же предикат, что внутри `extract_content_ngrams`."""

    def test_the_two_filters_agree_on_every_ngram_of_a_message(self) -> None:
        tokens = [
            "пепе",
            "опять",
            "сломал",
            "сборку",
            ".",
            "он",
            "же",
            "не",
            "я",
            "тестовый",
            "прогон",
        ]
        extracted = set(extract_content_ngrams(tokens))

        for i in range(len(tokens) - 1):
            for size in (2, 3):
                if i + size > len(tokens):
                    continue
                ngram = tuple(tokens[i : i + size])
                with self.subTest(ngram=ngram):
                    self.assertEqual(is_content_ngram(ngram), ngram in extracted)

    def test_punctuation_disqualifies_the_whole_ngram(self) -> None:
        self.assertFalse(is_content_ngram(("сломал", ".")))
        self.assertFalse(is_content_ngram(("сломал", ".", "сборку")))

    def test_an_ngram_of_stopwords_alone_does_not_qualify(self) -> None:
        self.assertFalse(is_content_ngram(("он", "же")))
        self.assertTrue(is_content_ngram(("он", "сломал")))


class TestCensusReadsTheChain(unittest.TestCase):
    """Счётчик биграммы — сумма группы строк по w3, как у `get_hot`."""

    def _database(self, rows: list[tuple[str, str, str, int]]) -> Path:
        path = Path(tempfile.mkdtemp()) / "markov.db"
        with closing(sqlite3.connect(path)) as conn:
            conn.execute(
                "CREATE TABLE transitions (chat_id INTEGER, w1 TEXT, w2 TEXT,"
                " w3 TEXT, cnt INTEGER)"
            )
            conn.executemany(
                "INSERT INTO transitions VALUES (?, ?, ?, ?, ?)",
                [(CHAT, *row) for row in rows],
            )
            # closing() закрывает, но не коммитит — в отличие от собственного
            # контекст-менеджера sqlite3, который коммитит, но не закрывает.
            conn.commit()
        self.addCleanup(path.unlink)
        return path

    def test_bigram_count_sums_the_row_group(self) -> None:
        path = self._database(
            [
                ("пепе", "сломал", "сборку", 2),
                ("пепе", "сломал", "прогон", 3),
                ("пепе", "сломал", ".", 1),
            ]
        )

        trigrams, bigrams = _counts(path, CHAT)

        # Триграмма с пунктуацией отсеяна, но её вхождения — часть биграммы:
        # пара «пепе сломал» встретилась шесть раз, чем бы она ни кончалась.
        self.assertEqual(
            dict(trigrams),
            {("пепе", "сломал", "сборку"): 2, ("пепе", "сломал", "прогон"): 3},
        )
        self.assertEqual(dict(bigrams), {("пепе", "сломал"): 6})

    def test_the_largest_chat_is_picked_when_none_is_named(self) -> None:
        path = Path(tempfile.mkdtemp()) / "markov.db"
        with closing(sqlite3.connect(path)) as conn:
            conn.execute(
                "CREATE TABLE transitions (chat_id INTEGER, w1 TEXT, w2 TEXT,"
                " w3 TEXT, cnt INTEGER)"
            )
            conn.executemany(
                "INSERT INTO transitions VALUES (?, ?, ?, ?, ?)",
                [
                    (CHAT, "пепе", "сломал", "сборку", 1),
                    (CHAT + 1, "другой", "чат", "тут", 50),
                ],
            )
            conn.commit()
        self.addCleanup(path.unlink)

        _, bigrams = _counts(path, None)

        self.assertEqual(dict(bigrams), {("другой", "чат"): 50})

    def test_missing_database_is_a_message_not_a_traceback(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--db", "no/such/markov.db"])


if __name__ == "__main__":
    unittest.main()
