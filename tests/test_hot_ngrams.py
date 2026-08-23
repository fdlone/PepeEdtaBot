from __future__ import annotations

import unittest

from app.core.hot_ngrams import (
    MIN_CONTENT_TOKEN_LEN,
    extract_content_ngrams,
    is_content_ngram,
)


class ExtractContentNgramsTest(unittest.TestCase):
    def test_extracts_bigrams_and_trigrams(self) -> None:
        ngrams = extract_content_ngrams(["крутой", "бобёр", "пришёл"])
        self.assertIn(("крутой", "бобёр"), ngrams)
        self.assertIn(("бобёр", "пришёл"), ngrams)
        self.assertIn(("крутой", "бобёр", "пришёл"), ngrams)

    def test_skips_stopword_only_ngrams(self) -> None:
        # "а", "он", "не" — no content token (stopword or shorter than 3 chars)
        self.assertEqual(extract_content_ngrams(["а", "он", "не"]), [])

    def test_keeps_ngram_with_one_content_token(self) -> None:
        ngrams = extract_content_ngrams(["не", "бобёр"])
        self.assertEqual(ngrams, [("не", "бобёр")])

    def test_skips_punctuation_tokens(self) -> None:
        # tokenize() may emit punctuation tokens; n-grams containing them are noise
        ngrams = extract_content_ngrams(["бобёр", "?", "пришёл"])
        self.assertNotIn(("бобёр", "?"), ngrams)
        self.assertNotIn(("бобёр", "?", "пришёл"), ngrams)

    def test_dedup_and_cap(self) -> None:
        tokens = ["бобёр", "пришёл"] * 40
        ngrams = extract_content_ngrams(tokens, max_per_message=5)
        self.assertEqual(len(ngrams), len(set(ngrams)))
        self.assertLessEqual(len(ngrams), 5)

    def test_capitalized_stopwords_are_not_content(self) -> None:
        # case-preserved profile (normalize_lower=false): stopword check
        # must casefold, otherwise "Что это" counts as a content bigram
        self.assertEqual(extract_content_ngrams(["Что", "это"]), [])

    def test_cap_keeps_a_mix_of_sizes(self) -> None:
        tokens = [f"слово{i}" for i in range(30)]
        ngrams = extract_content_ngrams(tokens, max_per_message=10)
        sizes = {len(ngram) for ngram in ngrams}
        self.assertEqual(sizes, {2, 3})

    def test_short_input_returns_empty(self) -> None:
        self.assertEqual(extract_content_ngrams(["бобёр"]), [])
        self.assertEqual(extract_content_ngrams([]), [])



class IsContentNgramTest(unittest.TestCase):
    """Предикат стал публичным: по нему фразовый индекс отбирает из цепи.

    Раньше он жил внутри цикла извлечения, и добраться до него было нечем —
    поэтому границы проверялись только через готовый список n-грамм. Проверка
    в точке границы, а не в середине диапазона: середина проходит и у неверно
    сдвинутого порога.
    """

    def test_length_boundary_is_checked_in_the_point(self) -> None:
        short = "б" * (MIN_CONTENT_TOKEN_LEN - 1)
        exact = "б" * MIN_CONTENT_TOKEN_LEN

        self.assertFalse(is_content_ngram((short, short)))
        self.assertTrue(is_content_ngram((short, exact)))

    def test_punctuation_disqualifies_the_whole_ngram(self) -> None:
        self.assertFalse(is_content_ngram(("бобёр", ".")))
        self.assertFalse(is_content_ngram(("бобёр", ".", "пришёл")))

    def test_stopwords_alone_do_not_qualify(self) -> None:
        self.assertFalse(is_content_ngram(("он", "же")))
        self.assertTrue(is_content_ngram(("он", "пришёл")))

    def test_the_predicate_decides_what_extraction_keeps(self) -> None:
        """Извлечение и предикат — одно правило, а не два похожих."""
        tokens = ["бобёр", "опять", "сломал", "сборку", ".", "он", "же", "не", "я"]
        extracted = set(extract_content_ngrams(tokens))

        for i in range(len(tokens) - 1):
            for size in (2, 3):
                if i + size > len(tokens):
                    continue
                ngram = tuple(tokens[i : i + size])
                with self.subTest(ngram=ngram):
                    self.assertEqual(is_content_ngram(ngram), ngram in extracted)


if __name__ == "__main__":
    unittest.main()
