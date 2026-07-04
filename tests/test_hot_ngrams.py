from __future__ import annotations

import unittest

from app.core.hot_ngrams import extract_content_ngrams


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

    def test_short_input_returns_empty(self) -> None:
        self.assertEqual(extract_content_ngrams(["бобёр"]), [])
        self.assertEqual(extract_content_ngrams([]), [])


if __name__ == "__main__":
    unittest.main()
