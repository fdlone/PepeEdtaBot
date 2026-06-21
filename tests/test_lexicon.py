from __future__ import annotations

import unittest

from app.core.lexicon import BAD_ENDING_WORDS, STOPWORDS


class TestLexicon(unittest.TestCase):
    def test_shared_word_sets_have_expected_members(self) -> None:
        self.assertTrue(STOPWORDS)
        self.assertTrue(BAD_ENDING_WORDS)
        self.assertIn("и", BAD_ENDING_WORDS)
        self.assertIn("the", BAD_ENDING_WORDS)


if __name__ == "__main__":
    unittest.main()
