from __future__ import annotations

import unittest

from app.core.candidate_scorer import (
    CONTEXT_RELEVANCE_CAP,
    completion_quality,
    context_relevance,
    lexical_diversity,
    natural_length,
    repetition_penalty,
    score_candidate,
)
from app.core.markov import tokenize


class TestCandidateScorer(unittest.TestCase):
    def test_completion_rewards_clean_balanced_ending(self) -> None:
        clean = completion_quality(
            'Это работает ("точно").',
            tokenize('Это работает ("точно").'),
        )
        unfinished = completion_quality("Это работает и", tokenize("Это работает и"))

        self.assertGreater(clean, unfinished)
        self.assertLess(unfinished, 0.0)

    def test_unbalanced_delimiter_is_penalized(self) -> None:
        balanced = completion_quality("(готово)", tokenize("(готово)"))
        unbalanced = completion_quality("(готово", tokenize("(готово"))

        self.assertGreater(balanced, unbalanced)

    def test_context_relevance_excludes_stopwords_and_is_capped(self) -> None:
        stopword_only = context_relevance(
            tokenize("и это в"),
            tokenize("и это в"),
        )
        meaningful = context_relevance(
            tokenize("синий двигатель мост"),
            tokenize("синий двигатель мост"),
        )

        self.assertEqual(stopword_only, 0.0)
        self.assertEqual(meaningful, CONTEXT_RELEVANCE_CAP)

    def test_short_candidate_uses_separate_diversity_threshold(self) -> None:
        short = lexical_diversity(tokenize("да да"))
        long = lexical_diversity(tokenize("да да да да"))

        self.assertGreater(short, 0.0)
        self.assertGreater(short, long)

    def test_natural_length_prefers_band_without_preferring_longest(self) -> None:
        short = natural_length(tokenize("один два"))
        natural = natural_length(tokenize("один два три четыре пять шесть"))
        long = natural_length(tokenize(" ".join(f"слово{i}" for i in range(30))))

        self.assertGreater(natural, short)
        self.assertGreater(natural, long)

    def test_repetition_penalty_counts_tokens_bigrams_and_trigrams(self) -> None:
        clean = repetition_penalty(tokenize("один два три четыре пять"))
        repeated = repetition_penalty(tokenize("эхо эхо эхо эхо эхо"))

        self.assertEqual(clean, 0.0)
        self.assertGreater(repeated, clean)

    def test_total_score_is_additive_and_deterministic(self) -> None:
        tokens = tokenize("синий двигатель работает хорошо.")
        context = tokenize("проверяем синий двигатель")

        first = score_candidate("синий двигатель работает хорошо.", tokens, context)
        second = score_candidate("синий двигатель работает хорошо.", tokens, context)

        self.assertEqual(first, second)
        self.assertAlmostEqual(
            first.total,
            first.completion_quality
            + first.lexical_diversity
            + first.natural_length
            + first.context_relevance
            - first.repetition_penalty,
        )
