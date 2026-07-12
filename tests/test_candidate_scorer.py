from __future__ import annotations

import random
import unittest

from app.core.candidate_scorer import (
    CONTEXT_RELEVANCE_CAP,
    LENGTH_MODES,
    build_recent_reply_trigrams,
    build_token_idf,
    completion_quality,
    context_relevance,
    idf_context_relevance,
    natural_length,
    recent_reply_overlap,
    repetition_penalty,
    sample_length_mode,
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

    def test_stem_token_folds_inflected_forms(self) -> None:
        from app.core.candidate_scorer import stem_token

        # Live finding: "гнойному пидору" scored context_rel=0 against
        # "гнойный пидор" because exact-token overlap sees different forms.
        self.assertEqual(stem_token("гнойному"), stem_token("гнойный"))
        self.assertEqual(stem_token("пидору"), stem_token("пидор"))
        self.assertEqual(stem_token("пидора"), stem_token("пидор"))
        self.assertEqual(stem_token("славу"), stem_token("слава"))
        self.assertEqual(stem_token("стасу"), stem_token("стас"))
        self.assertEqual(stem_token("хоссейна"), stem_token("хоссейн"))

    def test_stem_token_leaves_short_and_latin_untouched(self) -> None:
        from app.core.candidate_scorer import stem_token

        self.assertEqual(stem_token("кто"), "кто")
        self.assertEqual(stem_token("дом"), "дом")
        self.assertEqual(stem_token("fy"), "fy")
        self.assertEqual(stem_token("javascript"), "javascript")

    def test_idf_relevance_sees_through_inflection(self) -> None:
        from app.core.candidate_scorer import build_token_idf, idf_context_relevance

        idf = build_token_idf(
            [tokenize("гнойный пидор слава"), tokenize("обед готовила мама")]
        )
        context = tokenize("кто гнойный пидор")

        inflected = idf_context_relevance(
            tokenize("славу назвали гнойным пидором"), context, idf
        )
        unrelated = idf_context_relevance(
            tokenize("мама готовила обед"), context, idf
        )

        self.assertGreater(inflected, 0.0)
        self.assertEqual(unrelated, 0.0)

    def test_echo_guard_folds_inflection_too(self) -> None:
        from app.core.candidate_scorer import build_token_idf, idf_context_relevance

        idf = build_token_idf([tokenize("старая копия бд лежит на диске")])
        # "копию бд" is the same parrot as "копия бд", just declined.
        echoed = idf_context_relevance(
            tokenize("копию бд"), tokenize("старая копия бд"), idf
        )

        self.assertEqual(echoed, 0.0)

    def test_verbatim_quote_severity_tolerates_extended_copies(self) -> None:
        from app.core.candidate_scorer import verbatim_quote_severity

        self.assertEqual(verbatim_quote_severity(0.0), 0.0)
        self.assertEqual(verbatim_quote_severity(0.5), 0.0)  # quote + tail
        self.assertEqual(verbatim_quote_severity(0.6), 0.0)
        self.assertAlmostEqual(verbatim_quote_severity(0.8), 0.5)
        self.assertAlmostEqual(verbatim_quote_severity(1.0), 1.0)  # pure quote

    def test_pure_echo_scores_zero_relevance(self) -> None:
        # A candidate whose informative tokens all come from the context is a
        # parrot: it must not collect the context bonus, on either formula.
        idf = build_token_idf([tokenize("старая копия бд лежит на диске")])
        context = tokenize("старая копия бд")

        echo_with_idf = idf_context_relevance(tokenize("копия бд"), context, idf)
        echo_no_idf = idf_context_relevance(tokenize("копия бд"), context, {})
        novel = idf_context_relevance(
            tokenize("копия бд лежит на диске"), context, idf
        )

        self.assertEqual(echo_with_idf, 0.0)
        self.assertEqual(echo_no_idf, 0.0)
        self.assertGreater(novel, 0.0)

    def test_short_candidate_uses_softer_repeat_slope(self) -> None:
        # The short weight (1.00) plus the flat 0.20 offset must stay gentler
        # than the long weight (1.60) for the same repeated-token ratio.
        short = repetition_penalty(tokenize("да да"))
        long = repetition_penalty(tokenize("да да да да"))

        self.assertGreater(short, 0.0)
        self.assertLess(short, long)

    def test_natural_length_prefers_band_without_preferring_longest(self) -> None:
        short = natural_length(tokenize("один два"))
        natural = natural_length(tokenize("один два три четыре пять шесть"))
        long = natural_length(tokenize(" ".join(f"слово{i}" for i in range(30))))

        self.assertGreater(natural, short)
        self.assertGreater(natural, long)

    def test_natural_length_peak_follows_mode(self) -> None:
        short_text = tokenize("один два")
        medium_text = tokenize("один два три четыре пять шесть семь")
        long_text = tokenize(" ".join(f"слово{i}" for i in range(18)))

        self.assertEqual(natural_length(short_text, "short"), 1.0)
        self.assertLess(natural_length(medium_text, "short"), 1.0)
        self.assertEqual(natural_length(medium_text, "medium"), 1.0)
        self.assertLess(natural_length(short_text, "long"), 1.0)
        self.assertEqual(natural_length(long_text, "long"), 1.0)
        self.assertLess(natural_length(long_text, "medium"), 1.0)

    def test_natural_length_default_mode_is_medium(self) -> None:
        tokens = tokenize("один два три четыре пять шесть")

        self.assertEqual(natural_length(tokens), natural_length(tokens, "medium"))

    def test_sample_length_mode_respects_degenerate_weights(self) -> None:
        rng = random.Random(7)
        for index, mode in enumerate(LENGTH_MODES):
            weights = tuple(
                1.0 if position == index else 0.0 for position in range(3)
            )
            picked = {sample_length_mode(weights, rng) for _ in range(20)}
            self.assertEqual(picked, {mode})

    def test_sample_length_mode_covers_all_modes(self) -> None:
        rng = random.Random(11)
        picked = {
            sample_length_mode((0.25, 0.55, 0.2), rng) for _ in range(300)
        }
        self.assertEqual(picked, set(LENGTH_MODES))

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
            + first.natural_length
            + first.context_relevance
            - first.repetition_penalty
            - first.recent_penalty,
        )
        self.assertEqual(first.recent_penalty, 0.0)

    def test_recent_reply_overlap_measures_shared_trigrams(self) -> None:
        recent = build_recent_reply_trigrams(["один два три четыре пять"])

        full = recent_reply_overlap(
            tokenize("один два три четыре пять"), recent
        )
        partial = recent_reply_overlap(
            tokenize("один два три совсем другое"), recent
        )
        fresh = recent_reply_overlap(
            tokenize("совсем новый текст ответа"), recent
        )

        self.assertEqual(full, 1.0)
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 1.0)
        self.assertEqual(fresh, 0.0)

    def test_recent_reply_overlap_is_case_and_punctuation_insensitive(self) -> None:
        recent = build_recent_reply_trigrams(["Один Два Три!"])

        self.assertEqual(
            recent_reply_overlap(tokenize("один два три"), recent), 1.0
        )

    def test_recent_reply_overlap_ignores_short_candidates(self) -> None:
        recent = build_recent_reply_trigrams(["один два три четыре"])

        self.assertEqual(recent_reply_overlap(tokenize("один два"), recent), 0.0)
        self.assertEqual(recent_reply_overlap(tokenize("один два три"), set()), 0.0)
