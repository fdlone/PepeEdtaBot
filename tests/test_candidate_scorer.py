from __future__ import annotations

import random
import unittest

from app.core.candidate_scorer import (
    CONTEXT_LENGTH_LONG_TOKENS,
    CONTEXT_LENGTH_SHORT_TOKENS,
    CONTEXT_RELEVANCE_CAP,
    LENGTH_MODES,
    build_recent_reply_trigrams,
    build_token_idf,
    completion_quality,
    context_length_weights,
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

    def test_recognized_unit_window_count(self) -> None:
        """Единица — самая длинная серия корпусных окон в пределах длины.

        Серия из k окон покрывает k + 3 контентных токена (окно = 4), предел
        единицы — 6 токенов, поэтому единицей признаются серии до трёх окон.
        """
        from app.core.candidate_scorer import recognized_unit_windows

        self.assertEqual(recognized_unit_windows([], 4), 0)
        self.assertEqual(recognized_unit_windows([False, False], 4), 0)
        self.assertEqual(recognized_unit_windows([True], 4), 1)  # 4 токена
        self.assertEqual(recognized_unit_windows([True, True, True], 4), 3)  # 6
        # Четыре подряд — 7 токенов: это уже цитата, а не единица.
        self.assertEqual(recognized_unit_windows([True] * 4, 4), 0)
        # Берётся самая длинная серия, а не сумма разрозненных.
        self.assertEqual(
            recognized_unit_windows([True, False, True, True], 4), 2
        )

    def test_one_borrowed_unit_leaves_the_quote_share(self) -> None:
        # Кандидат: один корпусный фрагмент плюс своё продолжение. Без гарда
        # доля 2/5, с гардом — 0/3: заимствование выведено из вопроса целиком.
        from app.core.candidate_scorer import verbatim_ngram_overlap

        tokens = tokenize("наш кот опять уронил ёлку а потом ушёл спать")
        content = [t.casefold() for t in tokenize("наш кот опять уронил ёлку")]
        corpus = {tuple(content[0:4]), tuple(content[1:5])}

        raw = verbatim_ngram_overlap(tokens, corpus)
        guarded = verbatim_ngram_overlap(
            tokens, corpus, exempt_recognized_unit=True
        )

        self.assertGreater(raw, 0.0)
        self.assertEqual(guarded, 0.0)

    def test_second_borrowing_still_counts(self) -> None:
        # Две отдельные единицы: исключается одна, вторая платит.
        from app.core.candidate_scorer import verbatim_ngram_overlap

        tokens = tokenize(
            "наш кот опять уронил ёлку и собака съела весь новогодний оливье"
        )
        content = [t.casefold() for t in tokens]
        first = tuple(content[0:4])
        second = tuple(content[6:10])
        both = verbatim_ngram_overlap(
            tokens, {first, second}, exempt_recognized_unit=True
        )
        one_only = verbatim_ngram_overlap(
            tokens, {first}, exempt_recognized_unit=True
        )

        self.assertGreater(both, one_only)
        self.assertEqual(one_only, 0.0)

    def test_long_corpus_stretch_is_a_quote_not_a_unit(self) -> None:
        # Серия длиннее предела единицы не исключается вовсе: доля с гардом
        # совпадает с сырой.
        from app.core.candidate_scorer import verbatim_ngram_overlap

        tokens = tokenize("наш кот опять уронил ёлку на пол и убежал")
        content = [t.casefold() for t in tokens]
        corpus = {tuple(content[i : i + 4]) for i in range(5)}  # 5 окон подряд

        raw = verbatim_ngram_overlap(tokens, corpus)
        guarded = verbatim_ngram_overlap(
            tokens, corpus, exempt_recognized_unit=True
        )

        self.assertEqual(guarded, raw)
        self.assertGreater(raw, 0.0)

    def test_reply_that_is_nothing_but_the_unit_still_pays(self) -> None:
        # Роадмап: «штраф остаётся за ответ, целиком корпусный». Кроме единицы
        # в кандидате нет окон — исключение отменяется.
        from app.core.candidate_scorer import verbatim_ngram_overlap

        tokens = tokenize("наш кот уронил ёлку")
        content = [t.casefold() for t in tokens]
        corpus = {tuple(content[0:4])}

        self.assertEqual(verbatim_ngram_overlap(tokens, corpus), 1.0)
        self.assertEqual(
            verbatim_ngram_overlap(
                tokens, corpus, exempt_recognized_unit=True
            ),
            1.0,
        )

    def test_guard_off_matches_the_previous_number(self) -> None:
        from app.core.candidate_scorer import verbatim_ngram_overlap

        tokens = tokenize("наш кот опять уронил ёлку а потом ушёл спать")
        content = [t.casefold() for t in tokens]
        corpus = {tuple(content[0:4]), tuple(content[1:5])}

        self.assertEqual(
            verbatim_ngram_overlap(tokens, corpus, exempt_recognized_unit=False),
            verbatim_ngram_overlap(tokens, corpus),
        )

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

    def test_sample_length_mode_all_zero_falls_back_to_base_weights(self) -> None:
        # Mood clamping can zero every weight when the only positive base
        # weight meets a sub-1.0 multiplier at strength >= 2 (legal config);
        # random.choices raises on an all-zero vector.
        rng = random.Random(3)
        picked = {
            sample_length_mode((0.0, 0.0, 0.0), rng, (0.0, 0.0, 1.0))
            for _ in range(20)
        }
        self.assertEqual(picked, {"long"})

    def test_sample_length_mode_all_zero_without_base_is_uniform(self) -> None:
        rng = random.Random(3)
        picked = {
            sample_length_mode((0.0, 0.0, 0.0), rng) for _ in range(200)
        }
        self.assertEqual(picked, set(LENGTH_MODES))

    def test_context_length_weights_off_by_default(self) -> None:
        weights = (0.25, 0.55, 0.2)
        for incoming in (1, 8, 40):
            self.assertEqual(
                context_length_weights(weights, incoming, 0.0), weights
            )

    def test_context_length_weights_tilt_short_for_a_short_message(self) -> None:
        short, medium, long_ = context_length_weights((0.25, 0.55, 0.2), 2, 1.0)
        # At the short end the tilt is the full 1 + strength: short doubles,
        # long halves, medium is the pivot and does not move.
        self.assertAlmostEqual(short, 0.5)
        self.assertAlmostEqual(medium, 0.55)
        self.assertAlmostEqual(long_, 0.1)

    def test_context_length_weights_tilt_long_for_a_long_message(self) -> None:
        short, medium, long_ = context_length_weights((0.25, 0.55, 0.2), 30, 1.0)
        self.assertAlmostEqual(short, 0.125)
        self.assertAlmostEqual(medium, 0.55)
        self.assertAlmostEqual(long_, 0.4)

    def test_context_length_weights_are_neutral_midway(self) -> None:
        midpoint = (CONTEXT_LENGTH_SHORT_TOKENS + CONTEXT_LENGTH_LONG_TOKENS) // 2
        weights = context_length_weights((0.25, 0.55, 0.2), midpoint, 1.0)
        for actual, expected in zip(weights, (0.25, 0.55, 0.2), strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_context_length_weights_ramp_is_monotonic(self) -> None:
        long_weights = [
            context_length_weights((0.25, 0.55, 0.2), incoming, 1.0)[2]
            for incoming in range(1, 25)
        ]
        self.assertEqual(long_weights, sorted(long_weights))

    def test_context_length_weights_ignore_an_empty_message(self) -> None:
        weights = (0.25, 0.55, 0.2)
        self.assertEqual(context_length_weights(weights, 0, 1.0), weights)

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
