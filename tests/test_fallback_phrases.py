from __future__ import annotations

import random
import unittest
from datetime import datetime

from app.config.runtime_state import RuntimeState
from app.presentation.fallback_phrases import (
    GENERATION_FAILED_PHRASES,
    HEATED_FALLBACK_PHRASES,
    LATE_NIGHT_FALLBACK_PHRASES,
    NOT_ENOUGH_DATA_PHRASES,
    is_late_night,
    late_night_pool,
    mood_fallback_pool,
    pick_fallback_phrase,
)


def _make_runtime_state() -> RuntimeState:
    return RuntimeState(
        reply_probability=0.08,
        min_cooldown_sec=45,
        min_tokens_for_model=200,
        max_reply_chars=280,
        max_reply_tokens=45,
        normalize_lower=False,
        auto_capitalize_replies=False,
        typing_min_ms=350,
        typing_max_ms=1100,
        typing_per_char_ms=12,
        randomness_strength=2.0,
        candidate_selection_temperature=0.7,
        reply_flavor_strength=1.0,
        emoji_append_chance=0.15,
        repetition_penalty_strength=1.0,
        recent_reply_penalty_strength=1.0,
        verbatim_penalty_strength=1.0,
        length_mode_weights=(0.25, 0.55, 0.2),
        intonation_profile_strength=0.0,
        length_context_adaptation=0.0,
        markov_order=3,
        enable_backoff=True,
        markov_jump_probability=0.04,
        context_jump_boost=1.0,
        verbatim_extension_share=0.0,
        order_mix_probability=0.0,
        slot_mutation_probability=0.0,
        hot_ngram_seed_chance=0.05,
        hot_ngram_min_count=3,
        hot_ngram_recency_share=0.5,
        rare_event_chance=0.005,
        false_start_chance=0.03,
        rare_event_daily_cap=3,
        user_quirk_chance=0.1,
        user_quirk_min_interactions=25,
        use_reply_context=True,
        fuzzy_context_casefold=True,
        reply_context_max_tokens=12,
        reply_context_bias=1.8,
        reply_context_start_bias=2.2,
            context_start_affinity=3.0,
        context_anchor_splice_probability=0.0,
        reply_context_only_for_replies=True,
        reply_context_include_current_message=True,
        pivo_recent_pool_window=5,
        pivo_temporal_flavor_chance=0.5,
        mood_enabled=True,
        mood_modulation_strength=1.0,
        mood_ewma_alpha=0.3,
        mood_lively_rate_per_min=12.0,
        mood_sleepy_rate_per_min=2.0,
        mood_heated_intensity=0.4,
        mood_mention_heated_share=0.0,
        mood_max_rate_per_min=120.0,
        reply_director_enabled=True,
        reply_probability_min=0.02,
        reply_probability_max=0.30,
        reply_burst_boost_sec=180,
        reply_burst_boost_mult=2.0,
        reply_burst_suppress_sec=600,
        reply_burst_suppress_mult=0.5,
        reply_max_per_hour=20,
        mention_cooldown_sec=5,
        runtime_state_ttl_sec=10,
        runtime_state_max_chats=8,
    )


class TestFallbackPools(unittest.TestCase):
    def test_pools_are_non_trivial_and_unique(self) -> None:
        for pool in (NOT_ENOUGH_DATA_PHRASES, GENERATION_FAILED_PHRASES):
            self.assertGreaterEqual(len(pool), 10)
            self.assertEqual(len(pool), len(set(pool)))
            self.assertTrue(all(phrase.strip() for phrase in pool))


class TestUserQuirkVocatives(unittest.TestCase):
    def test_pool_is_non_trivial_unique_and_placeholder_free(self) -> None:
        from app.presentation.fallback_phrases import USER_QUIRK_VOCATIVES

        self.assertGreaterEqual(len(USER_QUIRK_VOCATIVES), 8)
        self.assertEqual(len(USER_QUIRK_VOCATIVES), len(set(USER_QUIRK_VOCATIVES)))
        for phrase in USER_QUIRK_VOCATIVES:
            self.assertTrue(phrase.strip())
            # Privacy by construction: nothing about the user is interpolated.
            self.assertNotIn("{", phrase)
            self.assertNotIn("%", phrase)

    def test_next_quirk_vocative_picks_from_pool_deterministically(self) -> None:
        from app.presentation.fallback_phrases import (
            USER_QUIRK_VOCATIVES,
            next_quirk_vocative,
        )

        picks = {next_quirk_vocative(random.Random(seed)) for seed in range(30)}
        self.assertTrue(picks.issubset(set(USER_QUIRK_VOCATIVES)))
        self.assertGreater(len(picks), 1)
        self.assertEqual(
            next_quirk_vocative(random.Random(7)),
            next_quirk_vocative(random.Random(7)),
        )


class TestPickFallbackPhrase(unittest.TestCase):
    def test_picks_from_pool(self) -> None:
        phrase = pick_fallback_phrase(
            NOT_ENOUGH_DATA_PHRASES, rng=random.Random(1)
        )
        self.assertIn(phrase, NOT_ENOUGH_DATA_PHRASES)

    def test_avoids_recent_phrases(self) -> None:
        pool = ("a", "b", "c")
        rng = random.Random(7)
        for _ in range(50):
            self.assertEqual(pick_fallback_phrase(pool, ["a", "c"], rng=rng), "b")

    def test_falls_back_to_full_pool_when_all_recent(self) -> None:
        pool = ("a", "b")
        phrase = pick_fallback_phrase(pool, ["a", "b"], rng=random.Random(3))
        self.assertIn(phrase, pool)

    def test_deterministic_with_seeded_rng(self) -> None:
        first = pick_fallback_phrase(GENERATION_FAILED_PHRASES, rng=random.Random(42))
        second = pick_fallback_phrase(GENERATION_FAILED_PHRASES, rng=random.Random(42))
        self.assertEqual(first, second)


class TestNextFallbackPhrase(unittest.TestCase):
    def test_remembers_recent_per_chat_and_avoids_repeats(self) -> None:
        from app.handlers.learning import RECENT_FALLBACK_LIMIT, next_fallback_phrase

        state = _make_runtime_state()
        rng = random.Random(5)
        picked = [
            next_fallback_phrase(state, 100, NOT_ENOUGH_DATA_PHRASES, rng=rng)
            for _ in range(RECENT_FALLBACK_LIMIT + 1)
        ]
        # Consecutive picks within the memory window never repeat.
        for index in range(1, len(picked)):
            window = picked[max(0, index - RECENT_FALLBACK_LIMIT) : index]
            self.assertNotIn(picked[index], window)
        recent = state.recent_fallbacks[100]
        self.assertEqual(recent.maxlen, RECENT_FALLBACK_LIMIT)
        self.assertEqual(list(recent), picked[-RECENT_FALLBACK_LIMIT:])

    def test_forget_chat_drops_fallback_memory(self) -> None:
        from app.handlers.learning import next_fallback_phrase

        state = _make_runtime_state()
        next_fallback_phrase(state, 100, GENERATION_FAILED_PHRASES, rng=random.Random(1))
        self.assertIn(100, state.recent_fallbacks)
        state.forget_chat(100)
        self.assertNotIn(100, state.recent_fallbacks)


class TestMoodFallbackPool(unittest.TestCase):
    """M1: heated mood adds punchier fallbacks; other moods stay neutral."""

    def test_heated_extends_pool(self) -> None:
        self.assertEqual(
            mood_fallback_pool(NOT_ENOUGH_DATA_PHRASES, "heated"),
            NOT_ENOUGH_DATA_PHRASES + HEATED_FALLBACK_PHRASES,
        )

    def test_other_moods_keep_base_pool(self) -> None:
        for mood in ("calm", "lively", "sleepy", None):
            self.assertEqual(
                mood_fallback_pool(GENERATION_FAILED_PHRASES, mood),
                GENERATION_FAILED_PHRASES,
            )


class TestLateNightFallback(unittest.TestCase):
    """S4: late-night phrases join the pool only in the small hours."""

    def test_is_late_night_bucket(self) -> None:
        self.assertTrue(is_late_night(datetime(2026, 7, 1, 0, 0)))
        self.assertTrue(is_late_night(datetime(2026, 7, 1, 5, 59)))
        self.assertFalse(is_late_night(datetime(2026, 7, 1, 6, 0)))
        self.assertFalse(is_late_night(datetime(2026, 7, 1, 15, 0)))

    def test_pool_extended_only_late_at_night(self) -> None:
        base = NOT_ENOUGH_DATA_PHRASES
        day = late_night_pool(base, datetime(2026, 7, 1, 15, 0))
        night = late_night_pool(base, datetime(2026, 7, 1, 2, 0))
        self.assertEqual(day, base)
        self.assertEqual(night, base + LATE_NIGHT_FALLBACK_PHRASES)

    def test_none_now_keeps_base_pool(self) -> None:
        self.assertEqual(
            late_night_pool(NOT_ENOUGH_DATA_PHRASES, None), NOT_ENOUGH_DATA_PHRASES
        )

    def test_next_fallback_can_return_late_night_phrase(self) -> None:
        from app.handlers.learning import next_fallback_phrase

        state = _make_runtime_state()
        night = datetime(2026, 7, 1, 2, 0)
        seen = set()
        for seed in range(60):
            state.recent_fallbacks.pop(200, None)
            seen.add(
                next_fallback_phrase(
                    state, 200, NOT_ENOUGH_DATA_PHRASES, rng=random.Random(seed), now=night
                )
            )
        self.assertTrue(seen & set(LATE_NIGHT_FALLBACK_PHRASES))


if __name__ == "__main__":
    unittest.main()
