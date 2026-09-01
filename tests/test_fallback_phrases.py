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
    """Общая фикстура состояния плюс два отклонения, нужные этому файлу.

    Прежде здесь стоял собственный список из без малого сотни полей — второе
    рукописное зеркало ``RUNTIME_FIELDS`` рядом с первым (O6). Оно отличалось
    от общей фикстуры ровно двумя значениями, что и осталось ниже; всё
    остальное совпадало дословно.
    """
    from tests.test_runtime_state import make_runtime_state

    return make_runtime_state(fuzzy_context_casefold=True, runtime_state_max_chats=8)


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


class TestNameQuirkVocative(unittest.TestCase):
    def test_name_vocative_is_bare_name_or_name_plus_pool_phrase(self) -> None:
        from app.presentation.fallback_phrases import (
            USER_QUIRK_VOCATIVES,
            next_quirk_vocative,
        )

        seen_bare = seen_fused = False
        for seed in range(60):
            vocative = next_quirk_vocative(
                random.Random(seed), first_name="саня"
            )
            if vocative == "саня":
                seen_bare = True
            else:
                self.assertTrue(vocative.startswith("саня, "))
                self.assertIn(
                    vocative.removeprefix("саня, "), USER_QUIRK_VOCATIVES
                )
                seen_fused = True
        self.assertTrue(seen_bare)
        self.assertTrue(seen_fused)

    def test_none_name_keeps_legacy_pool_behaviour(self) -> None:
        from app.presentation.fallback_phrases import (
            USER_QUIRK_VOCATIVES,
            next_quirk_vocative,
        )

        picks = {
            next_quirk_vocative(random.Random(seed), first_name=None)
            for seed in range(30)
        }
        self.assertTrue(picks.issubset(set(USER_QUIRK_VOCATIVES)))


class TestSanitizeFirstName(unittest.TestCase):
    def test_plain_names_lowercased(self) -> None:
        from app.core.text import sanitize_first_name

        self.assertEqual(sanitize_first_name("Саня"), "саня")
        self.assertEqual(sanitize_first_name("Anna-Maria"), "anna-maria")

    def test_takes_first_word_only(self) -> None:
        from app.core.text import sanitize_first_name

        self.assertEqual(sanitize_first_name("Иван Иваныч"), "иван")

    def test_strips_emoji_and_decorations(self) -> None:
        from app.core.text import sanitize_first_name

        self.assertEqual(sanitize_first_name("🔥Саня🔥"), "саня")
        self.assertEqual(sanitize_first_name("~*_Slava_*~"), "slava")

    def test_unusable_names_return_none(self) -> None:
        from app.core.text import sanitize_first_name

        for raw in ("", "   ", "🔥🔥🔥", "Я", "-", "12345", "х" * 25):
            self.assertIsNone(sanitize_first_name(raw), raw)


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
        from app.services.reply_pipeline import RECENT_FALLBACK_LIMIT, next_fallback_phrase

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
        from app.services.reply_pipeline import next_fallback_phrase

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

    def test_is_late_night_reads_the_wall_clock_of_its_zone(self) -> None:
        # O12: один и тот же момент — ночь по UTC, утро по MSK. Функция обязана
        # смотреть на настенные часы переданной зоны, а не на UTC.
        from datetime import UTC
        from zoneinfo import ZoneInfo

        instant = datetime(2026, 1, 1, 4, 0, tzinfo=UTC)  # 07:00 в MSK
        self.assertTrue(is_late_night(instant))
        self.assertFalse(
            is_late_night(instant.astimezone(ZoneInfo("Europe/Moscow")))
        )

    def test_none_now_keeps_base_pool(self) -> None:
        self.assertEqual(
            late_night_pool(NOT_ENOUGH_DATA_PHRASES, None), NOT_ENOUGH_DATA_PHRASES
        )

    def test_next_fallback_can_return_late_night_phrase(self) -> None:
        from app.services.reply_pipeline import next_fallback_phrase

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
