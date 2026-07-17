"""Tests for LearningService — verbatim-copy detection via text cache."""
from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from app.core.markov import MarkovGenerator
from app.infrastructure.database import Database


def _make_generator() -> MarkovGenerator:
    from unittest.mock import AsyncMock, MagicMock
    gen = MagicMock(spec=MarkovGenerator)
    gen.invalidate_chat_cache = MagicMock()
    gen.generate_text = AsyncMock(return_value="")
    return gen


class TestLearningServiceDedup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from app.services.learning_service import LearningService

        self.db_path = Path(f"test_ls_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path), messages_retention_per_chat=3)
        await self.db.init()
        self.svc = LearningService(
            self.db,
            _make_generator(),
            text_cache_max_messages=3,
        )
        self.chat = 42

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def _record(self, text: str) -> None:
        from app.core.markov import tokenize
        tokens = tokenize(text)
        await self.svc.record_message(self.chat, text, tokens)

    # --- MessagesRepo.get_recent_normalized ---

    async def test_get_recent_normalized_returns_stored_texts(self) -> None:
        await self._record("кофе утром бодрит")
        await self._record("привет всем")
        result = await self.db.get_recent_normalized_messages(self.chat, 10)
        self.assertEqual(len(result), 2)

    async def test_get_recent_normalized_excludes_other_chats(self) -> None:
        await self._record("кофе утром бодрит")
        other = Database(str(self.db_path))
        await other.init()
        result = await other.get_recent_normalized_messages(999, 10)
        self.assertEqual(result, [])
        await other.close()

    async def test_get_recent_normalized_respects_limit_and_keeps_recent_order(self) -> None:
        await self._record("первое сообщение")
        await self._record("второе сообщение")
        await self._record("третье сообщение")
        await self._record("четвертое сообщение")
        result = await self.db.get_recent_normalized_messages(self.chat, 2)
        self.assertEqual(result, ["третье сообщение", "четвертое сообщение"])

    # --- text cache build and lookup ---

    async def test_text_cache_built_on_first_check(self) -> None:
        await self._record("пойдём пить кофе вечером")
        self.assertNotIn(self.chat, self.svc._text_cache)
        await self.svc.is_verbatim_copy(self.chat, "пойдём пить кофе вечером")
        self.assertIn(self.chat, self.svc._text_cache)

    async def test_cache_invalidated_after_record(self) -> None:
        await self._record("кофе утром бодрит")
        await self.svc.is_verbatim_copy(self.chat, "кофе утром бодрит")
        self.assertIn(self.chat, self.svc._text_cache)
        await self._record("новое сообщение")
        self.assertNotIn(self.chat, self.svc._text_cache)

    async def test_text_cache_uses_only_recent_window(self) -> None:
        await self._record("старое сообщение давно было")
        await self._record("новое одно")
        await self._record("новое два")
        await self._record("новое три")

        self.assertFalse(
            await self.svc.is_verbatim_copy(self.chat, "старое сообщение давно было")
        )
        self.assertTrue(
            await self.svc.is_verbatim_copy(self.chat, "новое одно")
        )

    async def test_copy_is_detected_after_retention_pruning(self) -> None:
        for text in ("old", "retained one", "retained two", "retained three"):
            await self._record(text)

        self.assertTrue(
            await self.svc.is_verbatim_copy(self.chat, "retained one")
        )
        self.assertFalse(await self.svc.is_verbatim_copy(self.chat, "old"))

    # --- is_verbatim_copy behaviour ---

    async def test_exact_match_detected(self) -> None:
        await self._record("кофе утром бодрит")
        result = await self.svc.is_verbatim_copy(self.chat, "кофе утром бодрит")
        self.assertTrue(result)

    async def test_case_insensitive_match(self) -> None:
        await self._record("Кофе Утром Бодрит")
        result = await self.svc.is_verbatim_copy(self.chat, "кофе утром бодрит")
        self.assertTrue(result)

    async def test_unique_text_not_detected(self) -> None:
        await self._record("пойдём пить кофе")
        result = await self.svc.is_verbatim_copy(self.chat, "совершенно другое предложение")
        self.assertFalse(result)

    async def test_partial_match_not_detected(self) -> None:
        await self._record("кофе утром бодрит и даёт силы")
        result = await self.svc.is_verbatim_copy(self.chat, "кофе утром бодрит")
        self.assertFalse(result)

    async def test_empty_text_not_detected(self) -> None:
        await self._record("кофе утром бодрит")
        result = await self.svc.is_verbatim_copy(self.chat, "")
        self.assertFalse(result)

    async def test_empty_chat_not_detected(self) -> None:
        result = await self.svc.is_verbatim_copy(self.chat, "никаких сообщений ещё нет")
        self.assertFalse(result)

    async def test_different_chat_not_detected(self) -> None:
        await self._record("пойдём пить кофе утром")
        result = await self.svc.is_verbatim_copy(999, "пойдём пить кофе утром")
        self.assertFalse(result)

    # --- hot n-gram passthroughs (L1) ---

    async def test_record_and_get_hot_ngrams_roundtrip(self) -> None:
        ngram = ("крутой", "бобёр")
        for _ in range(4):
            await self.svc.record_hot_ngrams(self.chat, [ngram])
        hot = await self.svc.get_hot_ngrams(self.chat, min_count=3, recency_share=0.5)
        self.assertEqual(hot, [ngram])

    async def test_get_hot_ngrams_empty_chat(self) -> None:
        hot = await self.svc.get_hot_ngrams(self.chat, min_count=1, recency_share=0.0)
        self.assertEqual(hot, [])

    async def test_invalidate_clears_cache(self) -> None:
        await self._record("кофе утром бодрит")
        await self.svc.is_verbatim_copy(self.chat, "кофе утром бодрит")
        self.svc._invalidate_text_cache(self.chat)
        self.assertNotIn(self.chat, self.svc._text_cache)

    # --- intonation profile (P4) ---

    async def test_intonation_profile_none_below_floor_and_cached(self) -> None:
        await self._record("привет всем")
        self.assertIsNone(await self.svc.get_intonation_profile(self.chat))
        # None is cached too: no re-read until the next message invalidates.
        self.assertIn(self.chat, self.svc._intonation)

    async def test_intonation_profile_built_and_invalidated(self) -> None:
        from unittest.mock import patch

        with patch(
            "app.services.learning_service.build_intonation_profile"
        ) as builder:
            builder.return_value = object()
            await self._record("привет всем")
            profile = await self.svc.get_intonation_profile(self.chat)
            self.assertIs(profile, builder.return_value)
            # Cached: a second read must not rebuild.
            await self.svc.get_intonation_profile(self.chat)
            builder.assert_called_once()
        await self._record("новое сообщение")
        self.assertNotIn(self.chat, self.svc._intonation)
    # --- word frequencies (slot mutations) ---

    async def test_word_frequencies_counted_from_model(self) -> None:
        await self._record("сегодня хорошая погода")
        await self._record("завтра хорошая погода")
        frequencies = await self.svc.get_word_frequencies(self.chat)
        self.assertEqual(frequencies.get("погода"), 2)
        self.assertNotIn("сегодня", frequencies)  # opener, not a continuation

    async def test_word_frequencies_trim_short_words(self) -> None:
        await self._record("вот это да ну и дела")
        frequencies = await self.svc.get_word_frequencies(self.chat)
        self.assertNotIn("да", frequencies)
        self.assertNotIn("и", frequencies)

    async def test_word_frequencies_cached_and_invalidated(self) -> None:
        await self._record("сегодня хорошая погода")
        await self.svc.get_word_frequencies(self.chat)
        self.assertIn(self.chat, self.svc._word_frequencies)
        await self._record("новое сообщение пришло")
        self.assertNotIn(self.chat, self.svc._word_frequencies)

    # --- user-interaction passthroughs (L2) ---

    async def test_user_interaction_methods_require_hasher(self) -> None:
        # The default service (no user_hasher) must fail loudly, not silently
        # store raw ids.
        with self.assertRaises(RuntimeError):
            await self.svc.record_user_interaction(self.chat, 1001)
        with self.assertRaises(RuntimeError):
            await self.svc.get_user_interaction_count(self.chat, 1001)

    async def test_user_interactions_are_hashed_and_isolated(self) -> None:
        from app.services.learning_service import LearningService

        hashed: list[int] = []

        def hasher(user_id: int) -> str:
            hashed.append(user_id)
            return f"hash-{user_id}"

        svc = LearningService(self.db, _make_generator(), user_hasher=hasher)

        await svc.record_user_interaction(self.chat, 1001)
        await svc.record_user_interaction(self.chat, 1001)
        await svc.record_user_interaction(self.chat, 2002)

        self.assertEqual(await svc.get_user_interaction_count(self.chat, 1001), 2)
        self.assertEqual(await svc.get_user_interaction_count(self.chat, 2002), 1)
        # Every DB touch went through the hasher — raw ids never reach the DB.
        self.assertEqual(hashed, [1001, 1001, 2002, 1001, 2002])
        self.assertEqual(
            await self.db.get_user_interaction_count(self.chat, "hash-1001"), 2
        )
