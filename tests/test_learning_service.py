"""Tests for LearningService — prefix-match novelty heuristic."""
from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from db import Database
from markov import MarkovGenerator


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
        self.db = Database(str(self.db_path))
        await self.db.init()
        self.svc = LearningService(
            self.db,
            _make_generator(),
            prefix_cache_max_messages=3,
        )
        self.chat = 42

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def _record(self, text: str) -> None:
        from markov import tokenize
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

    # --- prefix cache build and lookup ---

    async def test_prefix_cache_built_on_first_check(self) -> None:
        await self._record("пойдём пить кофе вечером")
        key = (self.chat, False)
        self.assertNotIn(key, self.svc._prefix_cache)
        await self.svc.matches_training_prefix(
            self.chat, "пойдём пить кофе вечером", False
        )
        self.assertIn(key, self.svc._prefix_cache)

    async def test_cache_contains_all_prefix_lengths(self) -> None:
        await self._record("один два три четыре пять")
        await self.svc.matches_training_prefix(
            self.chat, "один два три", False
        )
        cache = self.svc._prefix_cache[(self.chat, False)]
        self.assertIn(("один", "два", "три"), cache)
        self.assertIn(("один", "два", "три", "четыре"), cache)
        self.assertIn(("один", "два", "три", "четыре", "пять"), cache)

    async def test_cache_invalidated_after_record(self) -> None:
        await self._record("кофе утром бодрит всех нас")
        await self.svc.matches_training_prefix(
            self.chat, "кофе утром бодрит", False
        )
        self.assertIn((self.chat, False), self.svc._prefix_cache)
        await self._record("новое сообщение пришло сюда")
        self.assertNotIn((self.chat, False), self.svc._prefix_cache)

    async def test_separate_cache_per_normalize_lower(self) -> None:
        await self._record("Кофе утром бодрит всех нас")
        await self.svc.matches_training_prefix(
            self.chat, "кофе утром бодрит", False
        )
        await self.svc.matches_training_prefix(
            self.chat, "кофе утром бодрит", True
        )
        self.assertIn((self.chat, False), self.svc._prefix_cache)
        self.assertIn((self.chat, True), self.svc._prefix_cache)

    # --- similarity heuristic behaviour ---

    async def test_exact_match_treated_as_too_close(self) -> None:
        await self._record("кофе утром бодрит")
        # 3 tokens → exact match via prefix cache
        result = await self.svc.matches_training_prefix(
            self.chat, "кофе утром бодрит", False
        )
        self.assertTrue(result)

    async def test_prefix_extension_is_present_in_similarity_cache(self) -> None:
        """Bot generates stored prefix + new tokens — 3-gram must be in cache."""
        await self._record("пойдём пить кофе")
        # Build cache by triggering a check
        await self.svc.matches_training_prefix(
            self.chat, "пойдём пить кофе утром вечером", False
        )
        cache = self.svc._prefix_cache.get((self.chat, False), set())
        # The 3-token prefix of the stored message must be in the cache
        self.assertIn(("пойдём", "пить", "кофе"), cache,
                      "3-gram from training must be available to the heuristic")

    async def test_prefix_cache_uses_only_recent_messages_window(self) -> None:
        await self._record("старый префикс должен выпасть")
        await self._record("новый один два три")
        await self._record("новый четыре пять шесть")
        await self._record("новый семь восемь девять")

        self.assertFalse(
            await self.svc.matches_training_prefix(
                self.chat, "старый префикс должен выпасть потом", False
            )
        )
        self.assertTrue(
            await self.svc.matches_training_prefix(
                self.chat, "новый один два три потом", False
            )
        )

    async def test_unique_text_passes(self) -> None:
        await self._record("пойдём пить кофе")
        result = await self.svc.matches_training_prefix(
            self.chat, "совершенно другое предложение здесь", False
        )
        self.assertFalse(result)

    async def test_short_text_passes_through(self) -> None:
        """Text with < 3 tokens is skipped by the prefix filter (returns False).

        Such candidates are already filtered by exact-match `message_exists`
        inside `MarkovGenerator.generate_text`, so a duplicate check here
        would be redundant and noisy.
        """
        await self._record("привет мир")
        result = await self.svc.matches_training_prefix(
            self.chat, "привет мир", False
        )
        self.assertFalse(result)

    async def test_short_unique_text_passes(self) -> None:
        result = await self.svc.matches_training_prefix(
            self.chat, "уникально", False
        )
        self.assertFalse(result)

    async def test_empty_chat_has_no_too_close_samples(self) -> None:
        result = await self.svc.matches_training_prefix(
            self.chat, "никаких сообщений ещё нет тут", False
        )
        self.assertFalse(result)

    async def test_different_chat_not_too_close(self) -> None:
        await self._record("пойдём пить кофе утром всем")
        result = await self.svc.matches_training_prefix(
            999, "пойдём пить кофе утром всем", False
        )
        self.assertFalse(result)

    async def test_invalidate_clears_all_normalize_variants(self) -> None:
        await self._record("кофе утром бодрит всех нас")
        await self.svc.matches_training_prefix(
            self.chat, "кофе утром", False
        )
        await self.svc.matches_training_prefix(
            self.chat, "кофе утром", True
        )
        self.svc._invalidate_prefix_cache(self.chat)
        self.assertNotIn((self.chat, False), self.svc._prefix_cache)
        self.assertNotIn((self.chat, True), self.svc._prefix_cache)
