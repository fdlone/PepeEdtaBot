"""Tests for LearningService — prefix cache deduplication."""
from __future__ import annotations

import asyncio
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
        self.svc = LearningService(self.db, _make_generator())
        self.chat = 42

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def _record(self, text: str) -> None:
        from markov import tokenize
        tokens = tokenize(text)
        await self.svc.record_message(self.chat, text, tokens)

    # --- MessagesRepo.get_all_normalized ---

    async def test_get_all_normalized_returns_stored_texts(self) -> None:
        await self._record("кофе утром бодрит")
        await self._record("привет всем")
        result = await self.db.get_all_normalized_messages(self.chat)
        self.assertEqual(len(result), 2)

    async def test_get_all_normalized_excludes_other_chats(self) -> None:
        await self._record("кофе утром бодрит")
        other = Database(str(self.db_path))
        await other.init()
        result = await other.get_all_normalized_messages(999)
        self.assertEqual(result, [])
        await other.close()

    # --- prefix cache build and lookup ---

    async def test_prefix_cache_built_on_first_check(self) -> None:
        await self._record("пойдём пить кофе вечером")
        key = (self.chat, False)
        self.assertNotIn(key, self.svc._prefix_cache)
        await self.svc.is_duplicate(self.chat, "пойдём пить кофе вечером", False)
        self.assertIn(key, self.svc._prefix_cache)

    async def test_cache_contains_all_prefix_lengths(self) -> None:
        await self._record("один два три четыре пять")
        await self.svc.is_duplicate(self.chat, "один два три", False)
        cache = self.svc._prefix_cache[(self.chat, False)]
        self.assertIn(("один", "два", "три"), cache)
        self.assertIn(("один", "два", "три", "четыре"), cache)
        self.assertIn(("один", "два", "три", "четыре", "пять"), cache)

    async def test_cache_invalidated_after_record(self) -> None:
        await self._record("кофе утром бодрит всех нас")
        await self.svc.is_duplicate(self.chat, "кофе утром бодрит", False)
        self.assertIn((self.chat, False), self.svc._prefix_cache)
        await self._record("новое сообщение пришло сюда")
        self.assertNotIn((self.chat, False), self.svc._prefix_cache)

    async def test_separate_cache_per_normalize_lower(self) -> None:
        await self._record("Кофе утром бодрит всех нас")
        await self.svc.is_duplicate(self.chat, "кофе утром бодрит", False)
        await self.svc.is_duplicate(self.chat, "кофе утром бодрит", True)
        self.assertIn((self.chat, False), self.svc._prefix_cache)
        self.assertIn((self.chat, True), self.svc._prefix_cache)

    # --- is_duplicate behaviour ---

    async def test_exact_match_detected(self) -> None:
        await self._record("кофе утром бодрит")
        # 3 tokens → exact match via prefix cache
        result = await self.svc.is_duplicate(self.chat, "кофе утром бодрит", False)
        self.assertTrue(result)

    async def test_prefix_extension_detected(self) -> None:
        """Bot generates stored prefix + new tokens — 3-gram must be in cache."""
        await self._record("пойдём пить кофе")
        # Build cache by triggering a check
        await self.svc.is_duplicate(self.chat, "пойдём пить кофе утром вечером", False)
        cache = self.svc._prefix_cache.get((self.chat, False), set())
        # The 3-token prefix of the stored message must be in the cache
        self.assertIn(("пойдём", "пить", "кофе"), cache,
                      "3-gram from training must be detected as a known prefix")

    async def test_unique_text_passes(self) -> None:
        await self._record("пойдём пить кофе")
        result = await self.svc.is_duplicate(
            self.chat, "совершенно другое предложение здесь", False
        )
        self.assertFalse(result)

    async def test_short_text_uses_sql_fallback(self) -> None:
        """Text with < 3 tokens falls back to SQL exact match."""
        await self._record("привет мир")
        # 2 tokens — prefix cache can't help, SQL must catch it
        result = await self.svc.is_duplicate(self.chat, "привет мир", False)
        self.assertTrue(result)

    async def test_short_unique_text_passes(self) -> None:
        result = await self.svc.is_duplicate(self.chat, "уникально", False)
        self.assertFalse(result)

    async def test_empty_chat_no_duplicates(self) -> None:
        result = await self.svc.is_duplicate(
            self.chat, "никаких сообщений ещё нет тут", False
        )
        self.assertFalse(result)

    async def test_different_chat_not_duplicate(self) -> None:
        await self._record("пойдём пить кофе утром всем")
        result = await self.svc.is_duplicate(
            999, "пойдём пить кофе утром всем", False
        )
        self.assertFalse(result)

    async def test_invalidate_clears_all_normalize_variants(self) -> None:
        await self._record("кофе утром бодрит всех нас")
        await self.svc.is_duplicate(self.chat, "кофе утром", False)
        await self.svc.is_duplicate(self.chat, "кофе утром", True)
        self.svc._invalidate_prefix_cache(self.chat)
        self.assertNotIn((self.chat, False), self.svc._prefix_cache)
        self.assertNotIn((self.chat, True), self.svc._prefix_cache)
