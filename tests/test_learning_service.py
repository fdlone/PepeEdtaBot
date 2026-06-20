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
        self.db = Database(str(self.db_path))
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

    async def test_invalidate_clears_cache(self) -> None:
        await self._record("кофе утром бодрит")
        await self.svc.is_verbatim_copy(self.chat, "кофе утром бодрит")
        self.svc._invalidate_text_cache(self.chat)
        self.assertNotIn(self.chat, self.svc._text_cache)
