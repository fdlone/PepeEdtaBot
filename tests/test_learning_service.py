"""Tests for LearningService — verbatim-copy detection via text cache."""
from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

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
        result = await self.db.messages.get_recent_normalized(self.chat, 10)
        self.assertEqual(len(result), 2)

    async def test_get_recent_normalized_excludes_other_chats(self) -> None:
        await self._record("кофе утром бодрит")
        other = Database(str(self.db_path))
        await other.init()
        result = await other.messages.get_recent_normalized(999, 10)
        self.assertEqual(result, [])
        await other.close()

    async def test_get_recent_normalized_respects_limit_and_keeps_recent_order(self) -> None:
        await self._record("первое сообщение")
        await self._record("второе сообщение")
        await self._record("третье сообщение")
        await self._record("четвертое сообщение")
        result = await self.db.messages.get_recent_normalized(self.chat, 2)
        self.assertEqual(result, ["третье сообщение", "четвертое сообщение"])

    # --- text cache build and lookup ---

    async def test_text_cache_built_on_first_check(self) -> None:
        await self._record("пойдём пить кофе вечером")
        self.assertNotIn(self.chat, self.svc._text_counts)
        await self.svc.is_verbatim_copy(self.chat, "пойдём пить кофе вечером")
        self.assertIn(self.chat, self.svc._text_counts)

    async def test_new_message_updates_the_cache_instead_of_dropping_it(self) -> None:
        """Кэш пополняется сообщением, а не сбрасывается им.

        Сброс на каждое сообщение означал, что в активном чате почти каждый
        ответ шёл по холодному пути: следующий же ответ перечитывал выборку
        целиком. Кэш, гарантированно холодный к моменту использования, своей
        задачи не выполняет.
        """
        from unittest.mock import patch

        await self._record("кофе утром бодрит")
        await self.svc.is_verbatim_copy(self.chat, "кофе утром бодрит")

        await self._record("новое сообщение")

        self.assertIn(self.chat, self.svc._text_counts)
        with patch.object(
            self.svc, "_get_recent_texts", side_effect=AssertionError("rebuilt")
        ):
            self.assertTrue(
                await self.svc.is_verbatim_copy(self.chat, "новое сообщение")
            )
            self.assertTrue(
                await self.svc.is_verbatim_copy(self.chat, "кофе утром бодрит")
            )

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

    async def test_forget_chat_clears_caches(self) -> None:
        await self._record("кофе утром бодрит")
        await self.svc.is_verbatim_copy(self.chat, "кофе утром бодрит")
        await self.svc.get_word_frequencies(self.chat)

        self.svc.forget_chat(self.chat)

        self.assertNotIn(self.chat, self.svc._text_counts)
        self.assertNotIn(self.chat, self.svc._word_frequencies)
        self.assertNotIn(self.chat, self.svc._ngram_index)

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
            # Одно новое сообщение не двигает статистику по выборке в сотни
            # сообщений — пересборка откладывается.
            await self._record("новое сообщение")
            await self.svc.get_intonation_profile(self.chat)
            builder.assert_called_once()
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

    async def test_word_frequencies_absorb_new_message(self) -> None:
        from unittest.mock import patch

        await self._record("сегодня хорошая погода")
        await self.svc.get_word_frequencies(self.chat)

        await self._record("завтра хорошая погода")

        with patch.object(
            self.svc._db.markov,
            "get_word_frequencies",
            side_effect=AssertionError("re-read"),
        ):
            frequencies = await self.svc.get_word_frequencies(self.chat)
        self.assertEqual(frequencies.get("погода"), 2)

    # --- эквивалентность инкрементального обновления и полной сборки ---

    async def _fresh_service(self):
        """Сервис с пустыми кэшами на той же базе — эталон полной сборки."""
        from app.services.learning_service import LearningService

        return LearningService(
            self.db, _make_generator(), text_cache_max_messages=3
        )

    async def test_incremental_text_window_matches_full_rebuild(self) -> None:
        texts = ["первое сообщение", "второе сообщение", "третье", "четвёртое"]
        for text in texts:
            await self._record(text)
            await self.svc.is_verbatim_copy(self.chat, text)

        fresh = await self._fresh_service()
        for text in texts:
            self.assertEqual(
                await self.svc.is_verbatim_copy(self.chat, text),
                await fresh.is_verbatim_copy(self.chat, text),
                text,
            )

    async def test_incremental_ngram_index_matches_full_rebuild(self) -> None:
        await self.svc.get_verbatim_ngram_index(self.chat)
        for text in ("кофе утром бодрит очень", "чай вечером успокаивает тоже"):
            await self._record(text)

        fresh = await self._fresh_service()

        self.assertEqual(
            set(await self.svc.get_verbatim_ngram_index(self.chat)),
            set(await fresh.get_verbatim_ngram_index(self.chat)),
        )

    async def test_incremental_frequencies_match_full_rebuild(self) -> None:
        await self.svc.get_word_frequencies(self.chat)
        for text in (
            "сегодня хорошая погода правда",
            "завтра хорошая погода тоже",
            "вчера была хорошая погода",
        ):
            await self._record(text)

        fresh = await self._fresh_service()

        self.assertEqual(
            dict(await self.svc.get_word_frequencies(self.chat)),
            dict(await fresh.get_word_frequencies(self.chat)),
        )

    # --- отсрочка тяжёлой статистики ---

    async def test_idf_rebuilds_after_the_deferred_threshold(self) -> None:
        from app.services.learning_service import STATS_REBUILD_EVERY_MESSAGES

        await self.svc.get_context_idf(self.chat)
        for index in range(STATS_REBUILD_EVERY_MESSAGES):
            await self._record(f"сообщение номер {index} здесь")

        reads: list[int] = []
        original = self.svc._get_recent_texts

        async def counting(chat_id: int) -> list[str]:
            reads.append(chat_id)
            return await original(chat_id)

        self.svc._get_recent_texts = counting  # type: ignore[method-assign]
        await self.svc.get_context_idf(self.chat)

        self.assertEqual(len(reads), 1)

    async def test_idf_is_not_rebuilt_on_every_message(self) -> None:
        await self.svc.get_context_idf(self.chat)
        await self._record("одно новое сообщение здесь")

        with patch.object(
            self.svc, "_get_recent_texts", side_effect=AssertionError("rebuilt")
        ):
            await self.svc.get_context_idf(self.chat)

    # --- вытеснение ---

    async def test_caches_of_a_quiet_chat_are_evicted_by_ttl(self) -> None:
        from app.services.learning_service import LearningService

        svc = LearningService(
            self.db, _make_generator(), text_cache_max_messages=3, cache_ttl_sec=0
        )
        await svc.record_message(self.chat, "кофе утром бодрит", ["кофе", "утром"])
        await svc.is_verbatim_copy(self.chat, "кофе утром бодрит")

        svc._prune(svc._last_touch[self.chat] + 1.0)

        self.assertNotIn(self.chat, svc._text_counts)

    async def test_caches_are_capped_by_chat_count(self) -> None:
        from app.services.learning_service import LearningService

        svc = LearningService(
            self.db, _make_generator(), text_cache_max_messages=3, cache_max_chats=2
        )
        for chat_id in (1, 2, 3):
            await svc.record_message(chat_id, "кофе утром бодрит", ["кофе", "утром"])
            await svc.is_verbatim_copy(chat_id, "кофе утром бодрит")

        svc._prune(max(svc._last_touch.values()))

        self.assertLessEqual(len(svc._last_touch), 2)
        # Вытеснение не меняет ответа: данные читаются заново из базы.
        self.assertTrue(await svc.is_verbatim_copy(1, "кофе утром бодрит"))

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
            await self.db.chat_user_interactions.get_count(self.chat, "hash-1001"), 2
        )


class TestMaintenanceAlerts(unittest.IsolatedAsyncioTestCase):
    """Когда сбой обслуживания стоит того, чтобы беспокоить владельца.

    Логи владельцу недоступны, а сбой снаружи выглядит как «бот жив, просто
    мемы перестали выветриваться». Но и сигнал не должен стать потоком: разовая
    неудача — это обычная занятая база.
    """

    def setUp(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from app.services.learning_service import LearningService

        self.db = MagicMock()
        self.db.decay_flavor_stats_if_due = AsyncMock()
        self.db.last_maintenance_error = "database is locked"
        self.svc = LearningService(self.db, _make_generator())

    def _returns(self, name: str) -> None:
        from app.infrastructure.database import MaintenanceOutcome

        self.db.decay_flavor_stats_if_due.return_value = MaintenanceOutcome[name]

    def _failing_since(self, seconds_ago: float) -> None:
        import time

        self.svc._maintenance_failing_since = time.monotonic() - seconds_ago

    async def test_skipped_run_says_nothing(self) -> None:
        self._returns("SKIPPED")
        self.assertIsNone(await self.svc.run_due_maintenance())

    async def test_single_failure_is_not_worth_a_message(self) -> None:
        self._returns("FAILED")
        self.assertIsNone(await self.svc.run_due_maintenance())

    async def test_sustained_failure_alerts_once(self) -> None:
        from app.services.learning_service import MAINTENANCE_ALERT_AFTER_SEC

        self._returns("FAILED")
        await self.svc.run_due_maintenance()
        self._failing_since(MAINTENANCE_ALERT_AFTER_SEC + 1.0)

        alert = await self.svc.run_due_maintenance()

        assert alert is not None
        self.assertFalse(alert.recovered)
        self.assertEqual(alert.reason, "database is locked")
        self.assertGreater(alert.failing_for_sec, MAINTENANCE_ALERT_AFTER_SEC)
        # Повторный сигнал не уходит, пока не пройдут сутки: отказ не должен
        # сам стать потоком сообщений от бота.
        self.assertIsNone(await self.svc.run_due_maintenance())

    async def test_reminder_returns_after_the_repeat_window(self) -> None:
        import time

        from app.services.learning_service import (
            MAINTENANCE_ALERT_AFTER_SEC,
            MAINTENANCE_ALERT_REPEAT_SEC,
        )

        self._returns("FAILED")
        self._failing_since(MAINTENANCE_ALERT_AFTER_SEC + 1.0)
        self.assertIsNotNone(await self.svc.run_due_maintenance())

        self.svc._maintenance_alerted_at = (
            time.monotonic() - MAINTENANCE_ALERT_REPEAT_SEC - 1.0
        )
        self.assertIsNotNone(await self.svc.run_due_maintenance())

    async def test_recovery_is_reported_only_if_the_breakage_was(self) -> None:
        from app.services.learning_service import MAINTENANCE_ALERT_AFTER_SEC

        # Починка без предшествующего сигнала — не новость.
        self._returns("DONE")
        self.assertIsNone(await self.svc.run_due_maintenance())

        self._returns("FAILED")
        self._failing_since(MAINTENANCE_ALERT_AFTER_SEC + 1.0)
        self.assertIsNotNone(await self.svc.run_due_maintenance())

        self._returns("DONE")
        alert = await self.svc.run_due_maintenance()

        assert alert is not None
        self.assertTrue(alert.recovered)
        # Состояние сброшено: следующая поломка снова копит свой час.
        self._returns("FAILED")
        self.assertIsNone(await self.svc.run_due_maintenance())
