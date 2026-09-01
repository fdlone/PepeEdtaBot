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

    # --- MarkovRepo.get_recent_normalized ---

    async def test_get_recent_normalized_returns_stored_texts(self) -> None:
        await self._record("кофе утром бодрит")
        await self._record("привет всем")
        result = await self.db.markov.get_recent_normalized(self.chat, 10)
        self.assertEqual(len(result), 2)

    async def test_get_recent_normalized_excludes_other_chats(self) -> None:
        await self._record("кофе утром бодрит")
        other = Database(str(self.db_path))
        await other.init()
        result = await other.markov.get_recent_normalized(999, 10)
        self.assertEqual(result, [])
        await other.close()

    async def test_get_recent_normalized_respects_limit_and_keeps_recent_order(self) -> None:
        await self._record("первое сообщение")
        await self._record("второе сообщение")
        await self._record("третье сообщение")
        await self._record("четвертое сообщение")
        result = await self.db.markov.get_recent_normalized(self.chat, 2)
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

        await self.svc.get_context_idf(self.chat)
        await self.svc.get_intonation_profile(self.chat)

        self.svc.forget_chat(self.chat)

        # Интроспекцией, а не списком имён. Прежняя форма перечисляла три
        # словаря из одиннадцати и потому не охраняла ловушку §5 CLAUDE.md
        # («новый пер-чатовый кэш обязан добавиться в forget_chat, иначе
        # /clear его переживёт»): новый забытый кэш проходил зелёным, а имя
        # теста при этом гасило подозрение. Здесь падение наступает от самого
        # факта появления кэша, не внесённого в forget_chat.
        leaked = [
            name
            for name, value in vars(self.svc).items()
            if isinstance(value, dict) and self.chat in value
        ]
        self.assertEqual(leaked, [], f"кэши пережили forget_chat: {leaked}")

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

    async def test_warm_frequency_cache_draws_like_a_cold_read(self) -> None:
        """Тёплый кэш и холодное чтение дают одну и ту же замену.

        Соседний тест сверяет только состав словаря, а в розыгрыш попадает ещё
        и порядок: ``pick_replacement`` строит пул обходом словаря и отдаёт его
        в ``rng.choices``, который выбирает по позиции. Тёплый кэш дописывает
        выученные слова в конец, холодное чтение отсортировано — то есть до
        сортировки в ``pick_replacement`` один и тот же чат при одном и том же
        сиде сэмплировал замену по-разному в проде (тёплый) и под
        ``generation_hash`` (холодный).

        «Банка» учится последней специально: в тёплом словаре она в конце, в
        холодном — первая по алфавиту. Без этого порядки совпали бы и тест
        ничего не сторожил.
        """
        import random

        from app.core.slot_mutation import pick_replacement

        for text in (
            "один два лодка лодка лодка",
            "три четыре марка марка марка",
            "пять шесть ветка ветка ветка",
        ):
            await self._record(text)
        await self.svc.get_word_frequencies(self.chat)  # прогрев кэша

        await self._record("семь восемь банка банка банка")

        warm = await self.svc.get_word_frequencies(self.chat)
        cold = await (await self._fresh_service()).get_word_frequencies(self.chat)

        self.assertEqual(dict(warm), dict(cold), "состав словаря обязан совпадать")
        self.assertNotEqual(
            list(warm), list(cold), "иначе тест не проверяет расхождение порядка"
        )

        draws = {
            pick_replacement(
                "кошка", source, excluded_tokens=frozenset(), rng=random.Random(seed)
            )
            for source in (warm, cold)
            for seed in range(30)
        }
        self.assertNotIn(None, draws, "пул замен пуст — тест ничего не проверяет")
        self.assertGreater(len(draws), 1, "все сиды дали одно слово — пул вырожден")

        for seed in range(30):
            self.assertEqual(
                pick_replacement(
                    "кошка", warm, excluded_tokens=frozenset(), rng=random.Random(seed)
                ),
                pick_replacement(
                    "кошка", cold, excluded_tokens=frozenset(), rng=random.Random(seed)
                ),
                f"тёплый и холодный путь разошлись на сиде {seed}",
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


class TestMemeSettingsPassThrough(unittest.IsolatedAsyncioTestCase):
    """M2R-300: the daily pass runs on the settings the caller passes in.

    The constructor snapshot is only the fallback — without the pass-through
    a /set of the meme knobs would do nothing until a restart.
    """

    async def test_maintenance_uses_the_passed_meme_settings(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from app.infrastructure.database import MaintenanceOutcome
        from app.services.learning_service import LearningService
        from app.services.meme_analyzer import AnalysisResult, MemeSettings

        db = MagicMock()
        db.decay_flavor_stats_if_due = AsyncMock(
            return_value=MaintenanceOutcome.DONE
        )
        db.list_chat_ids = AsyncMock(return_value=[1])
        generator = _make_generator()
        generator.telemetry = MagicMock()
        service = LearningService(db, generator)
        analyze = AsyncMock(
            return_value=AnalysisResult(
                scored_pairs=0, stored_pairs=0, duration_ms=1.0
            )
        )
        with patch(
            "app.services.learning_service.analyze_chat_memes", analyze
        ), patch(
            "app.services.learning_service.mask_chat_id", return_value="chat"
        ):
            await service.run_due_maintenance(
                MemeSettings(
                    min_joint_count=5,
                    min_support=20.0,
                    recency_days=7.0,
                    max_entries=50,
                )
            )

        kwargs = analyze.await_args.kwargs
        self.assertEqual(kwargs["min_joint_count"], 5)
        self.assertEqual(kwargs["min_support"], 20.0)
        self.assertEqual(kwargs["recency_days"], 7.0)
        self.assertEqual(kwargs["max_entries"], 50)


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


class TestPhraseIndexRebuildInTheDailyPass(unittest.IsolatedAsyncioTestCase):
    """Пересборка фразового индекса — фоновая бухгалтерия на learn-пути.

    Требования к ней те же, что к мем-пассу рядом, и по той же причине: её сбой
    не должен стоить выученного сообщения. Плюс одно своё — она обходит все
    чаты, поэтому не имеет права читать настройки чата-триггера.
    """

    def _service(self, *, chat_ids: list[int], rebuild: object):
        from unittest.mock import AsyncMock, MagicMock

        from app.infrastructure.database import MaintenanceOutcome
        from app.services.learning_service import LearningService

        db = MagicMock()
        db.decay_flavor_stats_if_due = AsyncMock(return_value=MaintenanceOutcome.DONE)
        db.list_chat_ids = AsyncMock(return_value=chat_ids)
        db.chat_phrase_ngrams.rebuild_chat = rebuild
        generator = _make_generator()
        generator.telemetry = MagicMock()
        return LearningService(db, generator), db

    async def test_every_chat_is_rebuilt(self) -> None:
        from unittest.mock import AsyncMock, patch

        rebuild = AsyncMock(return_value=3)
        service, _ = self._service(chat_ids=[11, 22], rebuild=rebuild)

        with patch(
            "app.services.learning_service.analyze_chat_memes", AsyncMock()
        ), patch("app.services.learning_service.mask_chat_id", return_value="chat"):
            await service.run_due_maintenance()

        self.assertEqual(
            [call.args[0] for call in rebuild.await_args_list], [11, 22]
        )

    async def test_a_failing_chat_does_not_stop_the_pass(self) -> None:
        from unittest.mock import AsyncMock, patch

        rebuild = AsyncMock(side_effect=[RuntimeError("database is locked"), 5])
        service, _ = self._service(chat_ids=[11, 22], rebuild=rebuild)

        with patch(
            "app.services.learning_service.analyze_chat_memes", AsyncMock()
        ), patch("app.services.learning_service.mask_chat_id", return_value="chat"):
            # Не пробрасывается наружу: наверху этого вызова — выученное
            # сообщение, и оно дороже пересборки индекса.
            await service.run_due_maintenance()

        self.assertEqual(rebuild.await_count, 2)

    async def test_listing_chats_failing_does_not_stop_the_pass(self) -> None:
        from unittest.mock import AsyncMock, patch

        rebuild = AsyncMock()
        service, db = self._service(chat_ids=[], rebuild=rebuild)
        db.list_chat_ids = AsyncMock(side_effect=RuntimeError("database is locked"))

        with patch(
            "app.services.learning_service.analyze_chat_memes", AsyncMock()
        ), patch("app.services.learning_service.mask_chat_id", return_value="chat"):
            await service.run_due_maintenance()

        rebuild.assert_not_awaited()

    async def test_rebuild_is_called_with_a_chat_and_nothing_else(self) -> None:
        """Ни одной ручки: иначе пасс применил бы настройки чата-триггера ко всем."""
        from unittest.mock import AsyncMock, patch

        rebuild = AsyncMock(return_value=0)
        service, _ = self._service(chat_ids=[11], rebuild=rebuild)

        with patch(
            "app.services.learning_service.analyze_chat_memes", AsyncMock()
        ), patch("app.services.learning_service.mask_chat_id", return_value="chat"):
            from app.services.meme_analyzer import MemeSettings

            await service.run_due_maintenance(
                MemeSettings(
                    min_joint_count=5,
                    min_support=20.0,
                    recency_days=7.0,
                    max_entries=50,
                )
            )

        self.assertEqual(rebuild.await_args.args, (11,))
        self.assertEqual(rebuild.await_args.kwargs, {})

    async def test_the_log_carries_numbers_and_no_phrases(self) -> None:
        from unittest.mock import AsyncMock, patch

        from app import log_masking

        # Настоящее маскирование, а не подменённое: проверяется как раз то,
        # что сырой chat_id в строку не попадает.
        log_masking.init_masking("test-secret-for-phrase-index")
        rebuild = AsyncMock(return_value=7)
        service, _ = self._service(chat_ids=[-1001234567890], rebuild=rebuild)

        with patch(
            "app.services.learning_service.analyze_chat_memes", AsyncMock()
        ), self.assertLogs("chat_markov", level="INFO") as logs:
            await service.run_due_maintenance()

        line = next(line for line in logs.output if "phrase index" in line)
        self.assertIn("phrases=7", line)
        # Сырого chat_id в строке нет — он маскируется, как везде (§4 CLAUDE.md).
        self.assertNotIn("1001234567890", line)


if __name__ == "__main__":
    unittest.main()
