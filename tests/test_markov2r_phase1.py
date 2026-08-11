"""Markov 2.0R Phase 1: diagnostics, shadow order-4, incremental cache.

The load-bearing suite is the equivalence block: a cache updated by folding
learned-message deltas must be indistinguishable — content-wise and
generation-wise — from a cache read cold from SQL (spec:
in-memory-state-eviction, "Генерация не зависит от пути к данным").
"""
from __future__ import annotations

import math
import os
import random
import unittest
import uuid

from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.context_state_matcher import ContextStateMatcher
from app.core.generation_telemetry import GenerationTelemetry
from app.core.markov import (
    MarkovGenerator,
    _fold_start_row,
    _fold_transition,
    pool_diagnostics,
    tokenize,
)
from app.core.shadow_order import (
    SHADOW_ORDER4_MIN_COUNT,
    shadow_order4_stats,
)
from app.infrastructure.database import Database
from app.presentation.bot_messages import format_stats_message
from app.services.learning_service import LearningService

CHAT_ID = -1001234567890


class TestPoolDiagnostics(unittest.TestCase):
    def test_degenerate_pools(self) -> None:
        self.assertEqual(pool_diagnostics([]), (0.0, 0.0, 0, 1.0))
        self.assertEqual(pool_diagnostics([7]), (0.0, 0.0, 1, 1.0))

    def test_uniform_pool_max_entropy(self) -> None:
        entropy, normalized, branching, confidence = pool_diagnostics([5, 5, 5, 5])
        self.assertAlmostEqual(entropy, 2.0)
        self.assertAlmostEqual(normalized, 1.0)
        self.assertEqual(branching, 4)
        self.assertAlmostEqual(confidence, 0.0)

    def test_skewed_pool_low_entropy(self) -> None:
        _entropy, normalized, _branching, confidence = pool_diagnostics([100, 1])
        self.assertLess(normalized, 0.2)
        self.assertGreater(confidence, 0.8)

    @settings(max_examples=100, deadline=None)
    @given(st.lists(st.integers(min_value=0, max_value=10_000), max_size=40))
    def test_invariants(self, counts: list[int]) -> None:
        entropy, normalized, branching, confidence = pool_diagnostics(counts)
        self.assertEqual(branching, len(counts))
        self.assertGreaterEqual(entropy, 0.0)
        if branching > 1:
            self.assertLessEqual(entropy, math.log2(branching) + 1e-9)
        self.assertGreaterEqual(normalized, 0.0)
        self.assertLessEqual(normalized, 1.0)
        self.assertAlmostEqual(confidence, 1.0 - normalized)


class TestShadowOrder4(unittest.TestCase):
    def test_planted_high_support_state_selected(self) -> None:
        # One 4-token state seen 5 times, always continuing the same way:
        # support >= threshold, confidence 1.0 -> the shadow selector fires.
        index = {("а", "б", "в", "г"): {"д": 5}}
        eligible, selected = shadow_order4_stats(
            ["а", "б", "в", "г", "д", "е"], index
        )
        self.assertEqual(eligible, 2)  # positions 4 and 5
        self.assertEqual(selected, 1)

    def test_low_support_not_selected(self) -> None:
        index = {("а", "б", "в", "г"): {"д": SHADOW_ORDER4_MIN_COUNT - 1}}
        _eligible, selected = shadow_order4_stats(["а", "б", "в", "г", "д"], index)
        self.assertEqual(selected, 0)

    def test_high_entropy_pool_not_selected(self) -> None:
        # Plenty of support but a uniform continuation pool: confidence 0.
        index = {("а", "б", "в", "г"): {"д": 3, "е": 3, "ж": 3, "з": 3}}
        _eligible, selected = shadow_order4_stats(["а", "б", "в", "г", "д"], index)
        self.assertEqual(selected, 0)

    def test_short_reply_has_no_eligible_steps(self) -> None:
        self.assertEqual(shadow_order4_stats(["а", "б", "в", "г"], {}), (0, 0))

    def test_casefold_matching(self) -> None:
        index = {("а", "б", "в", "г"): {"д": 5}}
        _eligible, selected = shadow_order4_stats(["А", "Б", "В", "Г", "Д"], index)
        self.assertEqual(selected, 1)


class TestTelemetryAndStats(unittest.TestCase):
    def test_snapshot_empty(self) -> None:
        snapshot = GenerationTelemetry().snapshot()
        self.assertEqual(snapshot["generations"], 0)
        self.assertIsNone(snapshot["mean_entropy_bits"])
        self.assertIsNone(snapshot["cache_hit_rate"])
        self.assertIsNone(snapshot["shadow_order4_selected_share"])

    def test_snapshot_aggregates(self) -> None:
        telemetry = GenerationTelemetry()
        telemetry.note_cache(hit=True)
        telemetry.note_cache(hit=True)
        telemetry.note_cache(hit=False)
        telemetry.note_generation(
            entropy_bits_sum=4.0,
            normalized_entropy_sum=1.0,
            branching_sum=6.0,
            steps=2,
        )
        telemetry.note_shadow(eligible=10, selected=3)
        snapshot = telemetry.snapshot()
        self.assertAlmostEqual(snapshot["cache_hit_rate"] or 0.0, 2 / 3)
        self.assertAlmostEqual(snapshot["mean_entropy_bits"] or 0.0, 2.0)
        self.assertAlmostEqual(snapshot["mean_branching"] or 0.0, 3.0)
        self.assertAlmostEqual(
            snapshot["shadow_order4_selected_share"] or 0.0, 0.3
        )

    def test_stats_message_without_telemetry(self) -> None:
        self.assertEqual(
            format_stats_message({"volume": 42}), "объём модели: 42"
        )

    def test_stats_message_with_telemetry(self) -> None:
        telemetry = GenerationTelemetry()
        telemetry.note_cache(hit=True)
        telemetry.note_generation(
            entropy_bits_sum=1.0,
            normalized_entropy_sum=0.5,
            branching_sum=3.0,
            steps=1,
        )
        telemetry.note_shadow(eligible=20, selected=2)
        text = format_stats_message({"volume": 7}, telemetry.snapshot())
        self.assertIn("объём модели: 7", text)
        self.assertIn("кэш распределений", text)
        self.assertIn("order-4 (тень", text)


class TestFoldHelpers(unittest.TestCase):
    @settings(max_examples=100, deadline=None)
    @given(
        st.lists(
            st.tuples(st.text("абвгд", min_size=1, max_size=3), st.integers(1, 50)),
            max_size=20,
            unique_by=lambda item: item[0],
        ),
        st.text("абвгд", min_size=1, max_size=3),
        st.integers(min_value=1, max_value=5),
    )
    def test_fold_transition_equals_rebuild(
        self, rows: list[tuple[str, int]], token: str, delta: int
    ) -> None:
        ordered = sorted(rows, key=lambda row: row[0])
        folded = _fold_transition(ordered, token, delta)
        merged = dict(ordered)
        merged[token] = merged.get(token, 0) + delta
        rebuilt = sorted(merged.items(), key=lambda row: row[0])
        self.assertEqual(folded, rebuilt)
        self.assertEqual(ordered, sorted(rows, key=lambda row: row[0]))  # copy-on-write

    def test_fold_start_row_insert_and_increment(self) -> None:
        rows = [("а", "б", 2), ("в", "г", 1)]
        incremented = _fold_start_row(rows, ("а", "б"))
        self.assertEqual(incremented, [("а", "б", 3), ("в", "г", 1)])
        inserted = _fold_start_row(rows, ("б", "б"))
        self.assertEqual(
            inserted, [("а", "б", 2), ("б", "б", 1), ("в", "г", 1)]
        )
        self.assertEqual(rows, [("а", "б", 2), ("в", "г", 1)])  # untouched


class _Phase1DbCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = f"test_db_{uuid.uuid4().hex}.sqlite"
        self.db = Database(self.db_path)
        await self.db.init()
        self.generator = MarkovGenerator(db=self.db.markov)
        self.learning = LearningService(self.db, self.generator)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        for suffix in ("", "-wal", "-shm"):
            path = self.db_path + suffix
            if os.path.exists(path):
                os.remove(path)

    async def _learn(self, text: str, *, incremental: bool) -> None:
        tokens = tokenize(text, normalize_lower=True)
        await self.learning.record_message(
            CHAT_ID, text, tokens, incremental_cache=incremental
        )

    async def _warm_caches(self, messages: list[str]) -> None:
        await self.generator._get_starts3(CHAT_ID)
        await self.generator._get_starts2(CHAT_ID)
        for message in messages:
            tokens = tokenize(message, normalize_lower=True)
            for i in range(len(tokens) - 3):
                await self.generator._get3(
                    CHAT_ID, tokens[i], tokens[i + 1], tokens[i + 2]
                )
            for i in range(len(tokens) - 2):
                await self.generator._get2(CHAT_ID, tokens[i], tokens[i + 1])


CORPUS = [
    "коты любят тёплое молоко утром",
    "коты любят спать на батарее",
    "собаки любят тёплое молоко тоже",
    "утром хорошо пить чай с мёдом",
    "вечером коты спят на батарее",
]
EXTRA = [
    "коты любят тёплое одеяло зимой",
    "зимой хорошо спать под одеялом",
    "чай с мёдом лучше пить вечером",
]


class TestIncrementalCacheEquivalence(_Phase1DbCase):
    async def test_folded_caches_equal_fresh_reads(self) -> None:
        for message in CORPUS:
            await self._learn(message, incremental=False)
        await self._warm_caches(CORPUS + EXTRA)
        for message in EXTRA:
            await self._learn(message, incremental=True)

        # Every cached entry must equal a fresh SQL read after the folds.
        for (chat_id, w1, w2, w3), cached in self.generator._cache3.items():
            fresh = await self.db.markov.get_transitions3(chat_id, w1, w2, w3)
            self.assertEqual(cached, fresh, (w1, w2, w3))
        for (chat_id, w1, w2), cached2 in self.generator._cache2.items():
            fresh2 = await self.db.markov.get_transitions(chat_id, w1, w2)
            self.assertEqual(cached2, fresh2, (w1, w2))
        self.assertEqual(
            self.generator._cache_starts3[CHAT_ID],
            await self.db.markov.get_starts3(CHAT_ID),
        )
        self.assertEqual(
            self.generator._cache_starts2[CHAT_ID],
            await self.db.markov.get_starts(CHAT_ID),
        )

    async def test_matcher_folded_index_equals_rebuilt(self) -> None:
        for message in CORPUS:
            await self._learn(message, incremental=False)
        matcher = self.generator._context_state_matcher
        for order in (2, 3):
            await matcher._get_index(CHAT_ID, order)
        for message in EXTRA:
            await self._learn(message, incremental=True)

        fresh_matcher = ContextStateMatcher(self.db.markov)
        for order in (2, 3):
            folded_index = matcher._cache[(CHAT_ID, order)]
            fresh_index = await fresh_matcher._get_index(CHAT_ID, order)
            self.assertEqual(folded_index.exact, fresh_index.exact)
            for key, bucket in fresh_index.casefolded.items():
                self.assertEqual(
                    folded_index.casefolded.get(key), bucket, (order, key)
                )

    async def test_generation_identical_warm_vs_cold(self) -> None:
        for message in CORPUS:
            await self._learn(message, incremental=False)
        await self._warm_caches(CORPUS + EXTRA)
        for message in EXTRA:
            await self._learn(message, incremental=True)

        cold_generator = MarkovGenerator(db=self.db.markov)
        for seed in (7, 42, 1337):
            warm_text, _ = await self.generator.generate_text_with_trace(
                CHAT_ID, 500, rng=random.Random(seed), attempt_budget=3
            )
            cold_text, _ = await cold_generator.generate_text_with_trace(
                CHAT_ID, 500, rng=random.Random(seed), attempt_budget=3
            )
            self.assertEqual(warm_text, cold_text, seed)

    async def test_kill_switch_restores_wipe_policy(self) -> None:
        for message in CORPUS:
            await self._learn(message, incremental=False)
        await self._warm_caches(CORPUS)
        self.assertTrue(self.generator._cache3)
        await self._learn(EXTRA[0], incremental=False)
        # The old policy: everything for the chat is gone.
        self.assertFalse(
            [k for k in self.generator._cache3 if k[0] == CHAT_ID]
        )
        self.assertNotIn(CHAT_ID, self.generator._cache_starts3)


class TestShadowIndexService(_Phase1DbCase):
    async def test_window_index_and_incremental_fold(self) -> None:
        for message in CORPUS:
            await self._learn(message, incremental=False)
        index = await self.learning.get_order4_shadow_index(CHAT_ID)
        self.assertIn(("коты", "любят", "тёплое", "молоко"), index)
        self.assertEqual(
            index[("коты", "любят", "тёплое", "молоко")], {"утром": 1}
        )
        # A new message folds into the cached index without a rebuild.
        await self._learn("коты любят тёплое молоко днём", incremental=True)
        folded = await self.learning.get_order4_shadow_index(CHAT_ID)
        self.assertEqual(
            folded[("коты", "любят", "тёплое", "молоко")],
            {"утром": 1, "днём": 1},
        )


if __name__ == "__main__":
    unittest.main()
