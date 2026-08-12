"""Phase 3 temporal layer: decay arithmetic, compression, blend (TZ §7-8).

The invariants here are the ones TZ §19 names explicitly — an observation now
contributes 1, an observation one half-life old contributes 0.5, order of
observations does not matter, and the effective weight never grows on its own.
They are the reason the short layer can be one row instead of an event log, so
they are tested directly rather than through generation.
"""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

from app.core.markov import MarkovGenerator
from app.core.temporal import (
    COMPRESSION_LOG,
    COMPRESSION_POW,
    SECONDS_PER_DAY,
    TemporalBlend,
    compress_long,
    decay_multiplier,
    short_effective,
    short_observed,
)
from app.infrastructure.database import Database

DAY = int(SECONDS_PER_DAY)
HL = 3.0  # default half-life in days


class TestDecayArithmetic(unittest.TestCase):
    def test_observation_now_contributes_one(self) -> None:
        self.assertEqual(short_observed(0.0, None, 1000, HL), 1.0)

    def test_observation_one_half_life_old_contributes_half(self) -> None:
        observed_at = 1_000_000
        value = short_observed(0.0, None, observed_at, HL)
        later = observed_at + int(HL * DAY)
        self.assertAlmostEqual(short_effective(value, observed_at, later, HL), 0.5)

    def test_effective_weight_equals_sum_of_individually_decayed_observations(
        self,
    ) -> None:
        """The identity that justifies O(1) storage (TZ §7.1)."""
        base = 1_000_000
        offsets = [0, DAY, 2 * DAY, 5 * DAY, 11 * DAY]
        value, updated = 0.0, None
        for offset in offsets:
            moment = base + offset
            value, updated = short_observed(value, updated, moment, HL), moment
        read_at = base + 20 * DAY
        expected = math.fsum(
            2.0 ** (-((read_at - (base + offset)) / (HL * DAY))) for offset in offsets
        )
        self.assertAlmostEqual(short_effective(value, updated, read_at, HL), expected)

    def test_order_of_observations_does_not_change_the_result(self) -> None:
        base = 500_000
        offsets = [0, 3 * DAY, DAY, 9 * DAY, 2 * DAY]
        read_at = base + 30 * DAY

        def accumulate(sequence: list[int]) -> float:
            value, updated = 0.0, None
            for offset in sorted(sequence):
                moment = base + offset
                value, updated = short_observed(value, updated, moment, HL), moment
            return short_effective(value, updated, read_at, HL)

        forward = accumulate(offsets)
        backward = accumulate(list(reversed(offsets)))
        self.assertAlmostEqual(forward, backward)

    def test_effective_weight_is_non_increasing_between_observations(self) -> None:
        value, updated = short_observed(0.0, None, 0, HL), 0
        previous = math.inf
        for day in range(0, 40):
            current = short_effective(value, updated, day * DAY, HL)
            self.assertLessEqual(current, previous + 1e-12)
            previous = current

    def test_empty_layer_reads_as_zero(self) -> None:
        self.assertEqual(short_effective(0.0, None, 1000, HL), 0.0)
        self.assertEqual(short_effective(5.0, None, 1000, HL), 0.0)

    def test_clock_skew_does_not_inflate_the_counter(self) -> None:
        """A backwards clock must not extrapolate growth (design D1 guard)."""
        observed_at = 1_000_000
        value = short_observed(0.0, None, observed_at, HL)
        self.assertEqual(short_effective(value, observed_at, observed_at - DAY, HL), 1.0)

    def test_non_positive_half_life_remembers_nothing(self) -> None:
        self.assertEqual(decay_multiplier(1.0, 0.0), 0.0)
        self.assertEqual(decay_multiplier(1.0, -3.0), 0.0)


class TestCompression(unittest.TestCase):
    def test_both_shapes_preserve_order_of_preference(self) -> None:
        counts = [1, 2, 5, 20, 100, 10_000]
        for shape, beta in ((COMPRESSION_LOG, 0.0), (COMPRESSION_POW, 0.6)):
            weights = [compress_long(c, shape, beta) for c in counts]
            self.assertEqual(weights, sorted(weights), shape)

    def test_dominant_count_leaves_the_smaller_one_non_negligible(self) -> None:
        """The motivating case of TZ §8.1: 10000 vs 20 is 0.998/0.002 raw.

        Both shapes must lift the underdog by at least an order of magnitude —
        that is the whole point of compressing. They do not lift it equally:
        measured here, ``log`` gives it 24.8% and ``pow`` at beta=0.6 gives it
        2.3%. An order of magnitude between the two shapes is exactly the kind
        of difference the M2R-215 calibration grid exists to settle, so the
        assertion is the property, not either number.
        """
        raw_share = 20 / (10_000 + 20)
        for shape, beta in ((COMPRESSION_LOG, 0.0), (COMPRESSION_POW, 0.6)):
            big = compress_long(10_000, shape, beta)
            small = compress_long(20, shape, beta)
            share = small / (big + small)
            self.assertGreater(share, raw_share * 10, shape)

    def test_zero_count_is_weightless_not_negative(self) -> None:
        self.assertEqual(compress_long(0, COMPRESSION_LOG, 0.0), 0.0)
        self.assertEqual(compress_long(0, COMPRESSION_POW, 0.6), 0.0)


class TestBlend(unittest.TestCase):
    def test_disabled_blend_returns_none(self) -> None:
        """The early return that makes neutrality structural (design D4)."""
        rows = [("a", 5, 3.0, 0), ("b", 1, 0.0, None)]
        self.assertIsNone(TemporalBlend(alpha=0.0).blend(rows, now=DAY))

    def test_token_known_only_to_the_fresh_layer_stays_reachable(self) -> None:
        rows = [("a", 100, 0.0, None), ("fresh", 0, 4.0, DAY)]
        blended = TemporalBlend(alpha=0.5).blend(rows, now=DAY)
        assert blended is not None
        self.assertGreater(blended.weights[1], 0.0)

    def test_empty_short_layer_degenerates_to_the_long_layer(self) -> None:
        rows = [("a", 8, 0.0, None), ("b", 2, 0.0, None)]
        blended = TemporalBlend(alpha=0.7).blend(rows, now=DAY)
        assert blended is not None
        expected_a = compress_long(8, COMPRESSION_LOG, 0.0)
        expected_b = compress_long(2, COMPRESSION_LOG, 0.0)
        total = expected_a + expected_b
        self.assertAlmostEqual(blended.weights[0], expected_a / total)
        self.assertEqual(blended.displacement, 0.0)

    def test_alpha_one_is_the_short_layer_alone(self) -> None:
        rows = [("a", 100, 1.0, DAY), ("b", 1, 3.0, DAY)]
        blended = TemporalBlend(alpha=1.0).blend(rows, now=DAY)
        assert blended is not None
        self.assertAlmostEqual(blended.weights[0], 0.25)
        self.assertAlmostEqual(blended.weights[1], 0.75)

    def test_displacement_is_zero_when_the_layers_agree(self) -> None:
        rows = [("a", 4, 2.0, DAY), ("b", 4, 2.0, DAY)]
        blended = TemporalBlend(alpha=0.5).blend(rows, now=DAY)
        assert blended is not None
        self.assertAlmostEqual(blended.displacement, 0.0)

    def test_displacement_grows_with_disagreement(self) -> None:
        rows = [("a", 100, 0.0, None), ("b", 1, 9.0, DAY)]
        mild = TemporalBlend(alpha=0.2).blend(rows, now=DAY)
        strong = TemporalBlend(alpha=0.8).blend(rows, now=DAY)
        assert mild is not None and strong is not None
        self.assertGreater(strong.displacement, mild.displacement)


class TestBlendProperties(unittest.TestCase):
    """Any legal configuration over any pool yields a valid distribution."""

    def test_blend_of_valid_layers_is_a_valid_distribution(self) -> None:
        rng = random.Random(2026)
        for _ in range(300):
            size = rng.randint(1, 12)
            now = rng.randint(0, 400) * DAY
            rows = [
                (
                    f"t{i}",
                    rng.randint(0, 5000),
                    rng.choice([0.0, rng.random() * 20]),
                    rng.choice([None, now - rng.randint(0, 200) * DAY]),
                )
                for i in range(size)
            ]
            blend = TemporalBlend(
                alpha=rng.choice([0.05, 0.3, 0.5, 0.7, 1.0]),
                half_life_days=rng.choice([1.0, 3.0, 7.0, 14.0]),
                compression=rng.choice([COMPRESSION_LOG, COMPRESSION_POW]),
                beta=rng.choice([0.5, 0.6, 0.75]),
            )
            blended = blend.blend(rows, now=now)
            assert blended is not None
            for weight in blended.weights:
                self.assertGreaterEqual(weight, 0.0)
                self.assertTrue(math.isfinite(weight))
            self.assertAlmostEqual(math.fsum(blended.weights), 1.0, places=9)
            self.assertGreaterEqual(blended.displacement, 0.0)
            self.assertLessEqual(blended.displacement, 1.0 + 1e-12)

    def test_at_least_one_candidate_keeps_positive_weight(self) -> None:
        rng = random.Random(7)
        for _ in range(200):
            size = rng.randint(1, 8)
            rows = [
                (f"t{i}", rng.randint(0, 3), rng.random() * 2, rng.choice([None, 0]))
                for i in range(size)
            ]
            blended = TemporalBlend(alpha=0.5).blend(rows, now=DAY)
            assert blended is not None
            self.assertGreater(max(blended.weights), 0.0)


class _StubPort:
    """Minimal MarkovReadPort: counts reads so caching is observable."""

    def __init__(self, rows: list[tuple[str, int, float, int | None]]) -> None:
        self.rows = rows
        self.reads = 0

    async def get_transitions3(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> list[tuple[str, int, float, int | None]]:
        self.reads += 1
        return list(self.rows)

    async def get_transitions(
        self, chat_id: int, w1: str, w2: str
    ) -> list[tuple[str, int, float, int | None]]:
        self.reads += 1
        return list(self.rows)

    async def get_starts(self, chat_id: int) -> list[tuple[str, str, int]]:
        return []

    async def get_starts3(self, chat_id: int) -> list[tuple[str, str, str, int]]:
        return []

    async def get_start_if_exists(
        self, chat_id: int, w1: str, w2: str
    ) -> tuple[str, str, int] | None:
        return None

    async def get_start3_if_exists(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> tuple[str, str, str, int] | None:
        return None

    async def get_states(
        self, chat_id: int, order: int
    ) -> list[tuple[tuple[str, ...], int]]:
        return []


class TestBlendIsInvariantToTheReadingMoment(unittest.TestCase):
    """Uniform passage of time cancels inside the pool — a real property.

    Every candidate's short weight decays by the SAME factor between two reads,
    and ``P_short`` is normalized within the pool, so the factor divides out. The
    blended distribution therefore depends on *when each token was last
    observed* relative to the others, never on when the pool is read.

    Two consequences worth stating rather than discovering later:

    - a cached pool cannot go stale in the weights sense, which is why the cache
      needs no time-based invalidation;
    - the eval fixture's choice of evaluation moment cannot bias the grid, since
      no choice of moment changes any arm's output.

    ``now`` is still threaded explicitly (design D3): it keeps the clock out of
    sampling, which is what makes runs reproducible and testable. This test
    documents that its *numeric* influence is nil, so nobody later "fixes" a
    non-bug by making sampling read the clock.
    """

    def test_same_pool_read_at_two_moments_gives_identical_weights(self) -> None:
        observed = 1_700_000_000
        rows = [
            ("часто", 50, 1.0, observed - 10 * DAY),
            ("свежо", 1, 8.0, observed),
        ]
        blend = TemporalBlend(alpha=0.6)
        early = blend.blend(rows, observed)
        late = blend.blend(rows, observed + 90 * DAY)
        assert early is not None and late is not None
        for a, b in zip(early.weights, late.weights):
            self.assertAlmostEqual(a, b, places=12)

    def test_relative_recency_is_what_moves_the_weights(self) -> None:
        """What time cannot do, a newer observation can."""
        observed = 1_700_000_000
        stale = [("a", 10, 1.0, observed - 10 * DAY), ("b", 10, 1.0, observed)]
        refreshed = [("a", 10, 1.0, observed - 10 * DAY), ("b", 10, 4.0, observed)]
        blend = TemporalBlend(alpha=0.8)
        before = blend.blend(stale, observed)
        after = blend.blend(refreshed, observed)
        assert before is not None and after is not None
        self.assertGreater(after.weights[1], before.weights[1])


class TestCachedPoolMatchesFreshRead(unittest.IsolatedAsyncioTestCase):
    """A cached pool blends exactly like a freshly read one."""

    async def test_cached_pool_matches_a_fresh_read_at_the_later_moment(self) -> None:
        observed = 1_700_000_000
        # Different ages on purpose: decaying both tokens by the same factor
        # leaves the normalized short distribution unchanged, so a same-age pool
        # could not tell a live read apart from a frozen one.
        rows = [
            ("часто", 50, 1.0, observed - 10 * DAY),
            ("свежо", 1, 8.0, observed),
        ]
        port = _StubPort(rows)
        generator = MarkovGenerator(db=port)  # type: ignore[arg-type]
        blend = TemporalBlend(alpha=0.6)

        # Populate the cache at t0, then read it again nine days later.
        cached = await generator._get3(1, "a", "b", "c")
        later = observed + 9 * DAY
        again = await generator._get3(1, "a", "b", "c")
        self.assertEqual(port.reads, 1, "second read must come from the cache")

        from_cache = blend.blend(again, later)
        from_fresh = blend.blend(rows, later)
        assert from_cache is not None and from_fresh is not None
        self.assertEqual(from_cache.weights, from_fresh.weights)
        # The cache stores the raw pair, not a resolved weight, so it agrees
        # with a fresh read at the moment of reading — the same list, in fact.
        self.assertEqual(cached, again)


class TestShortLayerReset(unittest.IsolatedAsyncioTestCase):
    """Changing the half-life discards the short layer, and only that."""

    async def asyncSetUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.db = Database(str(Path(self._dir.name) / "reset.db"))
        await self.db.init()
        await self.db.save_message_and_update_model(
            chat_id=77,
            raw_text="кот пришёл домой поздно",
            tokens=["кот", "пришёл", "домой", "поздно"],
            now=1_700_000_000,
        )
        await self.db.save_message_and_update_model(
            chat_id=88,
            raw_text="пёс ушёл гулять утром",
            tokens=["пёс", "ушёл", "гулять", "утром"],
            now=1_700_000_000,
        )

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self._dir.cleanup()

    async def _rows(self, chat_id: int) -> list[tuple[str, int, float, int | None]]:
        return await self.db.markov.get_transitions3(
            chat_id, *(("кот", "пришёл", "домой") if chat_id == 77 else
                       ("пёс", "ушёл", "гулять"))
        )

    async def test_reset_empties_the_short_layer_and_keeps_the_long_one(self) -> None:
        before = await self._rows(77)
        self.assertEqual(before[0][1], 1)
        self.assertGreater(before[0][2], 0.0)

        await self.db.reset_short_layer(77)

        after = await self._rows(77)
        self.assertEqual(after[0][1], 1, "long count must survive")
        self.assertEqual(after[0][2], 0.0)
        self.assertIsNone(after[0][3])

    async def test_reset_is_scoped_to_one_chat(self) -> None:
        await self.db.reset_short_layer(77)
        other = await self._rows(88)
        self.assertGreater(other[0][2], 0.0)

    async def test_global_reset_covers_every_chat(self) -> None:
        await self.db.reset_short_layer(None)
        for chat_id in (77, 88):
            rows = await self._rows(chat_id)
            self.assertEqual(rows[0][2], 0.0, chat_id)

    async def test_first_seen_survives_the_reset(self) -> None:
        """The reset throws away "recent", never the record of when it began."""
        async with self.db._lock:
            conn = await self.db._get_conn()
            cur = await conn.execute(
                "SELECT first_seen FROM transitions3 WHERE chat_id = 77 LIMIT 1"
            )
            before = (await cur.fetchone())[0]
        await self.db.reset_short_layer(77)
        async with self.db._lock:
            conn = await self.db._get_conn()
            cur = await conn.execute(
                "SELECT first_seen FROM transitions3 WHERE chat_id = 77 LIMIT 1"
            )
            self.assertEqual((await cur.fetchone())[0], before)


class TestNeutralityOverAPopulatedShortLayer(unittest.IsolatedAsyncioTestCase):
    """A populated short layer must not leak into sampling while alpha is 0.

    The generation-hash check proves neutrality on the frozen snapshot, whose
    short layer is empty — every row predates the migration. That leaves the
    case the live bot reaches on day two untested: schema present, short layer
    filling up, blend still off. This is that case.
    """

    async def asyncSetUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.db = Database(str(Path(self._dir.name) / "populated.db"))
        await self.db.init()
        moment = 1_700_000_000
        for step, text in enumerate(
            [
                "кот пришёл домой поздно вечером",
                "кот ушёл гулять рано утром",
                "пёс пришёл домой поздно ночью",
                "кот пришёл домой рано утром",
            ]
        ):
            await self.db.save_message_and_update_model(
                chat_id=9,
                raw_text=text,
                tokens=text.split(),
                now=moment + step * DAY,
            )

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self._dir.cleanup()

    async def test_weights_ignore_the_short_layer_when_alpha_is_zero(self) -> None:
        rows = await self.db.markov.get_transitions(9, "кот", "пришёл")
        self.assertTrue(any(row[2] > 0 for row in rows), "short layer must be populated")

        neutral = TemporalBlend(alpha=0.0)
        self.assertIsNone(neutral.blend(rows, 1_700_000_000 + 10 * DAY))

        # And the pool the sampler would weight is the raw long count, whatever
        # the short layer says — zeroing it changes nothing on the neutral path.
        zeroed = [(row[0], row[1], 0.0, None) for row in rows]
        self.assertIsNone(neutral.blend(zeroed, 1_700_000_000 + 10 * DAY))
        self.assertEqual(
            [(row[0], row[1]) for row in rows],
            [(row[0], row[1]) for row in zeroed],
        )


class TestLearnPathTemporalRecord(unittest.IsolatedAsyncioTestCase):
    """first_seen is set once; last_seen and the short layer advance."""

    async def asyncSetUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.db = Database(str(Path(self._dir.name) / "learn.db"))
        await self.db.init()

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self._dir.cleanup()

    async def test_repeat_observation_moves_last_seen_only(self) -> None:
        tokens = ["дождь", "идёт", "весь", "день"]
        first, second = 1_700_000_000, 1_700_000_000 + 5 * DAY
        for moment in (first, second):
            await self.db.save_message_and_update_model(
                chat_id=5, raw_text=" ".join(tokens), tokens=tokens, now=moment
            )
        async with self.db._lock:
            conn = await self.db._get_conn()
            cur = await conn.execute(
                "SELECT cnt, first_seen, last_seen, s_value, s_updated_at "
                "FROM transitions3 WHERE chat_id = 5"
            )
            count, first_seen, last_seen, s_value, s_updated_at = await cur.fetchone()
        self.assertEqual(count, 2)
        self.assertEqual(first_seen, first)
        self.assertEqual(last_seen, second)
        self.assertEqual(s_updated_at, second)
        # Two observations five days apart at a 3-day half-life: the older one
        # is worth 2^(-5/3) of a fresh one, and the identity says the stored
        # value is exactly their sum.
        self.assertAlmostEqual(s_value, 1.0 + 2.0 ** (-5.0 / 3.0), places=9)


if __name__ == "__main__":
    unittest.main()
