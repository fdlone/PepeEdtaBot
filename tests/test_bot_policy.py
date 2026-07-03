from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.core.reply_policy import (
    bot_is_mentioned,
    burst_factor,
    conversation_momentum,
    cooldown_allows_reply,
    effective_reply_probability,
    has_enough_model_data,
    should_reply_to_message,
    within_hourly_cap,
)


class TestBotPolicy(unittest.TestCase):
    def test_should_reply_when_mentioned_ignores_cooldown_and_probability(self) -> None:
        self.assertTrue(
            should_reply_to_message(
                mentioned=True,
                cooldown_ok=False,
                hourly_cap_ok=False,
                reply_probability=0.0,
                random_value=0.99,
            )
        )

    def test_should_reply_by_probability_when_cooldown_allows(self) -> None:
        self.assertTrue(
            should_reply_to_message(
                mentioned=False,
                cooldown_ok=True,
                hourly_cap_ok=True,
                reply_probability=0.25,
                random_value=0.10,
            )
        )

    def test_should_not_reply_by_probability_during_cooldown(self) -> None:
        self.assertFalse(
            should_reply_to_message(
                mentioned=False,
                cooldown_ok=False,
                hourly_cap_ok=True,
                reply_probability=1.0,
                random_value=0.0,
            )
        )

    def test_should_not_reply_when_hourly_cap_reached(self) -> None:
        self.assertFalse(
            should_reply_to_message(
                mentioned=False,
                cooldown_ok=True,
                hourly_cap_ok=False,
                reply_probability=1.0,
                random_value=0.0,
            )
        )

    def test_cooldown_allows_reply_after_min_interval(self) -> None:
        self.assertTrue(
            cooldown_allows_reply(
                now_ts=145.0, last_reply_ts=100.0, min_cooldown_sec=45
            )
        )
        self.assertFalse(
            cooldown_allows_reply(
                now_ts=144.9, last_reply_ts=100.0, min_cooldown_sec=45
            )
        )

    def test_has_enough_model_data_at_threshold(self) -> None:
        self.assertTrue(
            has_enough_model_data(token_volume=200, min_tokens_for_model=200)
        )
        self.assertFalse(
            has_enough_model_data(token_volume=199, min_tokens_for_model=200)
        )

    def test_bot_is_mentioned_by_entity_like_value(self) -> None:
        message = SimpleNamespace(
            text="hello @PepeEdtaBot",
            entities=[SimpleNamespace(type="mention", offset=6, length=12)],
            reply_to_message=None,
        )

        self.assertTrue(bot_is_mentioned(message, "PepeEdtaBot", 777))

    def test_bot_is_mentioned_by_entity_like_enum_name(self) -> None:
        entity_type = SimpleNamespace(name="MENTION", value=None)
        message = SimpleNamespace(
            text="hello @PepeEdtaBot",
            entities=[SimpleNamespace(type=entity_type, offset=6, length=12)],
            reply_to_message=None,
        )

        self.assertTrue(bot_is_mentioned(message, "PepeEdtaBot", 777))


class TestConversationMomentum(unittest.TestCase):
    def test_quiet_unaddressed_chat_scores_low(self) -> None:
        score = conversation_momentum(
            rate_ewma=0.5,
            mention_ewma=0.0,
            is_reply=False,
            lively_rate_per_min=12.0,
        )
        self.assertLess(score, 0.1)

    def test_busy_addressed_reply_thread_saturates(self) -> None:
        score = conversation_momentum(
            rate_ewma=100.0,  # well above lively → rate term saturates
            mention_ewma=1.0,
            is_reply=True,
            lively_rate_per_min=12.0,
        )
        self.assertEqual(score, 1.0)

    def test_rate_is_normalised_against_lively_threshold(self) -> None:
        # Exactly at the lively threshold, rate term is 1.0 → 0.55 weight.
        score = conversation_momentum(
            rate_ewma=12.0,
            mention_ewma=0.0,
            is_reply=False,
            lively_rate_per_min=12.0,
        )
        self.assertAlmostEqual(score, 0.55)

    def test_reply_thread_adds_chain_weight(self) -> None:
        without = conversation_momentum(
            rate_ewma=6.0, mention_ewma=0.0, is_reply=False, lively_rate_per_min=12.0
        )
        with_reply = conversation_momentum(
            rate_ewma=6.0, mention_ewma=0.0, is_reply=True, lively_rate_per_min=12.0
        )
        self.assertAlmostEqual(with_reply - without, 0.15)

    def test_result_bounded_to_unit_interval(self) -> None:
        score = conversation_momentum(
            rate_ewma=-5.0, mention_ewma=-1.0, is_reply=False, lively_rate_per_min=12.0
        )
        self.assertGreaterEqual(score, 0.0)


class TestBurstFactor(unittest.TestCase):
    KW = dict(
        boost_window_sec=180.0,
        boost_mult=2.0,
        suppress_window_sec=600.0,
        suppress_mult=0.5,
    )

    def test_boost_window_multiplies_up(self) -> None:
        self.assertEqual(burst_factor(seconds_since_reply=60.0, **self.KW), 2.0)

    def test_suppress_window_dampens(self) -> None:
        self.assertEqual(burst_factor(seconds_since_reply=300.0, **self.KW), 0.5)

    def test_after_both_windows_is_neutral(self) -> None:
        self.assertEqual(burst_factor(seconds_since_reply=1000.0, **self.KW), 1.0)

    def test_never_replied_is_neutral(self) -> None:
        # A fresh chat: seconds_since_reply is huge (now - 0.0).
        self.assertEqual(burst_factor(seconds_since_reply=10**9, **self.KW), 1.0)


class TestEffectiveReplyProbability(unittest.TestCase):
    def test_zero_momentum_sits_at_min(self) -> None:
        prob = effective_reply_probability(
            base_min=0.02, base_max=0.30, momentum=0.0, mood_mult=1.0, burst_mult=1.0
        )
        self.assertAlmostEqual(prob, 0.02)

    def test_full_momentum_sits_at_max(self) -> None:
        prob = effective_reply_probability(
            base_min=0.02, base_max=0.30, momentum=1.0, mood_mult=1.0, burst_mult=1.0
        )
        self.assertAlmostEqual(prob, 0.30)

    def test_mood_and_burst_multiply_the_base(self) -> None:
        prob = effective_reply_probability(
            base_min=0.0, base_max=0.20, momentum=0.5, mood_mult=1.5, burst_mult=2.0
        )
        self.assertAlmostEqual(prob, 0.10 * 1.5 * 2.0)

    def test_clamped_to_one(self) -> None:
        prob = effective_reply_probability(
            base_min=0.5, base_max=1.0, momentum=1.0, mood_mult=3.0, burst_mult=2.0
        )
        self.assertEqual(prob, 1.0)

    def test_never_negative(self) -> None:
        prob = effective_reply_probability(
            base_min=0.02, base_max=0.30, momentum=0.5, mood_mult=0.0, burst_mult=1.0
        )
        self.assertEqual(prob, 0.0)


class TestWithinHourlyCap(unittest.TestCase):
    def test_under_cap_allows(self) -> None:
        self.assertTrue(within_hourly_cap([100.0, 200.0], now=300.0, max_per_hour=20))

    def test_at_cap_blocks(self) -> None:
        times = [float(t) for t in range(3)]  # three replies
        self.assertFalse(within_hourly_cap(times, now=10.0, max_per_hour=3))

    def test_old_replies_outside_window_do_not_count(self) -> None:
        times = [0.0, 1.0, 2.0]  # all older than one hour at now=5000
        self.assertTrue(within_hourly_cap(times, now=5000.0, max_per_hour=1))

    def test_zero_cap_is_unlimited(self) -> None:
        times = [float(t) for t in range(100)]
        self.assertTrue(within_hourly_cap(times, now=1.0, max_per_hour=0))


if __name__ == "__main__":
    unittest.main()
