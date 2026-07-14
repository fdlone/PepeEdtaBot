from __future__ import annotations

import unittest
from collections import deque

from app.config.runtime_state import RuntimeState


def make_runtime_state(**overrides: object) -> RuntimeState:
    base: dict[str, object] = {
        "reply_probability": 0.08,
        "min_cooldown_sec": 45,
        "min_tokens_for_model": 200,
        "max_reply_chars": 280,
        "max_reply_tokens": 45,
        "normalize_lower": False,
        "auto_capitalize_replies": False,
        "typing_min_ms": 350,
        "typing_max_ms": 1100,
        "typing_per_char_ms": 12,
        "randomness_strength": 2.0,
        "candidate_selection_temperature": 0.7,
        "reply_flavor_strength": 1.0,
        "emoji_append_chance": 0.15,
        "repetition_penalty_strength": 1.0,
        "recent_reply_penalty_strength": 1.0,
        "verbatim_penalty_strength": 1.0,
        "length_mode_weights": (0.25, 0.55, 0.2),
        "length_context_adaptation": 0.0,
        "markov_order": 3,
        "enable_backoff": True,
        "markov_jump_probability": 0.04,
        "hot_ngram_seed_chance": 0.05,
        "hot_ngram_min_count": 3,
        "hot_ngram_recency_share": 0.5,
        "rare_event_chance": 0.005,
        "false_start_chance": 0.03,
        "rare_event_daily_cap": 3,
        "user_quirk_chance": 0.1,
        "user_quirk_min_interactions": 25,
        "use_reply_context": True,
        "fuzzy_context_casefold": False,
        "fuzzy_context_stem": False,
        "reply_context_max_tokens": 12,
        "reply_context_last_tokens": 3,
        "reply_context_bias": 1.8,
        "reply_context_start_bias": 2.2,
        "context_start_affinity": 3.0,
        "reply_context_only_for_replies": True,
        "reply_context_include_current_message": True,
        "pivo_recent_pool_window": 5,
        "pivo_temporal_flavor_chance": 0.5,
        "mood_enabled": True,
        "mood_modulation_strength": 1.0,
        "mood_ewma_alpha": 0.3,
        "mood_lively_rate_per_min": 12.0,
        "mood_sleepy_rate_per_min": 2.0,
        "mood_heated_intensity": 0.4,
        "mood_max_rate_per_min": 120.0,
        "reply_director_enabled": True,
        "reply_probability_min": 0.02,
        "reply_probability_max": 0.30,
        "reply_burst_boost_sec": 180,
        "reply_burst_boost_mult": 2.0,
        "reply_burst_suppress_sec": 600,
        "reply_burst_suppress_mult": 0.5,
        "reply_max_per_hour": 20,
        "mention_cooldown_sec": 5,
        "runtime_state_ttl_sec": 10,
        "runtime_state_max_chats": 2,
    }
    base.update(overrides)
    return RuntimeState(**base)


class TestRuntimeState(unittest.TestCase):
    def test_prune_inactive_removes_stale_chat_entries(self) -> None:
        state = make_runtime_state()
        state.last_reply_ts[100] = 1.0
        state.learned_messages[100] = 4
        state.recent_short_replies[100] = deque(["hi"], maxlen=5)
        state.recent_replies[100] = deque(["длинный недавний ответ"], maxlen=20)
        state.note_chat_activity(100, now=10.0)

        state.prune_inactive(now=21.0)

        self.assertNotIn(100, state.last_reply_ts)
        self.assertNotIn(100, state.learned_messages)
        self.assertNotIn(100, state.recent_short_replies)
        self.assertNotIn(100, state.recent_replies)

    def test_prune_inactive_keeps_recent_chat_entries(self) -> None:
        state = make_runtime_state()
        state.last_reply_ts[100] = 1.0
        state.learned_messages[100] = 4
        state.note_chat_activity(100, now=15.0)

        state.prune_inactive(now=20.0)

        self.assertIn(100, state.last_reply_ts)
        self.assertIn(100, state.learned_messages)

    def test_note_chat_activity_evicts_oldest_when_capacity_exceeded(self) -> None:
        state = make_runtime_state(runtime_state_max_chats=2)
        state.last_reply_ts[1] = 1.0
        state.last_reply_ts[2] = 2.0
        state.last_reply_ts[3] = 3.0

        state.note_chat_activity(1, now=1.0)
        state.note_chat_activity(2, now=2.0)
        state.note_chat_activity(3, now=3.0)
        state.prune_inactive(now=3.0)

        self.assertNotIn(1, state.last_reply_ts)
        self.assertIn(2, state.last_reply_ts)
        self.assertIn(3, state.last_reply_ts)

    def test_forget_chat_clears_all_runtime_maps(self) -> None:
        state = make_runtime_state()
        state.last_reply_ts[100] = 1.0
        state.learned_messages[100] = 4
        state.recent_short_replies[100] = deque(["hi"], maxlen=5)
        state.recent_replies[100] = deque(["длинный недавний ответ"], maxlen=20)
        state.recent_reply_times[100] = deque([1.0])
        state.note_rare_event(100, "2026-07-04")
        state.note_user_quirk(100, 1001, "2026-07-04")
        state.note_user_quirk(200, 1001, "2026-07-04")
        state.note_chat_activity(100, now=10.0)

        state.forget_chat(100)

        self.assertEqual(state.last_reply_ts, {})
        self.assertEqual(state.learned_messages, {})
        self.assertEqual(state.recent_short_replies, {})
        self.assertEqual(state.recent_replies, {})
        self.assertEqual(state.recent_reply_times, {})
        self.assertEqual(state.rare_events_today, {})
        # Only the forgotten chat's quirk stamps are swept.
        self.assertEqual(state.last_user_quirk_day, {(200, 1001): "2026-07-04"})

    def test_rare_event_cap_counts_per_day(self) -> None:
        state = make_runtime_state()
        self.assertTrue(state.can_fire_rare_event(1, "2026-07-04"))
        for _ in range(state.rare_event_daily_cap):
            state.note_rare_event(1, "2026-07-04")
        self.assertFalse(state.can_fire_rare_event(1, "2026-07-04"))
        # New day resets the counter.
        self.assertTrue(state.can_fire_rare_event(1, "2026-07-05"))
        state.note_rare_event(1, "2026-07-05")
        self.assertEqual(state.rare_events_today[1], ("2026-07-05", 1))

    def test_rare_event_zero_cap_never_fires(self) -> None:
        state = make_runtime_state(rare_event_daily_cap=0)
        self.assertFalse(state.can_fire_rare_event(1, "2026-07-04"))
        self.assertFalse(state.can_fire_rare_event(1, "2026-07-05"))

    def test_user_quirk_gate_is_per_user_per_utc_day(self) -> None:
        state = make_runtime_state()
        self.assertTrue(state.can_fire_user_quirk(1, 1001, "2026-07-13"))

        state.note_user_quirk(1, 1001, "2026-07-13")

        # Same user, same day: suppressed. Other user / other chat: allowed.
        self.assertFalse(state.can_fire_user_quirk(1, 1001, "2026-07-13"))
        self.assertTrue(state.can_fire_user_quirk(1, 2002, "2026-07-13"))
        self.assertTrue(state.can_fire_user_quirk(2, 1001, "2026-07-13"))
        # Next day: allowed again.
        self.assertTrue(state.can_fire_user_quirk(1, 1001, "2026-07-14"))

    def test_note_reply_sent_updates_last_ts_and_history(self) -> None:
        state = make_runtime_state()
        state.note_reply_sent(100, now=1000.0)
        state.note_reply_sent(100, now=1050.0)

        self.assertEqual(state.last_reply_ts[100], 1050.0)
        self.assertEqual(list(state.recent_reply_times[100]), [1000.0, 1050.0])

    def test_note_reply_sent_trims_history_older_than_one_hour(self) -> None:
        state = make_runtime_state()
        state.note_reply_sent(100, now=1000.0)
        # Advance more than an hour: the first timestamp falls out of the window.
        state.note_reply_sent(100, now=1000.0 + 3600.0 + 1.0)

        self.assertEqual(list(state.recent_reply_times[100]), [4601.0])

    def test_note_reply_sent_prompted_reply_skips_hourly_history(self) -> None:
        state = make_runtime_state()
        # A mention answer updates cooldown/burst but must not count against the
        # per-hour cap.
        state.note_reply_sent(100, now=1000.0, unprompted=False)

        self.assertEqual(state.last_reply_ts[100], 1000.0)
        self.assertNotIn(100, state.recent_reply_times)

    def test_note_reply_sent_mixes_prompted_and_unprompted(self) -> None:
        state = make_runtime_state()
        state.note_reply_sent(100, now=1000.0, unprompted=True)
        state.note_reply_sent(100, now=1050.0, unprompted=False)
        state.note_reply_sent(100, now=1100.0, unprompted=True)

        self.assertEqual(state.last_reply_ts[100], 1100.0)
        self.assertEqual(list(state.recent_reply_times[100]), [1000.0, 1100.0])


if __name__ == "__main__":
    unittest.main()
