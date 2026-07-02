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
        "repetition_penalty_strength": 1.0,
        "recent_reply_penalty_strength": 1.0,
        "markov_order": 3,
        "enable_backoff": True,
        "backoff_min_order": 1,
        "use_reply_context": True,
        "fuzzy_context_casefold": False,
        "fuzzy_context_prefix": False,
        "reply_context_max_tokens": 12,
        "reply_context_last_tokens": 3,
        "reply_context_bias": 1.8,
        "reply_context_start_bias": 2.2,
        "reply_context_only_for_replies": True,
        "reply_context_include_current_message": True,
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
        state.note_chat_activity(100, now=10.0)

        state.forget_chat(100)

        self.assertEqual(state.last_reply_ts, {})
        self.assertEqual(state.learned_messages, {})
        self.assertEqual(state.recent_short_replies, {})
        self.assertEqual(state.recent_replies, {})


if __name__ == "__main__":
    unittest.main()
