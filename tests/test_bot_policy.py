from __future__ import annotations

import unittest
from types import SimpleNamespace

from bot_policy import (
    bot_is_mentioned,
    cooldown_allows_reply,
    has_enough_model_data,
    should_reply_to_message,
)


class TestBotPolicy(unittest.TestCase):
    def test_should_reply_when_mentioned_ignores_cooldown_and_probability(self) -> None:
        self.assertTrue(
            should_reply_to_message(
                mentioned=True,
                cooldown_ok=False,
                reply_probability=0.0,
                random_value=0.99,
            )
        )

    def test_should_reply_by_probability_when_cooldown_allows(self) -> None:
        self.assertTrue(
            should_reply_to_message(
                mentioned=False,
                cooldown_ok=True,
                reply_probability=0.25,
                random_value=0.10,
            )
        )

    def test_should_not_reply_by_probability_during_cooldown(self) -> None:
        self.assertFalse(
            should_reply_to_message(
                mentioned=False,
                cooldown_ok=False,
                reply_probability=1.0,
                random_value=0.0,
            )
        )

    def test_cooldown_allows_reply_after_min_interval(self) -> None:
        self.assertTrue(cooldown_allows_reply(now_ts=145.0, last_reply_ts=100.0, min_cooldown_sec=45))
        self.assertFalse(cooldown_allows_reply(now_ts=144.9, last_reply_ts=100.0, min_cooldown_sec=45))

    def test_has_enough_model_data_at_threshold(self) -> None:
        self.assertTrue(has_enough_model_data(token_volume=200, min_tokens_for_model=200))
        self.assertFalse(has_enough_model_data(token_volume=199, min_tokens_for_model=200))

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


if __name__ == "__main__":
    unittest.main()
