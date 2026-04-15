from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from db import Database
from markov import (
    MarkovGenerator,
    detokenize,
    is_context_heavy_reply,
    tokenize,
    weighted_next_choice,
)
from main import bot_is_mentioned, extract_context_tokens
from text_utils import sanitize_text


class TestMarkovAndText(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_markov_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        self.generator = MarkovGenerator(self.db)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    def test_sanitize_and_tokenize(self) -> None:
        clean = sanitize_text("Привееееет!!!   https://x.y  @PepeEdta_Bot  Как   дела??")
        self.assertEqual(clean, "Привеет!! Как дела??")

        tokens = tokenize(clean)
        self.assertEqual(tokens, ["Привеет", "!", "!", "Как", "дела", "?", "?"])

    def test_detokenize(self) -> None:
        text = detokenize(["Привет", ",", "мир", "!", "Как", "дела", "?"], max_chars=100)
        self.assertEqual(text, "Привет, мир! Как дела?")

    def test_context_heavy_reply_detects_loop_on_parent_tokens(self) -> None:
        self.assertTrue(
            is_context_heavy_reply(
                generated_tokens=["кофе", "утром", "кофе", "утром", "кофе"],
                context_tokens=["Люблю", "кофе", "утром"],
            )
        )

    def test_context_heavy_reply_detects_near_copy_of_parent_context(self) -> None:
        self.assertTrue(
            is_context_heavy_reply(
                generated_tokens=["Люблю", "кофе", "утром", "всегда", "дома"],
                context_tokens=["сегодня", "Люблю", "кофе", "утром", "всегда"],
            )
        )

    def test_context_heavy_reply_allows_contextual_but_new_reply(self) -> None:
        self.assertFalse(
            is_context_heavy_reply(
                generated_tokens=["Люблю", "кофе", "но", "сегодня", "чай", "лучше"],
                context_tokens=["Люблю", "кофе", "утром"],
            )
        )

    def test_context_bias_prefers_context_tokens(self) -> None:
        random.seed(7)
        tea_count = 0
        coffee_count = 0
        for _ in range(400):
            choice = weighted_next_choice(
                items=[("кофе", 1), ("чай", 1)],
                explore_probability=0.0,
                power=1.0,
                context_token_set={"чай"},
                context_pairs={("утром", "чай")},
                context_triplets={("люблю", "утром", "чай")},
                current_state=("люблю", "утром"),
                context_bias=2.4,
                step_index=0,
            )
            if choice == "чай":
                tea_count += 1
            else:
                coffee_count += 1

        self.assertGreater(tea_count, coffee_count)

    def test_repetition_penalty_avoids_immediate_loop(self) -> None:
        random.seed(17)
        same_count = 0
        other_count = 0
        for _ in range(400):
            choice = weighted_next_choice(
                items=[("эхо", 4), ("ответ", 4)],
                explore_probability=0.0,
                power=1.0,
                current_state=("скажи", "эхо"),
                recent_tokens=["скажи", "эхо", "эхо"],
                seen_pairs={("эхо", "эхо")},
                seen_triplets={("скажи", "эхо", "эхо")},
            )
            if choice == "эхо":
                same_count += 1
            else:
                other_count += 1

        self.assertGreater(other_count, same_count)

    def test_extract_context_tokens_uses_reply_and_current_message(self) -> None:
        message = SimpleNamespace(reply_to_message=SimpleNamespace(text="Люблю кофе!!! @bot"))
        tokens = extract_context_tokens(
            message=message,
            current_text="А я утром",
            normalize_lower=False,
            max_tokens=8,
            only_for_replies=True,
            include_current_message=True,
        )
        self.assertEqual(tokens, ["Люблю", "кофе", "!", "!", "А", "я", "утром"])

    def test_extract_context_tokens_skips_non_reply_when_required(self) -> None:
        message = SimpleNamespace(reply_to_message=None)
        tokens = extract_context_tokens(
            message=message,
            current_text="случайный текст",
            normalize_lower=False,
            max_tokens=6,
            only_for_replies=True,
            include_current_message=True,
        )
        self.assertEqual(tokens, [])

    def test_bot_is_mentioned_by_plain_text_pepe(self) -> None:
        message = SimpleNamespace(text="Pepe, ответь что-нибудь", entities=None, reply_to_message=None)
        self.assertTrue(bot_is_mentioned(message, "PepeEdtaBot", 777))

    def test_bot_is_mentioned_by_plain_text_pepe_case_insensitive(self) -> None:
        message = SimpleNamespace(text="ПеПе, ответь что-нибудь", entities=None, reply_to_message=None)
        self.assertTrue(bot_is_mentioned(message, "PepeEdtaBot", 777))

    def test_bot_is_mentioned_by_plain_text_pepe_cyrillic(self) -> None:
        message = SimpleNamespace(text="пЕпЕ ты тут?", entities=None, reply_to_message=None)
        self.assertTrue(bot_is_mentioned(message, "PepeEdtaBot", 777))

    def test_bot_is_not_mentioned_by_substring_only(self) -> None:
        message = SimpleNamespace(text="pepega сегодня победил", entities=None, reply_to_message=None)
        self.assertFalse(bot_is_mentioned(message, "PepeEdtaBot", 777))

    def test_bot_is_mentioned_by_reply_to_bot_message(self) -> None:
        reply_to_message = SimpleNamespace(from_user=SimpleNamespace(id=777))
        message = SimpleNamespace(text="слушай", entities=None, reply_to_message=reply_to_message)
        self.assertTrue(bot_is_mentioned(message, "PepeEdtaBot", 777))

    async def test_generate_text_with_seed(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=4444,
            author_id=1,
            raw_text="Я очень люблю питон",
            tokens=["Я", "очень", "люблю", "питон"],
        )
        await self.db.save_message_and_update_model(
            chat_id=4444,
            author_id=2,
            raw_text="Я очень люблю кофе",
            tokens=["Я", "очень", "люблю", "кофе"],
        )
        await self.db.save_message_and_update_model(
            chat_id=4444,
            author_id=3,
            raw_text="Люблю кофе утром",
            tokens=["Люблю", "кофе", "утром"],
        )

        random.seed(42)
        text = await self.generator.generate_text(
            chat_id=4444,
            max_chars=12,
            seed_tokens=["Я", "очень", "люблю"],
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
        )
        self.assertTrue(text)
        self.assertTrue(text.startswith("Я очень"))
        self.assertGreaterEqual(len(text), 5)

    async def test_generate_text_uses_context_windows_for_start(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=5555,
            author_id=1,
            raw_text="Люблю кофе утром всегда",
            tokens=["Люблю", "кофе", "утром", "всегда"],
        )
        await self.db.save_message_and_update_model(
            chat_id=5555,
            author_id=2,
            raw_text="Люблю кофе вечером иногда",
            tokens=["Люблю", "кофе", "вечером", "иногда"],
        )

        random.seed(11)
        text = await self.generator.generate_text(
            chat_id=5555,
            max_chars=18,
            context_tokens=["сегодня", "Люблю", "кофе", "утром"],
            context_bias=1.8,
            context_start_bias=2.2,
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
        )

        self.assertTrue(text)
        self.assertTrue(text.startswith("Люблю кофе утром"))

    async def test_generate_text_falls_back_when_context_not_found(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=6666,
            author_id=1,
            raw_text="кошка любит солнце ярко",
            tokens=["кошка", "любит", "солнце", "ярко"],
        )
        await self.db.save_message_and_update_model(
            chat_id=6666,
            author_id=2,
            raw_text="кошка любит дождь тихо",
            tokens=["кошка", "любит", "дождь", "тихо"],
        )
        await self.db.save_message_and_update_model(
            chat_id=6666,
            author_id=3,
            raw_text="солнце ярко греет дом",
            tokens=["солнце", "ярко", "греет", "дом"],
        )

        random.seed(5)
        text = await self.generator.generate_text(
            chat_id=6666,
            max_chars=17,
            context_tokens=["совсем", "другой", "контекст"],
            context_bias=2.4,
            context_start_bias=2.6,
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
        )

        self.assertTrue(text)
        self.assertFalse(text.startswith("совсем другой"))
        self.assertIn(text.split()[0], {"кошка", "солнце"})

    async def test_generate_text_without_context_matches_legacy_path(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=7777,
            author_id=1,
            raw_text="утром люблю чай дома",
            tokens=["утром", "люблю", "чай", "дома"],
        )
        await self.db.save_message_and_update_model(
            chat_id=7777,
            author_id=2,
            raw_text="утром люблю кофе дома",
            tokens=["утром", "люблю", "кофе", "дома"],
        )

        random.seed(13)
        text = await self.generator.generate_text(
            chat_id=7777,
            max_chars=15,
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
        )

        self.assertTrue(text)
        self.assertTrue(text.startswith("утром люблю"))

    def test_context_bias_does_not_override_repetition_penalty(self) -> None:
        random.seed(23)
        repeated_count = 0
        fresh_count = 0
        for _ in range(400):
            choice = weighted_next_choice(
                items=[("кофе", 3), ("бодрит", 3)],
                explore_probability=0.0,
                power=1.0,
                context_token_set={"кофе"},
                context_pairs={("утром", "кофе")},
                context_triplets={("люблю", "утром", "кофе")},
                current_state=("люблю", "утром"),
                context_bias=3.0,
                step_index=0,
                recent_tokens=["люблю", "утром", "кофе", "кофе"],
                seen_pairs={("кофе", "кофе")},
                seen_triplets={("утром", "кофе", "кофе")},
            )
            if choice == "кофе":
                repeated_count += 1
            else:
                fresh_count += 1

        self.assertGreater(fresh_count, repeated_count)

    def test_repetition_penalty_strength_zero_disables_loop_penalty(self) -> None:
        random.seed(29)
        repeated_count = 0
        fresh_count = 0
        for _ in range(400):
            choice = weighted_next_choice(
                items=[("эхо", 4), ("ответ", 4)],
                explore_probability=0.0,
                power=1.0,
                current_state=("скажи", "эхо"),
                recent_tokens=["скажи", "эхо", "эхо"],
                seen_pairs={("эхо", "эхо")},
                seen_triplets={("скажи", "эхо", "эхо")},
                repetition_penalty_strength=0.0,
            )
            if choice == "эхо":
                repeated_count += 1
            else:
                fresh_count += 1

        self.assertGreater(repeated_count, 0)
        self.assertGreater(fresh_count, 0)
