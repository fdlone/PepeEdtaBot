from __future__ import annotations

import random
import unittest
import uuid
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.handlers.learning import extract_context_tokens
from bot_policy import bot_is_mentioned
from db import Database
from markov import (
    GenerationTrace,
    MarkovGenerator,
    _GenerationAttempt,
    detokenize,
    escalated_randomness_strength,
    has_degraded_recent_window,
    is_context_heavy_reply,
    is_low_diversity_reply,
    is_short_generated_reply,
    remember_bounded,
    tokenize,
    trim_repetitive_tail,
    weighted_next_choice,
    weighted_start3_choice,
)
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
        clean = sanitize_text(
            "Привееееет!!!   https://x.y  @PepeEdta_Bot  Как   дела??"
        )
        self.assertEqual(clean, "Привеет!! Как дела??")

        tokens = tokenize(clean)
        self.assertEqual(tokens, ["Привеет", "!", "!", "Как", "дела", "?", "?"])

    def test_sanitize_preserves_email_host(self) -> None:
        # Regression: `@host` inside an email must NOT be stripped as a mention.
        clean = sanitize_text("Напиши на user@example.com если что")
        self.assertEqual(clean, "Напиши на user@example.com если что")

    def test_sanitize_still_strips_leading_mentions(self) -> None:
        clean = sanitize_text("@PepeEdta_Bot привет")
        self.assertEqual(clean, "привет")

    def test_detokenize(self) -> None:
        text = detokenize(
            ["Привет", ",", "мир", "!", "Как", "дела", "?"], max_chars=100
        )
        self.assertEqual(text, "Привет, мир! Как дела?")

    def test_detokenize_does_not_cut_words_by_char_limit(self) -> None:
        text = detokenize(["Очень", "длинноеслово", "дальше"], max_chars=12)
        self.assertEqual(text, "Очень")

    def test_escalated_randomness_strength_grows_monotonically(self) -> None:
        values = [
            escalated_randomness_strength(0.6, attempt_index=i, total_attempts=5)
            for i in range(5)
        ]

        self.assertEqual(values[0], 0.6)
        self.assertGreater(values[-1], values[0])
        self.assertLessEqual(values[-1], 3.0)
        self.assertEqual(values, sorted(values))

    def test_remember_bounded_evicts_oldest_entry(self) -> None:
        values: OrderedDict[str, None] = OrderedDict()

        remember_bounded(values, "first", 2)
        remember_bounded(values, "second", 2)
        remember_bounded(values, "first", 2)
        remember_bounded(values, "third", 2)

        self.assertEqual(list(values), ["second", "third"])

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

    def test_is_short_generated_reply_detects_short_content(self) -> None:
        self.assertTrue(is_short_generated_reply(["ага"]))
        self.assertTrue(is_short_generated_reply(["ну", "да"]))
        self.assertFalse(is_short_generated_reply(["очень", "даже", "сегодня", "да"]))

    def test_has_degraded_recent_window_detects_repetitive_recent_loop(self) -> None:
        self.assertTrue(
            has_degraded_recent_window(
                [
                    "нормальный",
                    "ответ",
                    "но",
                    "потом",
                    "курлык",
                    "курлык",
                    "курлык",
                    "курлык",
                ]
            )
        )

    def test_has_degraded_recent_window_allows_short_moderate_repetition(self) -> None:
        self.assertFalse(
            has_degraded_recent_window(
                ["малафить", "курлык", "братишка", "вечно", "курлык", "курлык"]
            )
        )

    def test_trim_repetitive_tail_keeps_good_prefix(self) -> None:
        trimmed = trim_repetitive_tail(
            [
                "малафить",
                "курлык",
                "братишка",
                "вечно",
                "курлык",
                "курлык",
                "курлык",
                "курлык",
                "курлык",
            ]
        )
        self.assertEqual(trimmed, ["малафить", "курлык", "братишка", "вечно"])

    def test_is_low_diversity_reply_detects_near_monotone_reply(self) -> None:
        self.assertTrue(
            is_low_diversity_reply(
                [
                    "курлык",
                    "курлык",
                    "курлык",
                    "я",
                    "курлык",
                    "курлык",
                    "курлык",
                    "курлык",
                ]
            )
        )

    def test_is_low_diversity_reply_allows_mixed_reply(self) -> None:
        self.assertFalse(
            is_low_diversity_reply(
                ["малафить", "курлык", "братишка", "вечно", "ты", "че", "ку", "ку"]
            )
        )

    def test_context_bias_prefers_context_tokens(self) -> None:
        rng = random.Random(7)
        tea_count = 0
        coffee_count = 0
        for _ in range(400):
            choice = weighted_next_choice(
                items=[("кофе", 1), ("чай", 1)],
                explore_probability=0.0,
                power=1.0,
                rng=rng,
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
        rng = random.Random(17)
        same_count = 0
        other_count = 0
        for _ in range(400):
            choice = weighted_next_choice(
                items=[("эхо", 4), ("ответ", 4)],
                explore_probability=0.0,
                power=1.0,
                rng=rng,
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
        message = SimpleNamespace(
            reply_to_message=SimpleNamespace(text="Люблю кофе!!! @bot")
        )
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
        message = SimpleNamespace(
            text="Pepe, ответь что-нибудь", entities=None, reply_to_message=None
        )
        self.assertTrue(bot_is_mentioned(message, "PepeEdtaBot", 777))

    def test_bot_is_mentioned_by_plain_text_pepe_case_insensitive(self) -> None:
        message = SimpleNamespace(
            text="ПеПе, ответь что-нибудь", entities=None, reply_to_message=None
        )
        self.assertTrue(bot_is_mentioned(message, "PepeEdtaBot", 777))

    def test_bot_is_mentioned_by_plain_text_pepe_cyrillic(self) -> None:
        message = SimpleNamespace(
            text="пЕпЕ ты тут?", entities=None, reply_to_message=None
        )
        self.assertTrue(bot_is_mentioned(message, "PepeEdtaBot", 777))

    def test_bot_is_not_mentioned_by_substring_only(self) -> None:
        message = SimpleNamespace(
            text="pepega сегодня победил", entities=None, reply_to_message=None
        )
        self.assertFalse(bot_is_mentioned(message, "PepeEdtaBot", 777))

    def test_bot_is_mentioned_by_reply_to_bot_message(self) -> None:
        reply_to_message = SimpleNamespace(from_user=SimpleNamespace(id=777))
        message = SimpleNamespace(
            text="слушай", entities=None, reply_to_message=reply_to_message
        )
        self.assertTrue(bot_is_mentioned(message, "PepeEdtaBot", 777))

    async def test_generate_text_with_seed(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=4444,
            raw_text="Я очень люблю питон",
            tokens=["Я", "очень", "люблю", "питон"],
        )
        await self.db.save_message_and_update_model(
            chat_id=4444,
            raw_text="Я очень люблю кофе",
            tokens=["Я", "очень", "люблю", "кофе"],
        )
        await self.db.save_message_and_update_model(
            chat_id=4444,
            raw_text="Люблю кофе утром",
            tokens=["Люблю", "кофе", "утром"],
        )

        text = await self.generator.generate_text(
            chat_id=4444,
            max_chars=12,
            seed_tokens=["Я", "очень", "люблю"],
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
            rng=random.Random(42),
        )
        self.assertTrue(text)
        self.assertTrue(text.startswith("Я очень"))
        self.assertGreaterEqual(len(text), 5)

    async def test_generate_text_respects_one_token_limit(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=4445,
            raw_text="Привет мир снова",
            tokens=["Привет", "мир", "снова"],
        )

        text = await self.generator.generate_text(
            chat_id=4445,
            max_chars=100,
            max_tokens=1,
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
            rng=random.Random(42),
        )

        self.assertEqual(text, "Привет")

    async def test_generate_text_uses_context_windows_for_start(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=5555,
            raw_text="Люблю кофе утром всегда",
            tokens=["Люблю", "кофе", "утром", "всегда"],
        )
        await self.db.save_message_and_update_model(
            chat_id=5555,
            raw_text="Люблю кофе вечером иногда",
            tokens=["Люблю", "кофе", "вечером", "иногда"],
        )

        text = await self.generator.generate_text(
            chat_id=5555,
            max_chars=25,
            context_tokens=["сегодня", "Люблю", "кофе", "утром"],
            context_bias=1.8,
            context_start_bias=2.2,
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
            rng=random.Random(11),
        )

        self.assertTrue(text)
        self.assertTrue(text.startswith("Люблю кофе утром"))

    async def test_generate_text_falls_back_when_context_not_found(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=6666,
            raw_text="кошка любит солнце ярко",
            tokens=["кошка", "любит", "солнце", "ярко"],
        )
        await self.db.save_message_and_update_model(
            chat_id=6666,
            raw_text="кошка любит дождь тихо",
            tokens=["кошка", "любит", "дождь", "тихо"],
        )
        await self.db.save_message_and_update_model(
            chat_id=6666,
            raw_text="солнце ярко греет дом",
            tokens=["солнце", "ярко", "греет", "дом"],
        )

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
            rng=random.Random(5),
        )

        self.assertTrue(text)
        self.assertFalse(text.startswith("совсем другой"))
        self.assertIn(text.split()[0], {"кошка", "солнце"})

    async def test_generate_text_without_context_matches_legacy_path(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=7777,
            raw_text="утром люблю чай дома",
            tokens=["утром", "люблю", "чай", "дома"],
        )
        await self.db.save_message_and_update_model(
            chat_id=7777,
            raw_text="утром люблю кофе дома",
            tokens=["утром", "люблю", "кофе", "дома"],
        )

        text = await self.generator.generate_text(
            chat_id=7777,
            max_chars=15,
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
            rng=random.Random(13),
        )

        self.assertTrue(text)
        self.assertTrue(text.startswith("утром люблю"))

    async def test_generate_text_is_deterministic_with_injected_rng(self) -> None:
        corpus = [
            ["alpha", "beta", "gamma", "delta", "one"],
            ["alpha", "beta", "gamma", "delta", "two"],
            ["alpha", "beta", "gamma", "other", "three"],
            ["fresh", "path", "continues", "into", "four"],
        ]
        for tokens in corpus:
            await self.db.save_message_and_update_model(
                chat_id=7776,
                raw_text=" ".join(tokens),
                tokens=tokens,
            )

        first = await self.generator.generate_text(
            chat_id=7776,
            max_chars=100,
            max_tokens=10,
            randomness_strength=1.8,
            rng=random.Random(12345),
        )
        second = await self.generator.generate_text(
            chat_id=7776,
            max_chars=100,
            max_tokens=10,
            randomness_strength=1.8,
            rng=random.Random(12345),
        )

        self.assertEqual(first, second)

    async def test_generation_trace_contains_safe_metrics(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=7775,
            raw_text="alpha beta gamma delta",
            tokens=["alpha", "beta", "gamma", "delta"],
        )

        with self.assertLogs("chat_markov", level="DEBUG") as captured_logs:
            text, trace = await self.generator.generate_text_with_trace(
                chat_id=7775,
                max_chars=100,
                randomness_strength=0.0,
                rng=random.Random(7),
            )

        self.assertTrue(text)
        self.assertIsInstance(trace, GenerationTrace)
        self.assertEqual(trace.attempts_used, 1)
        self.assertEqual(trace.markov_order_used, 3)
        self.assertEqual(trace.jump_count, 0)
        self.assertIsNone(trace.rejection_reason)
        self.assertEqual(trace.token_count, len(tokenize(text)))
        self.assertNotIn(text, repr(trace))
        joined_logs = " ".join(captured_logs.output)
        self.assertIn("attempts=1", joined_logs)
        self.assertNotIn(text, joined_logs)

    async def test_generation_trace_reports_final_rejection(self) -> None:
        text, trace = await self.generator.generate_text_with_trace(
            chat_id=7774,
            max_chars=100,
            rng=random.Random(8),
        )

        self.assertEqual(text, "")
        self.assertEqual(trace.attempts_used, 8)
        self.assertEqual(trace.markov_order_used, 0)
        self.assertEqual(trace.jump_count, 0)
        self.assertEqual(trace.rejection_reason, "no_starts")
        self.assertEqual(trace.token_count, 0)

    async def test_generation_trace_reports_actual_backoff_order(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=7773,
            raw_text="alpha beta gamma delta",
            tokens=["alpha", "beta", "gamma", "delta"],
        )
        conn = await self.db._get_conn()
        await conn.execute("DELETE FROM starts3 WHERE chat_id = ?", (7773,))
        await conn.execute("DELETE FROM transitions3 WHERE chat_id = ?", (7773,))
        await conn.commit()

        text, trace = await self.generator.generate_text_with_trace(
            chat_id=7773,
            max_chars=100,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
            rng=random.Random(9),
        )

        self.assertTrue(text)
        self.assertEqual(trace.markov_order_used, 2)

    async def test_high_randomness_long_reply_without_context_is_not_force_rejected(
        self,
    ) -> None:
        tokens = [f"token{index}" for index in range(14)]
        await self.db.save_message_and_update_model(
            chat_id=7778,
            raw_text=" ".join(tokens),
            tokens=tokens,
        )

        text = await self.generator.generate_text(
            chat_id=7778,
            max_chars=500,
            max_tokens=14,
            randomness_strength=2.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
            rng=random.Random(3),
        )

        self.assertEqual(text, " ".join(tokens))

    async def test_jump_branch_is_disabled(self) -> None:
        tokens = [f"token{index}" for index in range(12)]
        await self.db.save_message_and_update_model(
            chat_id=7779,
            raw_text=" ".join(tokens),
            tokens=tokens,
        )

        with patch(
            "markov.weighted_start3_choice",
            wraps=weighted_start3_choice,
        ) as mock_start_choice:
            text = await self.generator.generate_text(
                chat_id=7779,
                max_chars=500,
                max_tokens=12,
                randomness_strength=2.0,
                markov_order=3,
                enable_backoff=True,
                backoff_min_order=1,
                rng=random.Random(4),
            )

        self.assertEqual(text, " ".join(tokens))
        mock_start_choice.assert_called_once()

    async def test_validation_uses_actual_char_truncated_output(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=7780,
            raw_text="fresh ctx1 ctx2 ctx3 ctx4",
            tokens=["fresh", "ctx1", "ctx2", "ctx3", "ctx4"],
        )

        text = await self.generator.generate_text(
            chat_id=7780,
            max_chars=6,
            max_tokens=5,
            seed_tokens=["fresh", "ctx1", "ctx2"],
            context_tokens=["ctx1", "ctx2", "ctx3", "ctx4"],
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
        )

        self.assertEqual(text, "fresh")

    async def test_generate_text_retries_after_internal_validation_failure(self) -> None:
        with patch.object(
            MarkovGenerator,
            "_generate_text_once",
            new=AsyncMock(
                side_effect=[
                    _GenerationAttempt("", 3, 0, "low_diversity", 0),
                    _GenerationAttempt("", 3, 0, "context_heavy", 0),
                    _GenerationAttempt("валидный ответ", 3, 0, None, 2),
                ]
            ),
        ) as mock_generate_once:
            text = await self.generator.generate_text(
                chat_id=8888,
                max_chars=50,
            )

        self.assertEqual(text, "валидный ответ")
        self.assertEqual(mock_generate_once.await_count, 3)

    async def test_generate_text_increases_randomness_on_retries(self) -> None:
        with patch.object(
            MarkovGenerator,
            "_generate_text_once",
            new=AsyncMock(
                side_effect=[
                    _GenerationAttempt("", 3, 0, "low_diversity", 0),
                    _GenerationAttempt("", 3, 0, "context_heavy", 0),
                    _GenerationAttempt("валидный ответ", 3, 0, None, 2),
                ]
            ),
        ) as mock_generate_once:
            await self.generator.generate_text(
                chat_id=8889,
                max_chars=50,
                randomness_strength=0.5,
            )

        strengths = [
            call.kwargs["randomness_strength"] for call in mock_generate_once.await_args_list
        ]
        self.assertEqual(strengths, sorted(strengths))
        self.assertEqual(strengths[0], 0.5)
        self.assertGreater(strengths[-1], strengths[0])

    async def test_generate_text_returns_short_reply_without_verbatim_check(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=8890,
            raw_text="Привет мир снова",
            tokens=["Привет", "мир", "снова"],
        )

        text = await self.generator.generate_text(
            chat_id=8890,
            max_chars=50,
            max_tokens=1,
            randomness_strength=0.0,
        )

        self.assertEqual(text, "Привет")

    def test_context_bias_does_not_override_repetition_penalty(self) -> None:
        rng = random.Random(23)
        repeated_count = 0
        fresh_count = 0
        for _ in range(400):
            choice = weighted_next_choice(
                items=[("кофе", 3), ("бодрит", 3)],
                explore_probability=0.0,
                power=1.0,
                rng=rng,
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
        rng = random.Random(29)
        repeated_count = 0
        fresh_count = 0
        for _ in range(400):
            choice = weighted_next_choice(
                items=[("эхо", 4), ("ответ", 4)],
                explore_probability=0.0,
                power=1.0,
                rng=rng,
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
