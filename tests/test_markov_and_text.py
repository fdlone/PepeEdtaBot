from __future__ import annotations

import random
import unittest
import uuid
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.core.markov import (
    GenerationTrace,
    MarkovGenerator,
    _GenerationAttempt,
    context_start_probability,
    detokenize,
    escalated_randomness_strength,
    exploration_adjusted_power,
    finalize_reply_ending,
    has_degraded_recent_window,
    is_context_heavy_reply,
    is_low_diversity_reply,
    is_short_generated_reply,
    remember_bounded,
    tokenize,
    trim_repetitive_tail,
    weighted_next_choice,
    weighted_start2_choice,
    weighted_start3_choice,
)
from app.core.reply_policy import bot_is_mentioned
from app.core.text import capitalize_reply_sentences, sanitize_text
from app.handlers.learning import extract_context_tokens
from app.infrastructure.database import Database


class TestCapitalizeReplySentences(unittest.TestCase):
    def test_empty_whitespace_and_punctuation_only_are_unchanged(self) -> None:
        for text in ("", "   ", "...?!", " \t!? "):
            with self.subTest(text=text):
                self.assertEqual(capitalize_reply_sentences(text), text)

    def test_capitalizes_cyrillic_latin_and_mixed_sentences(self) -> None:
        cases = {
            "привет": "Привет",
            "hello": "Hello",
            "привет. hello! пока? bye": "Привет. Hello! Пока? Bye",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(capitalize_reply_sentences(text), expected)

    def test_skips_non_letters_before_sentence_start(self) -> None:
        cases = {
            '  "привет': '  "Привет',
            "🙂[hello": "🙂[Hello",
            "42... (ответ": "42... (Ответ",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(capitalize_reply_sentences(text), expected)

    def test_treats_sentence_ending_runs_as_one_boundary(self) -> None:
        self.assertEqual(
            capitalize_reply_sentences("что?! правда... да!? конечно"),
            "Что?! Правда... Да!? Конечно",
        )

    def test_preserves_existing_uppercase_and_mid_token_letters(self) -> None:
        cases = {
            "Already Uppercase. OK": "Already Uppercase. OK",
            "3d printer. abc_def works": "3d printer. abc_def works",
            "_name stays. 7zip archive": "_name stays. 7zip archive",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(capitalize_reply_sentences(text), expected)

    def test_is_idempotent_and_preserves_length(self) -> None:
        for text in ("ßeta. привет", "🙂 «hello?!» пока", "3d abc_def"):
            with self.subTest(text=text):
                result = capitalize_reply_sentences(text)
                self.assertEqual(capitalize_reply_sentences(result), result)
                self.assertEqual(len(result), len(text))


class TestContextStartProbability(unittest.TestCase):
    def test_expected_probabilities(self) -> None:
        self.assertEqual(context_start_probability(1.0), 0.0)
        self.assertAlmostEqual(context_start_probability(2.2), 0.5454545454545454)
        self.assertEqual(context_start_probability(4.0), 0.75)
        self.assertEqual(context_start_probability(0.5), 0.0)
        self.assertEqual(context_start_probability(0.0), 0.0)

    def test_is_monotonically_increasing(self) -> None:
        biases = [1.0, 1.5, 2.2, 3.0, 4.0]
        probabilities = [context_start_probability(bias) for bias in biases]

        self.assertEqual(probabilities, sorted(probabilities))
        self.assertTrue(
            all(
                lower < upper
                for lower, upper in zip(probabilities, probabilities[1:])
            )
        )


class TestFinalizeReplyEnding(unittest.TestCase):
    def test_trims_dangling_russian_connector_and_adds_period(self) -> None:
        tokens = ["я", "думаю", "что", "это", "и"]

        result = finalize_reply_ending(tokens)

        self.assertEqual(result, ["я", "думаю", "что", "это", "."])
        self.assertEqual(tokens, ["я", "думаю", "что", "это", "и"])

    def test_keeps_already_terminated_text_unchanged(self) -> None:
        tokens = ["привет", "всем", "сегодня", "друзья", "."]

        self.assertEqual(finalize_reply_ending(tokens), tokens)

    def test_preserves_terminal_question_mark(self) -> None:
        tokens = ["как", "у", "тебя", "дела", "?"]

        self.assertEqual(finalize_reply_ending(tokens), tokens)

    def test_preserves_terminal_exclamation_mark_for_short_reply(self) -> None:
        tokens = ["привет", "всем", "!"]

        self.assertEqual(finalize_reply_ending(tokens), tokens)

    def test_adds_period_when_terminal_punctuation_is_missing(self) -> None:
        tokens = ["это", "полностью", "готовый", "ответ"]

        self.assertEqual(finalize_reply_ending(tokens), [*tokens, "."])

    def test_leaves_short_reply_unchanged(self) -> None:
        tokens = ["ну", "да", "ладно"]

        result = finalize_reply_ending(tokens)

        self.assertEqual(result, tokens)
        self.assertIsNot(result, tokens)

    def test_never_strips_below_minimum_content_tokens(self) -> None:
        tokens = ["да", "нет", "и", "но"]

        result = finalize_reply_ending(tokens)

        self.assertEqual(result, ["да", "нет", "и", "но", "."])
        self.assertGreaterEqual(len([token for token in result if token.isalpha()]), 4)

    def test_cleans_trailing_comma(self) -> None:
        tokens = ["это", "точно", "готовый", "ответ", ","]

        self.assertEqual(finalize_reply_ending(tokens), [*tokens[:-1], "."])


class TestHiddenStartGeneration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_hidden_start_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        self.generator = MarkovGenerator(self.db)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def test_hidden_start_emits_successors_and_seeds_dedup_from_state(
        self,
    ) -> None:
        chat_id = 4401
        start_triplet = ["alpha", "beta", "gamma"]
        await self.db.save_message_and_update_model(
            chat_id=chat_id,
            raw_text="alpha beta gamma alpha beta",
            tokens=[*start_triplet, "alpha", "beta"],
        )
        await self.db.save_message_and_update_model(
            chat_id=chat_id,
            raw_text="gamma alpha beta gamma",
            tokens=["gamma", "alpha", "beta", "gamma"],
        )
        await self.db.save_message_and_update_model(
            chat_id=chat_id,
            raw_text="gamma alpha beta delta continues safely",
            tokens=["gamma", "alpha", "beta", "delta", "continues", "safely"],
        )

        attempt = await self.generator._generate_text_once(
            chat_id=chat_id,
            max_chars=100,
            max_tokens=6,
            seed_tokens=start_triplet,
            randomness_strength=0.0,
            rng=random.Random(42),
            emit_start=False,
        )

        self.assertTrue(attempt.text, attempt.rejection_reason)
        self.assertFalse(attempt.text.startswith("alpha beta gamma"))
        self.assertNotIn("alpha beta gamma", attempt.text)
        self.assertTrue(attempt.text.startswith("alpha beta delta"))


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

    def test_sanitize_redacts_email_keeping_surrounding_words(self) -> None:
        # Phase 4.2b: emails are PII and must be redacted out of the corpus,
        # while the surrounding words are preserved.
        clean = sanitize_text("Напиши на user@example.com если что")
        self.assertEqual(clean, "Напиши на если что")

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

    def test_high_exploration_keeps_context_and_repetition_weights(self) -> None:
        rng = random.Random(20260620)
        context_choice_count = 0
        repeated_choice_count = 0

        for _ in range(500):
            context_choice = weighted_next_choice(
                items=[("context", 1), ("neutral", 1)],
                explore_probability=1.0,
                power=1.0,
                rng=rng,
                context_token_set={"context"},
                context_pairs={("state", "context")},
                context_triplets={("previous", "state", "context")},
                current_state=("previous", "state"),
                context_bias=3.0,
            )
            if context_choice == "context":
                context_choice_count += 1

            repetition_choice = weighted_next_choice(
                items=[("repeat", 10), ("fresh", 1)],
                explore_probability=1.0,
                power=1.0,
                rng=rng,
                current_state=("state", "repeat"),
                recent_tokens=["state", "repeat", "repeat"],
                seen_pairs={("repeat", "repeat")},
                seen_triplets={("state", "repeat", "repeat")},
            )
            if repetition_choice == "repeat":
                repeated_choice_count += 1

        self.assertGreater(context_choice_count, 400)
        self.assertLess(repeated_choice_count, 100)

    def test_high_exploration_start_choice_remains_weighted(self) -> None:
        rng = random.Random(20260620)
        frequent_count = 0
        for _ in range(500):
            choice = weighted_start2_choice(
                items=[("frequent", "start", 10_000), ("rare", "start", 1)],
                explore_probability=1.0,
                power=1.0,
                rng=rng,
            )
            if choice == ("frequent", "start"):
                frequent_count += 1

        self.assertGreater(frequent_count, 350)
        self.assertGreater(exploration_adjusted_power(1.0, 1.0), 0.0)

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

    async def test_context_trigram_uses_hidden_transition_state(self) -> None:
        await self.db.save_message_and_update_model(
            chat_id=5555,
            raw_text="global lead Люблю кофе утром продолжается вполне безопасно",
            tokens=[
                "global",
                "lead",
                "Люблю",
                "кофе",
                "утром",
                "продолжается",
                "вполне",
                "безопасно",
            ],
        )

        text, trace = await self.generator.generate_text_with_trace(
            chat_id=5555,
            max_chars=100,
            context_tokens=["сегодня", "Люблю", "кофе", "утром"],
            context_bias=1.8,
            context_start_bias=4.0,
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
            rng=random.Random(11),
        )

        self.assertTrue(text)
        self.assertFalse(text.startswith("Люблю кофе утром"))
        self.assertTrue(text.startswith("продолжается вполне безопасно"))
        self.assertEqual(trace.start_source, "hidden_context")
        self.assertEqual(trace.markov_order_used, 3)

    async def test_context_start_bias_gates_contextual_start_path(self) -> None:
        chat_id = 5556
        context_tokens = ["topic", "context", "alpha", "beta"]
        await self.db.save_message_and_update_model(
            chat_id=chat_id,
            raw_text="context alpha beta continues safely",
            tokens=["context", "alpha", "beta", "continues", "safely"],
        )
        for _ in range(12):
            await self.db.save_message_and_update_model(
                chat_id=chat_id,
                raw_text="global path starts elsewhere safely",
                tokens=["global", "path", "starts", "elsewhere", "safely"],
            )

        without_contextual_start = await self.generator.generate_text(
            chat_id=chat_id,
            max_chars=100,
            context_tokens=context_tokens,
            context_start_bias=1.0,
            randomness_strength=0.0,
            rng=random.Random(1),
        )
        with_contextual_start, contextual_trace = (
            await self.generator.generate_text_with_trace(
                chat_id=chat_id,
                max_chars=100,
                context_tokens=context_tokens,
                context_start_bias=4.0,
                randomness_strength=0.0,
                rng=random.Random(1),
            )
        )

        self.assertTrue(without_contextual_start)
        self.assertFalse(without_contextual_start.startswith("context alpha beta"))
        self.assertTrue(with_contextual_start)
        self.assertFalse(with_contextual_start.startswith("context alpha beta"))
        self.assertTrue(with_contextual_start.startswith("continues safely"))
        self.assertEqual(contextual_trace.start_source, "hidden_context")

    async def test_context_pair_backoff_builds_hidden_state(self) -> None:
        chat_id = 5557
        await self.db.save_message_and_update_model(
            chat_id=chat_id,
            raw_text="global lead alpha beta hidden output continues safely",
            tokens=[
                "global",
                "lead",
                "alpha",
                "beta",
                "hidden",
                "output",
                "continues",
                "safely",
            ],
        )

        text, trace = await self.generator.generate_text_with_trace(
            chat_id=chat_id,
            max_chars=100,
            context_tokens=["alpha", "beta"],
            context_start_bias=4.0,
            randomness_strength=0.0,
            rng=random.Random(1),
        )

        self.assertTrue(text)
        self.assertFalse(text.startswith("alpha beta"))
        self.assertTrue(text.startswith("output continues safely"))
        self.assertEqual(trace.start_source, "hidden_context")
        self.assertEqual(trace.markov_order_used, 2)

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

        text, trace = await self.generator.generate_text_with_trace(
            chat_id=6666,
            max_chars=17,
            context_tokens=["совсем", "другой", "контекст"],
            context_bias=2.4,
            context_start_bias=4.0,
            randomness_strength=0.0,
            markov_order=3,
            enable_backoff=True,
            backoff_min_order=1,
            rng=random.Random(5),
        )

        self.assertTrue(text)
        self.assertFalse(text.startswith("совсем другой"))
        self.assertIn(text.split()[0], {"кошка", "солнце"})
        self.assertEqual(trace.start_source, "global")

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
        self.assertEqual(trace.start_source, "global")
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
        self.assertEqual(trace.attempts_used, 1)
        self.assertEqual(trace.markov_order_used, 0)
        self.assertEqual(trace.jump_count, 0)
        self.assertEqual(trace.rejection_reason, "no_starts")
        self.assertEqual(trace.token_count, 0)
        self.assertEqual(trace.start_source, "global")

    async def test_generation_trace_reports_seed_start_source(self) -> None:
        chat_id = 7772
        await self.db.save_message_and_update_model(
            chat_id=chat_id,
            raw_text="seed alpha beta continues safely",
            tokens=["seed", "alpha", "beta", "continues", "safely"],
        )

        text, trace = await self.generator.generate_text_with_trace(
            chat_id=chat_id,
            max_chars=100,
            seed_tokens=["seed", "alpha", "beta"],
            randomness_strength=0.0,
            rng=random.Random(4),
        )

        self.assertTrue(text)
        self.assertTrue(text.startswith("seed alpha beta"))
        self.assertEqual(trace.start_source, "seed")

    async def test_generation_attempt_budget_is_bounded(self) -> None:
        rejected_attempt = _GenerationAttempt("", 0, 0, "no_starts", 0, "global")
        with patch(
            "app.core.markov.MarkovGenerator._generate_text_once",
            new=AsyncMock(return_value=rejected_attempt),
        ) as generate_once:
            text, trace = await self.generator.generate_text_with_trace(
                chat_id=7774,
                max_chars=100,
                rng=random.Random(8),
                attempt_budget=3,
            )

        self.assertEqual(text, "")
        self.assertEqual(trace.attempts_used, 3)
        self.assertEqual(generate_once.await_count, 3)

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

        self.assertEqual(text, " ".join(tokens) + ".")

    async def test_jump_branch_is_disabled(self) -> None:
        tokens = [f"token{index}" for index in range(12)]
        await self.db.save_message_and_update_model(
            chat_id=7779,
            raw_text=" ".join(tokens),
            tokens=tokens,
        )

        with patch(
            "app.core.markov.weighted_start3_choice",
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

        self.assertEqual(text, " ".join(tokens) + ".")
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
                    _GenerationAttempt("", 3, 0, "low_diversity", 0, "global"),
                    _GenerationAttempt("", 3, 0, "context_heavy", 0, "global"),
                    _GenerationAttempt("валидный ответ", 3, 0, None, 2, "global"),
                ]
            ),
        ) as mock_generate_once:
            text = await self.generator.generate_text(
                chat_id=8888,
                max_chars=50,
                attempt_budget=3,
            )

        self.assertEqual(text, "валидный ответ")
        self.assertEqual(mock_generate_once.await_count, 3)

    async def test_generate_text_increases_randomness_on_retries(self) -> None:
        with patch.object(
            MarkovGenerator,
            "_generate_text_once",
            new=AsyncMock(
                side_effect=[
                    _GenerationAttempt("", 3, 0, "low_diversity", 0, "global"),
                    _GenerationAttempt("", 3, 0, "context_heavy", 0, "global"),
                    _GenerationAttempt("валидный ответ", 3, 0, None, 2, "global"),
                ]
            ),
        ) as mock_generate_once:
            await self.generator.generate_text(
                chat_id=8889,
                max_chars=50,
                randomness_strength=0.5,
                attempt_budget=3,
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
