"""Characterization tests for MarkovGenerator._generate_text_once (audit R8).

These tests pin the EXACT current output of the generation pipeline for a
fixed corpus and seeded RNG across the distinct branches of
``_generate_text_once`` (global/seed/contextual start selection, n-gram
backoff, emit_start, and trace aggregation). They intentionally assert on
concrete strings and trace fields rather than properties: their purpose is to
make the planned complexity refactor *behaviour-preserving* — any change that
alters generated text or trace metadata for these inputs must be deliberate.

Outputs were captured from the implementation as of the refactor baseline and
verified stable across PYTHONHASHSEED values (set iteration order does not feed
RNG consumption, which is membership-only).
"""
from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.context_state_matcher import ContextStateMatch
from app.core.markov import (
    MarkovGenerator,
    _ContextualStateSelection,
)
from app.infrastructure.database import Database

# Fixed corpus. Insertion order matters — start/transition weights depend on it.
CORPUS: tuple[str, ...] = (
    "мама мыла раму очень чисто и аккуратно дома",
    "папа читал газету утром на кухне за столом",
    "мама готовила обед для всей большой семьи дома",
    "дети играли в саду весь день до самого вечера",
    "папа чинил машину в гараже целый выходной день",
    "мама мыла посуду после ужина быстро и тихо",
    "кошка спала на диване весь длинный зимний день",
    "собака бегала по двору и громко лаяла утром",
)

CHAT_ID = 9001


class _GenerationCharacterizationBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_markov_char_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        self.generator = MarkovGenerator(self.db)
        for line in CORPUS:
            await self.db.save_message_and_update_model(
                chat_id=CHAT_ID, raw_text=line, tokens=line.split()
            )

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)


class TestGenerateTextOnceCharacterization(_GenerationCharacterizationBase):
    async def test_empty_model_rejects_with_no_starts(self) -> None:
        empty_path = Path(f"test_markov_empty_{uuid.uuid4().hex}.sqlite")
        empty_db = Database(str(empty_path))
        await empty_db.init()
        try:
            gen = MarkovGenerator(empty_db)
            attempt = await gen._generate_text_once(
                chat_id=777, max_chars=100, max_tokens=10, rng=random.Random(1)
            )
        finally:
            await empty_db.close()
            empty_path.unlink(missing_ok=True)
        self.assertEqual(attempt.text, "")
        self.assertEqual(attempt.rejection_reason, "no_starts")
        self.assertEqual(attempt.start_source, "global")

    async def test_two_word_corpus_rejects_with_no_start_transition(self) -> None:
        # Only 2-token messages: a 2-gram start exists but has no stored
        # transition to extend it, so the global-start path rejects.
        ns_path = Path(f"test_markov_ns_{uuid.uuid4().hex}.sqlite")
        ns_db = Database(str(ns_path))
        await ns_db.init()
        try:
            gen = MarkovGenerator(ns_db)
            for line in ("альфа бета", "гамма дельта", "эхо фокс"):
                await ns_db.save_message_and_update_model(
                    chat_id=1, raw_text=line, tokens=line.split()
                )
            attempt = await gen._generate_text_once(
                chat_id=1, max_chars=200, max_tokens=12,
                randomness_strength=0.0, rng=random.Random(7), emit_start=True,
            )
        finally:
            await ns_db.close()
            ns_path.unlink(missing_ok=True)
        self.assertEqual(attempt.text, "")
        self.assertEqual(attempt.rejection_reason, "no_start_transition")
        self.assertEqual(attempt.markov_order_used, 2)
        self.assertEqual(attempt.start_source, "global")
        self.assertEqual(attempt.token_count, 0)

    async def test_global_start_order3_seed1(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            randomness_strength=0.0, rng=random.Random(1), emit_start=True,
        )
        self.assertEqual(
            attempt.text, "папа читал газету утром на кухне за столом."
        )
        self.assertIsNone(attempt.rejection_reason)
        self.assertEqual(attempt.markov_order_used, 3)
        self.assertEqual(attempt.token_count, 9)
        self.assertEqual(attempt.start_source, "global")
        self.assertEqual(attempt.jump_count, 0)

    async def test_global_start_order3_seed7(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            randomness_strength=0.0, rng=random.Random(7), emit_start=True,
        )
        self.assertEqual(
            attempt.text, "мама мыла раму очень чисто и аккуратно дома."
        )
        self.assertEqual(attempt.markov_order_used, 3)
        self.assertEqual(attempt.start_source, "global")

    async def test_global_start_stops_where_order1_used_to_continue_seed42(self) -> None:
        # Pre-013 the walk spliced "до самого вечера" here via the order-1
        # chain; with order 1 removed it ends cleanly at the order-2 dead end.
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            randomness_strength=0.0, rng=random.Random(42), emit_start=True,
        )
        self.assertEqual(
            attempt.text,
            "папа чинил машину в гараже целый выходной день.",
        )
        self.assertEqual(attempt.markov_order_used, 3)
        self.assertEqual(attempt.token_count, 9)
        self.assertEqual(attempt.start_source, "global")

    async def test_global_start_stops_where_order1_used_to_continue_seed123(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            randomness_strength=0.0, rng=random.Random(123), emit_start=True,
        )
        self.assertEqual(
            attempt.text,
            "кошка спала на диване весь длинный зимний день.",
        )
        self.assertEqual(attempt.markov_order_used, 3)
        self.assertEqual(attempt.token_count, 9)

    async def test_seed3_start_uses_seed_source(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            seed_tokens=["мама", "мыла", "раму"],
            randomness_strength=0.0, rng=random.Random(5), emit_start=True,
        )
        self.assertEqual(
            attempt.text, "мама мыла раму очень чисто и аккуратно дома."
        )
        self.assertEqual(attempt.start_source, "seed")
        self.assertEqual(attempt.markov_order_used, 3)

    async def test_seed2_start_uses_seed_source_order2(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            seed_tokens=["папа", "читал"],
            randomness_strength=0.0, rng=random.Random(5), emit_start=True,
        )
        self.assertEqual(
            attempt.text, "папа читал газету утром на кухне за столом."
        )
        self.assertEqual(attempt.start_source, "seed")
        self.assertEqual(attempt.markov_order_used, 2)

    async def test_emit_start_false_drops_leading_state_tokens(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            seed_tokens=["мама", "мыла", "раму"],
            randomness_strength=0.0, rng=random.Random(5), emit_start=False,
        )
        self.assertEqual(attempt.text, "очень чисто и аккуратно дома.")
        self.assertEqual(attempt.start_source, "seed")
        self.assertEqual(attempt.markov_order_used, 3)

    async def test_contextual_start_exact_match(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            context_tokens=["папа", "чинил", "машину"],
            context_start_bias=4.0, randomness_strength=0.0,
            rng=random.Random(3), emit_start=True,
        )
        self.assertEqual(
            attempt.text, "чинил машину в гараже целый выходной день."
        )
        self.assertEqual(attempt.start_source, "context")
        self.assertEqual(attempt.context_exact_matches, 1)
        self.assertEqual(attempt.context_casefold_matches, 0)
        self.assertEqual(attempt.markov_order_used, 3)

    async def test_contextual_start_casefold_match(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            context_tokens=["ПАПА", "ЧИНИЛ", "МАШИНУ"],
            context_start_bias=4.0, randomness_strength=0.0,
            fuzzy_context_casefold=True, rng=random.Random(3), emit_start=True,
        )
        self.assertEqual(
            attempt.text, "чинил машину в гараже целый выходной день."
        )
        self.assertEqual(attempt.start_source, "context")
        self.assertEqual(attempt.context_exact_matches, 0)
        self.assertEqual(attempt.context_casefold_matches, 1)

    async def test_contextual_start_falls_back_when_context_absent(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            context_tokens=["zzz", "qqq", "www"],
            context_start_bias=4.0, randomness_strength=0.0,
            rng=random.Random(3), emit_start=True,
        )
        self.assertEqual(
            attempt.text, "мама мыла раму очень чисто и аккуратно дома."
        )
        self.assertEqual(attempt.start_source, "global")
        self.assertEqual(attempt.hidden_context_fallbacks, 1)

    async def test_no_backoff_order3_seed7(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            randomness_strength=0.0, enable_backoff=False,
            rng=random.Random(7), emit_start=True,
        )
        self.assertEqual(
            attempt.text, "мама мыла раму очень чисто и аккуратно дома."
        )
        self.assertEqual(attempt.markov_order_used, 3)

    async def test_order2_model_seed7(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            markov_order=2, randomness_strength=0.0,
            rng=random.Random(7), emit_start=True,
        )
        self.assertEqual(
            attempt.text,
            "папа чинил машину в гараже целый выходной день.",
        )
        self.assertEqual(attempt.markov_order_used, 2)
        self.assertEqual(attempt.token_count, 9)


class TestMidGenerationJump(_GenerationCharacterizationBase):
    """M4: a mid-generation topic-drift jump splices a connective + new start."""

    def _connective_words(self) -> set[str]:
        from app.core.markov import JUMP_CONNECTIVE_TOKENS

        return {tok for phrase in JUMP_CONNECTIVE_TOKENS for tok in phrase}

    async def test_jump_probability_zero_matches_pre_m4_output(self) -> None:
        # The default (0.0) must reproduce the exact pre-M4 characterization
        # output — the feature is off unless explicitly enabled.
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            randomness_strength=0.0, jump_probability=0.0,
            rng=random.Random(1), emit_start=True,
        )
        self.assertEqual(attempt.text, "папа читал газету утром на кухне за столом.")
        self.assertEqual(attempt.jump_count, 0)

    async def test_certain_jump_splices_connective_and_extends(self) -> None:
        # With a certain jump and room to grow, the walk drifts to new starts
        # once it passes 8 tokens: the trace records the jumps and a connective
        # word appears spliced in, running well past a single 9-token sentence.
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=30,
            randomness_strength=0.0, jump_probability=1.0,
            rng=random.Random(2), emit_start=True,
        )
        self.assertGreaterEqual(attempt.jump_count, 1)
        words = set(attempt.text.replace(",", " ").split())
        self.assertTrue(
            words & self._connective_words(),
            f"expected a connective in {attempt.text!r}",
        )
        self.assertGreater(attempt.token_count, 9)

    async def test_jumps_capped_per_reply(self) -> None:
        # Even a certain per-step jump may fire at most JUMP_MAX_PER_REPLY
        # times: extra length must come from the chain, not from splicing more
        # topics (uncapped, 18% of winners carried >=2 jumps — the salad tail).
        from app.core.markov import JUMP_MAX_PER_REPLY

        for seed in range(1, 6):
            attempt = await self.generator._generate_text_once(
                chat_id=CHAT_ID, max_chars=400, max_tokens=45,
                randomness_strength=0.0, jump_probability=1.0,
                rng=random.Random(seed), emit_start=True,
            )
            self.assertLessEqual(attempt.jump_count, JUMP_MAX_PER_REPLY)

    async def test_certain_jump_leaves_no_splice_stutter(self) -> None:
        # The splice point must not read ",," or repeat the connective word
        # ("хотя, хотя") — the trim drops dangling commas/conjunctions first.
        connective_words = self._connective_words() - {","}
        for seed in range(1, 8):
            attempt = await self.generator._generate_text_once(
                chat_id=CHAT_ID, max_chars=400, max_tokens=45,
                randomness_strength=0.0, jump_probability=1.0,
                rng=random.Random(seed), emit_start=True,
            )
            self.assertNotIn(",,", attempt.text)
            tokens = attempt.text.replace(",", " ").split()
            for first, second in zip(tokens, tokens[1:], strict=False):
                if first in connective_words:
                    self.assertNotEqual(
                        first, second,
                        f"stuttered connective in {attempt.text!r}",
                    )


class TestContextStartAffinity(unittest.TestCase):
    """Context-affine global start sampling (pure helpers)."""

    def test_context_start_stems_folds_and_drops_stopwords(self) -> None:
        from app.core.markov import context_start_stems

        stems = context_start_stems("А гнойному пидору навалил".split())
        self.assertIn("гнойн", stems)
        self.assertIn("пидор", stems)
        self.assertNotIn("а", stems)  # stopword

    def test_affinity_boosts_context_sharing_start(self) -> None:
        from app.core.markov import context_start_stems, weighted_start3_choice

        # 1 on-topic start among 50 heavier off-topic ones: plain sampling
        # almost never draws it, affinity 3.0 (два общих стема => x9) must.
        starts = [("слава", "гнойный", "пидор", 2)] + [
            (f"слово{i}", f"текст{i}", f"конец{i}", 10) for i in range(50)
        ]
        stems = context_start_stems("кто гнойный пидор ?".split())

        rng = random.Random(7)
        plain = sum(
            weighted_start3_choice(starts, 0.0, 0.75, rng)[0] == "слава"
            for _ in range(300)
        )
        rng = random.Random(7)
        boosted = sum(
            weighted_start3_choice(
                starts, 0.0, 0.75, rng,
                context_stems=stems, context_start_affinity=3.0,
            )[0] == "слава"
            for _ in range(300)
        )

        self.assertLess(plain, 10)
        self.assertGreater(boosted, plain * 3)

    def test_affinity_skips_question_echo_start(self) -> None:
        # A start that is entirely made of context stems re-asks the question
        # («кто гнойный пидор» is itself a learned start) — it must not be
        # boosted, or it out-boosts the actual answers.
        from app.core.markov import context_start_stems, weighted_start3_choice

        starts = [
            ("кто", "гнойный", "пидор", 12),   # echo of the question
            ("слава", "гнойный", "пидор", 2),  # the answer
        ]
        stems = context_start_stems("кто гнойный пидор ?".split())

        rng = random.Random(3)
        wins = sum(
            weighted_start3_choice(
                starts, 0.0, 0.75, rng,
                context_stems=stems, context_start_affinity=6.0,
            )[0]
            == "слава"
            for _ in range(300)
        )
        # answer boosted 36x vs echo cnt-advantage 6x -> answer must dominate
        self.assertGreater(wins, 200)

    def test_affinity_one_is_identical_to_plain(self) -> None:
        from app.core.markov import context_start_stems, weighted_start3_choice

        starts = [(f"а{i}", f"б{i}", f"в{i}", i + 1) for i in range(20)]
        stems = context_start_stems("а3 б3".split())
        for seed in range(20):
            plain = weighted_start3_choice(
                starts, 0.3, 0.75, random.Random(seed)
            )
            with_knob_off = weighted_start3_choice(
                starts, 0.3, 0.75, random.Random(seed),
                context_stems=stems, context_start_affinity=1.0,
            )
            self.assertEqual(plain, with_knob_off)


class TestJumpSpliceHelpers(unittest.TestCase):
    """Pure helpers behind the M4 connective splice."""

    def test_trim_splice_tail_drops_dangling_tail(self) -> None:
        from app.core.markov import trim_splice_tail

        generated = ["мама", "мыла", "раму", ",", "хотя"]
        trim_splice_tail(generated)
        self.assertEqual(generated, ["мама", "мыла", "раму"])

    def test_trim_splice_tail_drops_dangling_preposition(self) -> None:
        # Live trace: "я и такие ессе писал на, а вообще ..." — the jump fired
        # while the walk stood on a preposition, leaving a fragment.
        from app.core.markov import trim_splice_tail

        generated = ["ессе", "писал", "на"]
        trim_splice_tail(generated)
        self.assertEqual(generated, ["ессе", "писал"])

    def test_trim_splice_tail_keeps_at_least_one_token(self) -> None:
        from app.core.markov import trim_splice_tail

        generated = [",", "и", "ну"]
        trim_splice_tail(generated)
        self.assertEqual(generated, [","])

    def test_pick_jump_connective_skips_excluded(self) -> None:
        from app.core.markov import JUMP_CONNECTIVE_TOKENS, pick_jump_connective

        exclude = list(JUMP_CONNECTIVE_TOKENS[1:])
        for seed in range(10):
            picked = pick_jump_connective(random.Random(seed), exclude=exclude)
            self.assertEqual(picked, JUMP_CONNECTIVE_TOKENS[0])

    def test_pick_jump_connective_falls_back_when_all_excluded(self) -> None:
        from app.core.markov import JUMP_CONNECTIVE_TOKENS, pick_jump_connective

        picked = pick_jump_connective(
            random.Random(1), exclude=list(JUMP_CONNECTIVE_TOKENS)
        )
        self.assertIn(picked, JUMP_CONNECTIVE_TOKENS)


class TestGenerateTextWithTraceCharacterization(_GenerationCharacterizationBase):
    async def test_trace_single_attempt_success(self) -> None:
        text, trace = await self.generator.generate_text_with_trace(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            randomness_strength=0.0, rng=random.Random(7), attempt_budget=3,
        )
        self.assertEqual(text, "мама мыла раму очень чисто и аккуратно дома.")
        self.assertEqual(trace.attempts_used, 1)
        self.assertEqual(trace.markov_order_used, 3)
        self.assertEqual(trace.token_count, 9)
        self.assertIsNone(trace.rejection_reason)
        self.assertEqual(trace.start_source, "global")
        self.assertEqual(trace.jump_count, 0)


class TestContextualMatchCounts(unittest.TestCase):
    """Pure mapping of contextual match kind to trace counters (audit R8)."""

    def _counts(self, match_kind: str) -> tuple[int, ...]:
        selection = _ContextualStateSelection(
            state=("a", "b", "c"),
            order=3,
            match_kind=match_kind,
        )
        return MarkovGenerator._contextual_match_counts(selection)

    def test_exact(self) -> None:
        self.assertEqual(self._counts("exact"), (1, 0))

    def test_casefold(self) -> None:
        self.assertEqual(self._counts("casefold"), (0, 1))


class TestFinalizeAttempt(unittest.TestCase):
    """Direct tests for the finalize/quality-gate tail (audit R8).

    _finalize_attempt is RNG-free and DB-free, so it is exercised directly to
    pin both the too-short reject and the success path, and to prove that the
    accumulated trace metadata is propagated unchanged onto the result.
    """

    def _finalize(self, tokens: list[str], context_tokens: list[str] | None = None):
        generator = MarkovGenerator(MagicMock())
        return generator._finalize_attempt(
            list(tokens),
            max_chars=200,
            context_tokens=context_tokens or [],
            order_used=2,
            jump_count=1,
            start_source="seed",
            context_exact_matches=3,
            context_casefold_matches=4,
            hidden_context_fallbacks=7,
        )

    def test_too_short_rejects_and_propagates_trace(self) -> None:
        attempt = self._finalize(["я"])
        self.assertEqual(attempt.text, "")
        self.assertEqual(attempt.rejection_reason, "result_too_short")
        self.assertEqual(attempt.token_count, 0)
        # All accumulated trace metadata flows through the reject unchanged.
        self.assertEqual(attempt.markov_order_used, 2)
        self.assertEqual(attempt.jump_count, 1)
        self.assertEqual(attempt.start_source, "seed")
        self.assertEqual(attempt.context_exact_matches, 3)
        self.assertEqual(attempt.context_casefold_matches, 4)
        self.assertEqual(attempt.hidden_context_fallbacks, 7)

    def test_success_returns_text_and_token_count(self) -> None:
        attempt = self._finalize(
            ["мама", "мыла", "раму", "очень", "чисто", "и", "аккуратно"]
        )
        self.assertEqual(attempt.text, "мама мыла раму очень чисто и аккуратно.")
        self.assertIsNone(attempt.rejection_reason)
        self.assertEqual(attempt.token_count, 8)
        self.assertEqual(attempt.start_source, "seed")
        self.assertEqual(attempt.markov_order_used, 2)


class TestCasefoldCandidateBuilders(unittest.IsolatedAsyncioTestCase):
    """Mocked-matcher tests for the casefold candidate builders (audit R8):
    filtering by match kind and count/recency weighting."""

    def _generator(self, matches: list[ContextStateMatch]) -> MarkovGenerator:
        generator = MarkovGenerator(MagicMock())
        generator._context_state_matcher = MagicMock()
        generator._context_state_matcher.match = AsyncMock(return_value=matches)
        return generator

    async def test_casefold3_skips_exact_and_weights_by_count(self) -> None:
        matches = [
            ContextStateMatch(("a", "b", "c"), "exact", similarity=1.0, transition_count=9),
            ContextStateMatch(("d", "e", "f"), "casefold", similarity=1.0, transition_count=4),
        ]
        generator = self._generator(matches)
        out = await generator._build_casefold3_candidates(1, [("x", "y", "z")], 1, 1.0)
        # The exact match belongs to the caller's own tier; weight = 4 * (1 + 1*0.35).
        self.assertEqual(len(out), 1)
        state, weight, count = out[0]
        self.assertEqual(state, ("d", "e", "f"))
        self.assertEqual(count, 4)
        self.assertAlmostEqual(weight, 4 * 1.35)
        generator._context_state_matcher.match.assert_awaited_with(1, ("x", "y", "z"), 3)

    async def test_casefold2_returns_states_with_transitions(self) -> None:
        matches = [
            ContextStateMatch(("a", "b"), "casefold", similarity=1.0, transition_count=1),
        ]
        generator = self._generator(matches)
        with patch.object(MarkovGenerator, "_get2", AsyncMock(return_value=[("c", 2)])):
            additions = await generator._build_casefold2_candidates(
                1, [("x", "y")], 1, 1.0
            )
        self.assertEqual(len(additions), 1)
        self.assertEqual(additions[0][0], ("a", "b"))

    async def test_casefold2_skips_states_without_transitions(self) -> None:
        matches = [
            ContextStateMatch(("a", "b"), "casefold", similarity=1.0, transition_count=3),
        ]
        generator = self._generator(matches)
        with patch.object(MarkovGenerator, "_get2", AsyncMock(return_value=[])):
            additions = await generator._build_casefold2_candidates(
                1, [("x", "y")], 1, 1.0
            )
        self.assertEqual(additions, [])


if __name__ == "__main__":
    unittest.main()
