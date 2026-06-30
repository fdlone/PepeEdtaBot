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

from app.core.markov import MarkovGenerator
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

    async def test_global_start_backoff_to_order1_seed42(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            randomness_strength=0.0, rng=random.Random(42), emit_start=True,
        )
        self.assertEqual(
            attempt.text,
            "папа чинил машину в гараже целый выходной день до самого вечера.",
        )
        self.assertEqual(attempt.markov_order_used, 1)
        self.assertEqual(attempt.token_count, 12)
        self.assertEqual(attempt.start_source, "global")

    async def test_global_start_backoff_to_order1_seed123(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            randomness_strength=0.0, rng=random.Random(123), emit_start=True,
        )
        self.assertEqual(
            attempt.text,
            "кошка спала на диване весь длинный зимний день до самого вечера.",
        )
        self.assertEqual(attempt.markov_order_used, 1)
        self.assertEqual(attempt.token_count, 12)

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
            attempt.text, "в гараже целый выходной день до самого вечера."
        )
        self.assertEqual(attempt.start_source, "hidden_context")
        self.assertEqual(attempt.context_exact_matches, 1)
        self.assertEqual(attempt.context_casefold_matches, 0)
        self.assertEqual(attempt.markov_order_used, 1)

    async def test_contextual_start_casefold_match(self) -> None:
        attempt = await self.generator._generate_text_once(
            chat_id=CHAT_ID, max_chars=200, max_tokens=12,
            context_tokens=["ПАПА", "ЧИНИЛ", "МАШИНУ"],
            context_start_bias=4.0, randomness_strength=0.0,
            fuzzy_context_casefold=True, rng=random.Random(3), emit_start=True,
        )
        self.assertEqual(
            attempt.text, "в гараже целый выходной день до самого вечера."
        )
        self.assertEqual(attempt.start_source, "hidden_context")
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
            "папа чинил машину в гараже целый выходной день до самого вечера.",
        )
        self.assertEqual(attempt.markov_order_used, 1)
        self.assertEqual(attempt.token_count, 12)


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


if __name__ == "__main__":
    unittest.main()
