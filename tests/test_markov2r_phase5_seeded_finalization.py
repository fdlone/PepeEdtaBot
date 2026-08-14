"""Seeded candidates go through the same finalization as the main walk (M3R-101).

Before this, ``_append_seeded_candidates`` called ``detokenize`` straight after
assembly: no tail pipeline, no form gates. Measured cost (REVIEW §2, map §3.4):
terminal punctuation on 21% of seeded candidates against 91% of organic ones,
4.5% starting with punctuation, mean ``completion`` 0.036 against 0.315. The
scorer was marking the branch down for how it was built, not for what it said,
so the phase-5 promotion gate measured an unfinished implementation.

Two levels here: the shared pipeline itself, and the wiring that makes the
seeded branch use it.
"""

from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.core.markov import (
    REJECTION_CONTEXT_HEAVY,
    REJECTION_LOW_DIVERSITY,
    REJECTION_RESULT_TOO_SHORT,
    REJECTION_SHORT_CONTEXT_COPY,
    MarkovGenerator,
    finalize_candidate_tokens,
    tokenize,
)
from app.core.response_generator import GenerationRequest, ResponseGenerator
from app.infrastructure.database import Database
from app.services.learning_service import LearningService
from tests.test_runtime_state import make_runtime_state

CHAT = 5151


class TestSharedFinalization(unittest.IsolatedAsyncioTestCase):
    """The pipeline body: tail trims plus the four form gates."""

    def _finalize(self, tokens: list[str], context: list[str] | None = None):
        return finalize_candidate_tokens(
            tokens, max_chars=280, context_tokens=context or []
        )

    def test_ending_is_completed(self) -> None:
        final = self._finalize(["кот", "сидит", "на", "тёплой", "крыше"])
        self.assertIsNone(final.rejection_reason)
        self.assertTrue(
            final.text.endswith((".", "!", "?")),
            f"нет терминальной пунктуации: {final.text!r}",
        )

    def test_leading_punctuation_is_stripped_and_counted(self) -> None:
        final = self._finalize([",", "кот", "сидит", "на", "тёплой", "крыше"])
        self.assertIsNone(final.rejection_reason)
        self.assertFalse(final.text.startswith(","), final.text)
        self.assertEqual(final.leading_punctuation_stripped, 1)

    def test_gate_result_too_short(self) -> None:
        self.assertEqual(
            self._finalize(["да"]).rejection_reason, REJECTION_RESULT_TOO_SHORT
        )

    def test_gate_low_diversity(self) -> None:
        """Повтор в начале — единственная форма, доживающая до этого гейта.

        Хвостовой конвейер идёт первым, и ``trim_repetitive_tail`` срезает
        повтор в конце раньше, чем гейт его увидит. Проверено: ``["кот"] * 12``
        и «восемь одинаковых плюс четыре разных» доходят до гейта уже
        подрезанными и проходят. То есть гейт разнообразия ловит остаток —
        вырождение, которое не является хвостовым.
        """
        final = self._finalize(["кот"] * 5 + ["пёс", "волк", "лиса", "мышь"])
        self.assertEqual(final.rejection_reason, REJECTION_LOW_DIVERSITY)

    def test_gate_short_context_copy(self) -> None:
        context = tokenize("холодная зима пришла")
        final = self._finalize(tokenize("холодная зима пришла"), context)
        self.assertEqual(final.rejection_reason, REJECTION_SHORT_CONTEXT_COPY)

    def test_gate_context_heavy(self) -> None:
        context = tokenize("красный дракон летит над городом ночью и утром")
        final = self._finalize(
            tokenize("красный дракон летит над городом ночью и утром"), context
        )
        self.assertEqual(final.rejection_reason, REJECTION_CONTEXT_HEAVY)

    def test_rejection_carries_no_text(self) -> None:
        final = self._finalize(["да"])
        self.assertEqual(final.text, "")
        self.assertEqual(final.tokens, [])


class TestSeededBranchUsesIt(unittest.IsolatedAsyncioTestCase):
    """Wiring: the seeded branch runs assembled tokens through that body.

    The ratio is raised explicitly — at the shipping default of 0 the branch
    does not execute at all, so a green suite would say nothing about it.
    """

    async def asyncSetUp(self) -> None:
        from app import log_masking

        log_masking.init_masking("seeded-finalization")
        self.db_path = Path(f"test_seed_final_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        for text in (
            "красный дракон летит над городом ночью",
            "синий кот сидит на тёплой крыше днём",
            "зелёный дракон спит под старым мостом",
            "рыжий кот гуляет по мокрой дороге вечером",
        ):
            await self.db.save_message_and_update_model(
                chat_id=CHAT, raw_text=text, tokens=tokenize(text)
            )
        self.generator = MarkovGenerator(self.db.markov)
        self.state = make_runtime_state()
        self.state.markov_seeded_candidate_ratio = 1.0
        self.state.markov_seed_min_score = 0.0
        self.pipeline = ResponseGenerator(
            generator=self.generator,
            learning_service=LearningService(self.db, self.generator),
            runtime_state=self.state,
        )

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)
        for suffix in ("-shm", "-wal"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)

    async def _seeded_texts(self, assembled: list[str]) -> set[str]:
        """Run the seeded branch with a fixed assembly and collect its texts."""
        from app.core.markov import EntropySampling
        from app.core.seed import SeedScore
        from app.core.temporal import TemporalBlend

        ranked = [
            SeedScore(
                token="дракон",
                normalized_idf=1.0,
                support=1.0,
                branching=2.0,
                score=1.0,
            )
        ]
        # Патчится класс, а не экземпляр: MarkovGenerator слотирован.
        with (
            patch.object(MarkovGenerator, "rank_seeds", AsyncMock(return_value=ranked)),
            patch.object(
                MarkovGenerator,
                "generate_seeded_candidate",
                AsyncMock(return_value=assembled),
            ),
        ):
            return await self.pipeline._append_seeded_candidates(
                GenerationRequest(
                    chat_id=CHAT,
                    context_tokens=["дракон"],
                    seed=None,
                    current_message_normalized="красный дракон летит",
                ),
                [],
                set(),
                length_mode="medium",
                max_tokens=24,
                context_idf={},
                recent_trigrams=set(),
                recent_penalty_strength=0.0,
                corpus_ngrams=frozenset(),
                verbatim_penalty_strength=0.0,
                active_collocations=frozenset(),
                entropy_sampling=EntropySampling(),
                temporal_blend=TemporalBlend(),
                now=0,
                target=5,
                rng=random.Random(3),
            )

    async def test_assembled_candidate_is_finalized(self) -> None:
        texts = await self._seeded_texts(
            [",", "кот", "сидит", "на", "тёплой", "крыше"]
        )
        self.assertEqual(len(texts), 1)
        text = next(iter(texts))
        self.assertFalse(text.startswith(","), f"ведущая пунктуация уцелела: {text!r}")
        self.assertTrue(
            text.endswith((".", "!", "?")), f"нет терминальной пунктуации: {text!r}"
        )

    async def test_malformed_candidate_is_rejected_not_scored(self) -> None:
        """A form gate keeps it out of the pool instead of scoring it low."""
        malformed = ["кот"] * 5 + ["пёс", "волк", "лиса", "мышь"]
        self.assertEqual(await self._seeded_texts(malformed), set())


if __name__ == "__main__":
    unittest.main()
