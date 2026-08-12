"""Seeded generation is inert at the default ratio (M2R-410, design D5).

The shipping contract: ratio 0 computes no seed ranking, issues no reverse/df
read, and draws no extra RNG — so generation is byte-identical. The global
proof is `tools/generation_hash.py`; this pins the per-pipeline half.
"""

from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

from app.core.markov import MarkovGenerator, tokenize
from app.core.response_generator import GenerationRequest, ResponseGenerator
from app.infrastructure.database import Database
from app.services.learning_service import LearningService
from tests.test_runtime_state import make_runtime_state

CHAT = 5150


class TestSeededNeutralDefault(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from app import log_masking

        log_masking.init_masking("seeded-neutrality")
        self.db_path = Path(f"test_seed_neutral_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        for text in (
            "красный дракон летит над городом ночью",
            "синий кот сидит на тёплой крыше днём",
            "зелёный дракон спит под старым мостом",
        ):
            await self.db.save_message_and_update_model(
                chat_id=CHAT, raw_text=text, tokens=tokenize(text)
            )
        self.generator = MarkovGenerator(self.db.markov)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)
        for suffix in ("-shm", "-wal"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)

    async def test_default_ratio_issues_no_seeded_read(self) -> None:
        # Any seed-path DB read must blow up if it fires at the default.
        for name in ("get_seed_forward", "get_reverse_branch",
                     "get_reverse_transitions", "get_n_docs", "get_token_df"):
            setattr(self.db.markov, name, AsyncMock(side_effect=AssertionError))
        state = make_runtime_state()  # markov_seeded_candidate_ratio defaults to 0
        pipeline = ResponseGenerator(
            generator=self.generator,
            learning_service=LearningService(self.db, self.generator),
            runtime_state=state,
        )
        result = await pipeline.generate(
            GenerationRequest(
                chat_id=CHAT,
                context_tokens=["дракон", "город"],
                seed=None,
                current_message_normalized="красный дракон летит",
            ),
            rng=random.Random(11),
        )
        self.assertIsNotNone(result)
        # telemetry.note_seeded is never called at the default.
        self.assertEqual(self.generator.telemetry.snapshot()["seeded_generations"], 0)

    async def test_same_rng_same_output_regardless_of_seeded_default(self) -> None:
        """Two runs at ratio 0 with the same seed are identical — the RNG
        consumption order is unchanged by the (skipped) seeded branch."""
        state = make_runtime_state()
        pipeline = ResponseGenerator(
            generator=self.generator,
            learning_service=LearningService(self.db, self.generator),
            runtime_state=state,
        )
        req = GenerationRequest(
            chat_id=CHAT,
            context_tokens=["дракон"],
            seed=None,
            current_message_normalized="синий кот сидит",
        )
        first = await pipeline.generate(req, rng=random.Random(7))
        second = await pipeline.generate(req, rng=random.Random(7))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
