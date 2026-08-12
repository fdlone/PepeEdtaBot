"""Bidirectional seeded assembly (M2R-410, TZ §9.5).

The anchor must be able to sit mid-reply: the tail grows forward and the head
grows backward on the reverse index. These tests build a chain by hand so the
predecessor/successor of the seed is known, then assert the assembled candidate
places the anchor between them.
"""

from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path

from app.core.markov import MarkovGenerator, tokenize
from app.infrastructure.database import Database

CHAT = 7777


class TestSeededAssembly(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_seeded_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        self.generator = MarkovGenerator(self.db.markov)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)
        for suffix in ("-shm", "-wal"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)

    async def _learn(self, *texts: str) -> None:
        for text in texts:
            await self.db.save_message_and_update_model(
                chat_id=CHAT, raw_text=text, tokens=tokenize(text)
            )

    async def _assemble(self, seed: str, **kw: object) -> list[str] | None:
        params = dict(
            max_tokens=12, head_share=0.5, next_explore=0.0, next_power=1.0,
            repetition_penalty_strength=1.0, rng=random.Random(3),
        )
        params.update(kw)
        return await self.generator.generate_seeded_candidate(CHAT, seed, **params)

    async def test_anchor_lands_between_head_and_tail(self) -> None:
        # One rigid path: the only way through the chain is the sentence itself.
        await self._learn("красный дракон летит над городом ночью")
        result = await self._assemble("летит")
        assert result is not None
        self.assertIn("летит", result)
        idx = result.index("летит")
        # both a predecessor (head) and a successor (tail) were assembled
        self.assertGreater(idx, 0, f"no head grew: {result}")
        self.assertLess(idx, len(result) - 1, f"no tail grew: {result}")
        # and the neighbours are the chain's real neighbours of the seed
        self.assertEqual(result[idx - 1], "дракон")
        self.assertEqual(result[idx + 1], "над")

    async def test_seed_without_forward_continuation_returns_none(self) -> None:
        # "ночью" is only ever the last token — no forward continuation to
        # bootstrap the tail from.
        await self._learn("красный дракон летит над городом ночью")
        self.assertIsNone(await self._assemble("ночью"))

    async def test_head_stops_when_reverse_pool_is_empty(self) -> None:
        # "красный" is only ever the first token — nothing precedes it, so the
        # head cannot grow, but the tail still assembles the candidate.
        await self._learn("красный дракон летит над городом ночью")
        result = await self._assemble("красный", head_share=0.5)
        assert result is not None
        self.assertEqual(result[0], "красный")  # anchor stayed at the start
        self.assertGreater(len(result), 1)       # tail grew

    async def test_reverse_walk_distribution_follows_the_chain(self) -> None:
        # Two messages share the tail "X ловит рыбу" with different heads, so
        # the predecessor of the pair (ловит, рыбу) is "кот" twice vs "пёс"
        # once. Over many seeds the head should prefer "кот".
        await self._learn(
            "кот ловит рыбу быстро", "кот ловит рыбу ловко", "пёс ловит рыбу молча"
        )
        heads: list[str] = []
        for seed in range(60):
            result = await self.generator.generate_seeded_candidate(
                CHAT, "ловит", max_tokens=6, head_share=0.5,
                next_explore=0.0, next_power=1.0,
                repetition_penalty_strength=1.0, rng=random.Random(seed),
            )
            if result and "ловит" in result:
                i = result.index("ловит")
                if i > 0:
                    heads.append(result[i - 1])
        self.assertTrue(heads, "no head ever grew")
        self.assertGreater(heads.count("кот"), heads.count("пёс"))


if __name__ == "__main__":
    unittest.main()
