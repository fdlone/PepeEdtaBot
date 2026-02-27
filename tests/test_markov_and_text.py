from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path

from db import Database
from markov import MarkovGenerator, detokenize, tokenize
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
