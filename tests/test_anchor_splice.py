"""Mechanism tests for anchor segmentation (2026-07-16):

``context_anchor_splice_probability`` defers a *visible* contextual anchor —
the walk starts from a global start and the anchor's emission tokens are
spliced in later (connective + chain continues from the anchor state), so the
context surfaces mid- or end-reply. The deferred anchor shares the
one-jump-per-reply budget with M4 drift: a pending anchor suppresses global
jumps and its splice consumes the slot.
"""
from __future__ import annotations

import random
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.core.markov import MarkovGenerator, tokenize
from app.infrastructure.database import Database

_ANCHOR_STATE = ("пробное", "сообщение", "чата")
_CONTEXT = ["пробное", "сообщение", "чата"]


class _AnchorSpliceBase(unittest.IsolatedAsyncioTestCase):
    corpus: tuple[str, ...] = ()

    async def asyncSetUp(self) -> None:
        self.db_path = Path(f"test_anchor_{uuid.uuid4().hex}.sqlite")
        self.db = Database(str(self.db_path))
        await self.db.init()
        for text in self.corpus:
            await self.db.save_message_and_update_model(
                chat_id=1, raw_text=text, tokens=tokenize(text)
            )
        self.generator = MarkovGenerator(self.db.markov)

    async def asyncTearDown(self) -> None:
        await self.db.close()
        self.db_path.unlink(missing_ok=True)

    async def _attempt(
        self,
        *,
        splice_probability: float,
        anchor_state: tuple[str, str, str] = _ANCHOR_STATE,
        jump_probability: float = 0.0,
        max_tokens: int = 20,
        seed: int = 3,
    ):
        contextual_state = SimpleNamespace(
            state=anchor_state, order=3, match_kind="exact"
        )
        with patch.object(
            MarkovGenerator,
            "_pick_contextual_start",
            AsyncMock(return_value=contextual_state),
        ):
            return await self.generator._generate_text_once(
                chat_id=1,
                max_chars=280,
                max_tokens=max_tokens,
                context_tokens=_CONTEXT,
                context_start_bias=4.0,
                jump_probability=jump_probability,
                context_anchor_splice_probability=splice_probability,
                rng=random.Random(seed),
            )


class TestAnchorSplice(_AnchorSpliceBase):
    # One long single-path chain (global start material) plus a message where
    # the anchor state sits mid-sentence, so the anchor has a continuation to
    # walk after the splice (and the knob-off reply is longer than the
    # short_context_copy gate).
    corpus = (
        "один два три четыре пять шесть семь восемь девять десять конец",
        "ну вот пробное сообщение чата растёт дальше вширь и вглубь",
    )

    async def test_knob_off_keeps_anchor_at_start(self) -> None:
        attempt = await self._attempt(splice_probability=0.0)
        self.assertEqual(attempt.start_source, "context")
        self.assertTrue(attempt.text.startswith("сообщение чата"))

    async def test_deferred_anchor_is_spliced_later(self) -> None:
        attempt = await self._attempt(splice_probability=1.0)
        self.assertEqual(attempt.start_source, "context_spliced")
        self.assertEqual(attempt.jump_count, 1)
        self.assertIn("сообщение чата", attempt.text)
        # The reply opens on a global start, never on the anchor's emission.
        self.assertFalse(attempt.text.startswith("сообщение чата"))
        self.assertEqual(attempt.context_exact_matches, 1)

    async def test_pending_anchor_suppresses_global_jumps(self) -> None:
        # With a certain per-step jump the reply would normally drift once;
        # the pending anchor must own that budget instead.
        attempt = await self._attempt(
            splice_probability=1.0, jump_probability=1.0
        )
        self.assertEqual(attempt.jump_count, 1)
        self.assertEqual(attempt.start_source, "context_spliced")
        self.assertIn("сообщение чата", attempt.text)

    async def test_hidden_anchor_is_never_deferred(self) -> None:
        # An all-stopword window has an empty emission: nothing to splice, so
        # the deferral roll is skipped and the start stays hidden.
        attempt = await self._attempt(
            splice_probability=1.0, anchor_state=("ну", "и", "на")
        )
        self.assertEqual(attempt.start_source, "hidden_context")


class TestAnchorSpliceAtDeadEnd(_AnchorSpliceBase):
    # The only global chain dead-ends after four tokens — before any rollable
    # splice position — so the anchor must be spliced at the dead end.
    corpus = ("первое пробное сообщение чата",)

    async def test_dead_end_splices_the_anchor(self) -> None:
        attempt = await self._attempt(splice_probability=1.0)
        self.assertEqual(attempt.start_source, "context_spliced")
        self.assertEqual(attempt.jump_count, 1)


class TestAnchorNotSpliced(_AnchorSpliceBase):
    corpus = (
        "один два три четыре пять шесть семь восемь девять десять конец",
    )

    async def test_token_limit_before_target_reports_no_splice(self) -> None:
        # Direct loop call: a target beyond the token budget is never reached,
        # the walk exits on the limit and reports the anchor as not spliced.
        starts3 = await self.db.markov.get_starts3(1)
        generated, _, jump_count, anchor_spliced = (
            await self.generator._run_generation_loop(
                    1,
                    ("один", "два", "три"),
                    emit_tokens=["один", "два", "три"],
                    order=3,
                    order_used=3,
                    max_tokens=6,
                    max_chars=280,
                    enable_backoff=True,
                    starts3=starts3,
                    context_token_set=set(),
                    context_pairs=set(),
                    context_triplets=set(),
                    start_explore=0.2,
                    start_power=0.7,
                    next_explore=0.2,
                    next_power=0.7,
                    context_bias=1.0,
                    repetition_penalty_strength=1.0,
                    jump_probability=0.0,
                    order_mix_probability=0.0,
                    anchor_state=_ANCHOR_STATE,
                    anchor_emit_tokens=["сообщение", "чата"],
                    anchor_target_tokens=50,
                    rng=random.Random(1),
                )
            )
        self.assertFalse(anchor_spliced)
        self.assertEqual(jump_count, 0)
        self.assertEqual(len(generated), 6)


if __name__ == "__main__":
    unittest.main()
