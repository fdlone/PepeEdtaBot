"""Byte-identical generation hash — proof that a refactor changed nothing.

A refactor of the generation hot path is easy to *claim* behaviour-neutral and
hard to show: the pipeline draws from an RNG dozens of times per reply, so any
reordering of those draws diverges on the first generation while still looking
plausible. This harness pins exactly that: it replays a fixed number of
generations through the real ``ResponseGenerator`` against the prod copy and
prints a SHA256 of the concatenated replies.

Same code, same DB, same seeds -> same hash. A differing hash means the
refactor moved behaviour, wherever it claims not to.

Precedent: PR #102 verified the ``generate_with_result`` extraction this way
(1000 generations, 4 seeds x 250) but the script itself was ad hoc and never
committed, so the next refactor had to reinvent it. This one is committed.

Usage::

    python tools/generation_hash.py                 # 4 seeds x 250
    python tools/generation_hash.py --per-seed 100  # quicker smoke run

The hash is meaningless in isolation — it only means something compared with
the hash of another revision produced by the same command.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import log_masking  # noqa: E402
from app.config.registry import RUNTIME_FIELDS  # noqa: E402
from app.core.markov import content_tokens, tokenize  # noqa: E402
from app.core.response_generator import (  # noqa: E402
    CANDIDATE_TARGET,
    GenerationRequest,
    ResponseGenerator,
)
from app.core.text import sanitize_text  # noqa: E402
from app.infrastructure.database import Database  # noqa: E402
from tools.eval_prod import (  # noqa: E402
    VERBATIM_MIN_N,
    _ProdVerbatimChecker,
    _TraceCapturingGenerator,
    build_verbatim_index,
    copy_database,
    load_messages,
    pick_chat_id,
)

DEFAULT_DB = PROJECT_ROOT / "db_prod_copy" / "markov.db"
DEFAULT_SEEDS = (101, 102, 103, 104)
DEFAULT_PER_SEED = 250


def _runtime_state() -> SimpleNamespace:
    """Registry defaults — the configuration the bot actually ships with.

    Deliberately not a hand-tuned config: the hash must pin the live default
    path, so that a refactor is checked against what production runs.
    """
    state = SimpleNamespace(
        **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS}
    )
    # Channels that would inject non-generation randomness (emoji picks, flavor
    # rolls) are silenced: they add noise to the hash without exercising the
    # walk itself.
    state.reply_flavor_strength = 0.0
    state.emoji_append_chance = 0.0
    state.recent_short_replies = {}
    state.recent_replies = {}
    return state


async def run(db_path: Path, seeds: tuple[int, ...], per_seed: int) -> str:
    log_masking.init_masking("generation-hash-harness")
    db_copy, _temp_dir = copy_database(db_path)
    db: Database | None = None
    try:
        chat = pick_chat_id(db_copy, None)
        messages = load_messages(db_copy, chat)
        verbatim_index = build_verbatim_index(messages)
        db = Database(str(db_copy))
        await db.init()
        alltime = frozenset(tuple(row) for row in await db.get_verbatim_ngrams(chat))
        verbatim_index[VERBATIM_MIN_N] |= set(alltime)

        generator = _TraceCapturingGenerator(db.markov)
        state = _runtime_state()
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_ProdVerbatimChecker(
                messages, ngram_index=alltime, db=db
            ),
            runtime_state=state,
        )
        pool = [m for m in messages if len(content_tokens(tokenize(m))) >= 3]
        if not pool:
            raise SystemExit("prod copy has no usable messages")

        digest = hashlib.sha256()
        produced = 0
        for seed in seeds:
            sampler = random.Random(seed)
            for index in range(per_seed):
                source = sampler.choice(pool)
                context = content_tokens(tokenize(source))[
                    : state.reply_context_max_tokens
                ]
                generator.reset_generation()
                result = await response_generator.generate_with_result(
                    GenerationRequest(
                        chat_id=chat,
                        context_tokens=context,
                        seed=None,
                        current_message_normalized=sanitize_text(source).lower(),
                    ),
                    # A fresh RNG per generation keeps runs paired across
                    # revisions: the hash then pins the *order* in which the
                    # pipeline consumes it, which is what a refactor breaks.
                    rng=random.Random(seed * 100_000 + index),
                    candidate_target=CANDIDATE_TARGET,
                )
                digest.update((result.text or "").encode("utf-8"))
                digest.update(b"\n")
                produced += 1
        print(f"generations: {produced} ({len(seeds)} seeds x {per_seed})")
        return digest.hexdigest()
    finally:
        if db is not None:
            await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--per-seed", type=int, default=DEFAULT_PER_SEED)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    args = parser.parse_args()
    digest = asyncio.run(run(args.db, tuple(args.seeds), args.per_seed))
    print(f"SHA256 {digest}")


if __name__ == "__main__":
    main()
