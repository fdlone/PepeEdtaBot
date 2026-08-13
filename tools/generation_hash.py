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

    python -m tools.generation_hash                     # prod copy, 4 seeds x 250
    python -m tools.generation_hash --per-seed 100      # quicker smoke run
    python -m tools.generation_hash --synthetic --check # CI guard vs the record

On the prod copy the hash is meaningless in isolation — it only means something
compared with the hash of another revision produced by the same command. On the
synthetic snapshot it is compared for you, against
``tools/generation_hash_baseline.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
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
from tools.eval.synthetic import build_synthetic_snapshot  # noqa: E402
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

# The recorded baseline. It is anchored on the SYNTHETIC snapshot, not on the
# prod copy: the prod copy never leaves the owner's machine, so a value
# recorded against it is checkable on exactly one machine — which is how the
# constant quoted in three documents drifted from 5a72e2d4 to 13e496c0 without
# anyone noticing. The synthetic corpus is reproducible everywhere, so the
# record is checkable in CI on the commit that moves it.
BASELINE_PATH = PROJECT_ROOT / "tools" / "generation_hash_baseline.json"
SYNTHETIC_LABEL = "synthetic"


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


async def run_synthetic(seeds: tuple[int, ...], per_seed: int) -> str:
    """Same harness over the reproducible synthetic corpus (doc 05 §7)."""
    db_path, temp_dir = await build_synthetic_snapshot()
    try:
        return await run(db_path, seeds, per_seed)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def check_against_baseline(
    digest: str, snapshot: str, seeds: tuple[int, ...], per_seed: int
) -> int:
    """Compare a computed hash with the record; return a process exit code.

    A hash is only comparable against a record produced by the same corpus and
    the same draw budget, so a run parameterised differently refuses to compare
    rather than reporting a mismatch nobody should act on.
    """
    if not BASELINE_PATH.exists():
        print(f"no baseline record at {BASELINE_PATH} — nothing to compare")
        return 0
    record = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if record.get("snapshot") != snapshot:
        print(
            f"no baseline recorded for snapshot {snapshot!r} "
            f"(the record covers {record.get('snapshot')!r}) — nothing to compare"
        )
        return 0
    if tuple(record.get("seeds", ())) != seeds or record.get("per_seed") != per_seed:
        print(
            "BASELINE NOT COMPARABLE: the record was computed with "
            f"seeds={record.get('seeds')} per_seed={record.get('per_seed')}, "
            f"this run used seeds={list(seeds)} per_seed={per_seed}. "
            "Re-run with the recorded parameters."
        )
        return 1
    if digest == record.get("hash"):
        print(f"baseline OK: matches the record for snapshot {snapshot!r}")
        return 0
    print(
        "BASELINE MISMATCH — generation output moved.\n"
        f"  recorded: {record.get('hash')}\n"
        f"  computed: {digest}\n"
        f"  record set by: {record.get('revision')} — {record.get('reason')}\n"
        f"If the move is intended, update {BASELINE_PATH.name} and state which "
        "change moved it and why."
    )
    return 1


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
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run on the reproducible synthetic snapshot instead of --db",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "compare the result with the recorded baseline and exit non-zero "
            "on a mismatch (the record is anchored on --synthetic)"
        ),
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if args.synthetic:
        digest = asyncio.run(run_synthetic(seeds, args.per_seed))
        snapshot = SYNTHETIC_LABEL
    else:
        digest = asyncio.run(run(args.db, seeds, args.per_seed))
        snapshot = str(args.db)
    print(f"SHA256 {digest}")
    if args.check:
        raise SystemExit(check_against_baseline(digest, snapshot, seeds, args.per_seed))


if __name__ == "__main__":
    main()
