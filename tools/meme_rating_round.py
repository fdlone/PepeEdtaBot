"""Prepare one manual meme-rating round (docs/MEME_RATING_ROUND.md, doc 05 §5).

Produces two local files in ``rating_rounds/<label>/``:

- ``rating_list.txt`` — one numbered, shuffled, source-blind list mixing the
  top-20 by ``meme_score``, the top-20 by the current frequency selection
  (the hidden control) and 5 random above-threshold decoys. This is what gets
  handed to raters.
- ``rating_key.json`` — position -> source(s). Stays with the owner; without
  it the blind list cannot be scored, with it it stops being blind.

Everything here is verbatim private chat content, which is why the output
folder is gitignored and nothing below ever prints a phrase to stdout.

The DB is copied first (same pattern as tools/generation_hash.py): the
analyzer pass writes the registry, and a preparation tool must not mutate
the checked-in prod copy.

Usage:
    python -m tools.meme_rating_round --db db_prod_copy/markov.db --label 2026-08-12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import log_masking  # noqa: E402
from app.infrastructure.database import Database  # noqa: E402
from app.services.meme_analyzer import MemeSettings, analyze_chat_memes  # noqa: E402
from tools.eval_prod import copy_database, pick_chat_id  # noqa: E402

MEME_TOP = 20
CONTROL_TOP = 20
DECOYS = 5

HEADER = """\
Раунд разметки повторяющихся фраз чата (docs/MEME_RATING_ROUND.md).

На каждую позицию — одна метка:
  м     — настоящий локальный мем этого чата
  ч     — просто частое сочетание, мемом не является
  мусор — шум, обрывок, случайная пара

Ответ одной строкой, например:  1:м 2:ч 3:мусор 4:м ...

"""


async def prepare_round(
    db_source: Path, chat_id: int | None, out_dir: Path, seed: int
) -> None:
    log_masking.init_masking("meme-rating-round")
    db_copy, temp_dir = copy_database(db_source)
    try:
        resolved_chat = pick_chat_id(db_copy, chat_id)
        db = Database(str(db_copy))
        await db.init()
        try:
            conn = await db._get_conn()
            cursor = await conn.execute(
                "SELECT MAX(last_seen) FROM transitions WHERE chat_id = ?",
                (resolved_chat,),
            )
            row = await cursor.fetchone()
            # The corpus's own last moment, like the eval runner: wall-clock
            # "now" on an old snapshot would read the whole registry as
            # decayed to nothing. NULL means a pre-temporal snapshot.
            moment = int(row[0]) if row and row[0] is not None else int(time.time())

            settings = MemeSettings()
            await analyze_chat_memes(
                db.collocations,
                resolved_chat,
                now=moment,
                min_joint_count=settings.min_joint_count,
                min_support=settings.min_support,
                recency_days=settings.recency_days,
                max_entries=settings.max_entries,
            )

            ranked = await db.collocations.get_ranked(resolved_chat, MEME_TOP)
            meme_phrases = [f"{left} {right}" for left, right, _, _ in ranked]

            # The hidden control is the frequency selection this phase claims
            # to beat: first whatever the live hot-ngram window offers, then —
            # because on an aged snapshot the 7-day window has decayed to
            # nothing — the same pair universe the meme ranking scored,
            # ordered by raw joint count. Association vs frequency over the
            # same objects is exactly the comparison doc 05 §5 asks for.
            hot = await db.chat_hot_ngrams.get_hot(
                resolved_chat,
                min_count=settings.min_joint_count,
                recency_share=0.5,
                limit=CONTROL_TOP,
            )
            control_phrases = [" ".join(ngram) for ngram in hot]

            pairs = await db.collocations.read_pair_counts(
                resolved_chat, min_joint_count=settings.min_joint_count
            )
            if len(control_phrases) < CONTROL_TOP:
                by_count = sorted(pairs, key=lambda item: -item[2])
                for left, right, _, _ in by_count:
                    phrase = f"{left} {right}"
                    if phrase not in control_phrases:
                        control_phrases.append(phrase)
                    if len(control_phrases) >= CONTROL_TOP:
                        break
            taken = set(meme_phrases) | set(control_phrases)
            decoy_pool = [
                f"{left} {right}"
                for left, right, _, _ in pairs
                if f"{left} {right}" not in taken
            ]
            rng = random.Random(seed)
            decoy_phrases = rng.sample(decoy_pool, min(DECOYS, len(decoy_pool)))
        finally:
            await db.close()
    finally:
        temp_dir.cleanup()

    # One entry per distinct phrase; a phrase in both tops is shown once and
    # credited to both sources (docs/MEME_RATING_ROUND.md).
    sources: dict[str, list[str]] = {}
    for phrase in meme_phrases:
        sources.setdefault(phrase, []).append("meme")
    for phrase in control_phrases:
        sources.setdefault(phrase, []).append("control")
    for phrase in decoy_phrases:
        sources.setdefault(phrase, []).append("decoy")

    order = sorted(sources)
    rng.shuffle(order)

    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "rating_list.txt"
    key_path = out_dir / "rating_key.json"
    list_path.write_text(
        HEADER
        + "\n".join(
            f"{position}. {phrase}" for position, phrase in enumerate(order, 1)
        )
        + "\n",
        encoding="utf-8",
    )
    key_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "evaluation_moment": moment,
                "counts": {
                    "meme": len(meme_phrases),
                    "control": len(control_phrases),
                    "decoy": len(decoy_phrases),
                    "positions": len(order),
                },
                "positions": {
                    str(position): sources[phrase]
                    for position, phrase in enumerate(order, 1)
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # Counts only — the phrases themselves stay in the files.
    print(f"chat: {log_masking.mask_chat_id(resolved_chat)}")
    print(
        f"positions: {len(order)} "
        f"(meme {len(meme_phrases)}, control {len(control_phrases)}, "
        f"decoys {len(decoy_phrases)}; overlaps shown once)"
    )
    print(f"list: {list_path}")
    print(f"key : {key_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=str, default="db_prod_copy/markov.db")
    parser.add_argument("--chat-id", type=int, default=None)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out_dir = PROJECT_ROOT / "rating_rounds" / args.label
    asyncio.run(
        prepare_round(Path(args.db), args.chat_id, out_dir, args.seed)
    )


if __name__ == "__main__":
    main()
