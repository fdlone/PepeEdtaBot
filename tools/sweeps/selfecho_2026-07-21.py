"""Targeted self-echo measurement for the reply-to-own-message fix.

Motivation: eval_prod builds context from a single real corpus message and
never models the "human replies to the bot" chain, so the generic sweep is
blind to the extract_context_tokens fix. This harness reproduces the loop:

    bot A  ->  human H (reply to A)  ->  bot B (reply to H)

For each trial it generates bot A from a real corpus message, then generates
bot B twice from the SAME rng, differing only in how the reply-context is
assembled — exactly the two branches of the fixed extract_context_tokens:

  BEFORE (pre-fix):  reply_to A is treated as NOT the bot -> context = [A, H]
  AFTER  (fixed):    reply_to A is the bot's own          -> context = [H]

Both branches call the REAL fixed extract_context_tokens; the only difference
is the bot_id passed, which is faithful (pre-fix == fixed when reply_to is not
the bot). Metrics isolate the self-echo:

  echo_run_vs_A     mean longest contiguous content run B shares with bot A
  prefix_in_A_rate  share of B whose opening <=3 content tokens sit inside A
  overlap_vs_A      context_token_overlap(B, A)  -- self-echo magnitude
  overlap_vs_H      context_token_overlap(B, H)  -- topicality on the RIGHT msg
  start_ctx_rate    share of winning walks that started on a context anchor

Run on the prod-copy DB. Fresh Random(seed) per trial keeps BEFORE/AFTER paired.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app import log_masking  # noqa: E402
from app.config.registry import RUNTIME_FIELDS  # noqa: E402
from app.core.markov import (  # noqa: E402
    content_tokens,
    longest_shared_run,
    tokenize,
)
from app.core.response_generator import (  # noqa: E402
    CANDIDATE_TARGET,
    GenerationRequest,
    ResponseGenerator,
)
from app.core.text import sanitize_text  # noqa: E402
from app.handlers.learning import extract_context_tokens  # noqa: E402
from app.infrastructure.database import Database  # noqa: E402
from tools.eval_generation import context_token_overlap  # noqa: E402
from tools.eval_prod import (  # noqa: E402
    VERBATIM_MIN_N,
    _ProdVerbatimChecker,
    _TraceCapturingGenerator,
    build_verbatim_index,
    copy_database,
    load_messages,
    pick_chat_id,
)

DB = PROJECT_ROOT / "db_prod_copy" / "markov.db"
BOT_ID = 424242  # the id we stamp on bot A's message
HUMAN_ID = 111  # a distinct id used to reproduce the pre-fix branch


def _context(a_text: str, h_text: str, *, drop_bot: bool, state) -> list[str]:
    """Assemble reply-context via the REAL fixed function.

    drop_bot=True  -> pass bot_id=BOT_ID so A is recognized as the bot's own
                      and dropped (the fixed behaviour).
    drop_bot=False -> pass a non-matching bot_id so A is kept (pre-fix).
    """
    reply_to = SimpleNamespace(text=a_text, from_user=SimpleNamespace(id=BOT_ID))
    message = SimpleNamespace(reply_to_message=reply_to)
    return extract_context_tokens(
        message=message,
        current_text=h_text,
        normalize_lower=state.normalize_lower,
        max_tokens=state.reply_context_max_tokens,
        only_for_replies=state.reply_context_only_for_replies,
        include_current_message=state.reply_context_include_current_message,
        bot_id=BOT_ID if drop_bot else HUMAN_ID,
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--trials", type=int, default=400)
    args = ap.parse_args()

    log_masking.init_masking("self-echo-measurement")
    db_copy, temp_dir = copy_database(DB)
    db: Database | None = None
    try:
        chat = pick_chat_id(db_copy, None)
        messages = load_messages(db_copy, chat)
        verbatim_index = build_verbatim_index(messages)
        db = Database(str(db_copy))
        await db.init()
        alltime = frozenset(
            tuple(row) for row in await db.get_verbatim_ngrams(chat)
        )
        verbatim_index[VERBATIM_MIN_N] |= set(alltime)
        generator = _TraceCapturingGenerator(db)
        state = SimpleNamespace(
            **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS}
        )
        # live context+chaos package, matching the sweep baseline
        state.context_jump_boost = 2.0
        state.verbatim_extension_share = 0.8
        state.order_mix_probability = 0.5
        state.reply_flavor_strength = 0.0
        state.emoji_append_chance = 0.0
        state.recent_short_replies = {}
        state.recent_replies = {}
        rg = ResponseGenerator(
            generator=generator,
            learning_service=_ProdVerbatimChecker(
                messages, ngram_index=alltime, db=db
            ),
            runtime_state=state,
        )
        pool = [m for m in messages if len(content_tokens(tokenize(m))) >= 3]
        sampler = random.Random(args.seed)

        agg = {
            "before": {"echo": [], "ge3": 0, "ov_a": [], "ov_h": [], "ctx": 0},
            "after": {"echo": [], "ge3": 0, "ov_a": [], "ov_h": [], "ctx": 0},
        }
        counted = 0

        async def gen(ctx_tokens: list[str], trial_seed: int, current: str):
            generator.reset_generation()
            res = await rg.generate_with_result(
                GenerationRequest(
                    chat_id=chat,
                    context_tokens=ctx_tokens,
                    seed=None,
                    current_message_normalized=sanitize_text(current).lower(),
                ),
                rng=random.Random(trial_seed),
                candidate_target=CANDIDATE_TARGET,
            )
            src = generator.attempt_sources.get(res.text or "", "unknown")
            return res.text, src

        for t in range(args.trials):
            x = sampler.choice(pool)
            trial_seed = sampler.randrange(1 << 30)
            a_text, _ = await gen(tokenize(x), trial_seed, x)
            if not a_text:
                continue
            h = sampler.choice(pool)
            content_a = content_tokens(tokenize(a_text))
            content_h = content_tokens(tokenize(h))
            if len(content_a) < 3:
                continue
            # Same rng seed for both branches -> BEFORE/AFTER differ only in the
            # assembled context, not in sampling luck (paired comparison).
            b_seed = sampler.randrange(1 << 30)

            for label, drop in (("before", False), ("after", True)):
                ctx = _context(a_text, h, drop_bot=drop, state=state)
                b_text, src = await gen(ctx, b_seed, h)
                if not b_text:
                    continue
                content_b = content_tokens(tokenize(b_text))
                if not content_b:
                    continue
                echo = longest_shared_run(content_b, content_a)
                d = agg[label]
                d["echo"].append(echo)
                d["ge3"] += 1 if echo >= 3 else 0
                d["ov_a"].append(context_token_overlap(content_b, content_a))
                d["ov_h"].append(context_token_overlap(content_b, content_h))
                d["ctx"] += 1 if src in ("context", "hidden_context", "context_spliced") else 0
            counted += 1
    finally:
        if db is not None:
            await db.close()
        temp_dir.cleanup()

    def m(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"seed={args.seed} trials_counted={counted}\n")
    print(f"{'metric':22s} {'BEFORE':>10s} {'AFTER':>10s} {'delta':>10s}")
    rows = [
        ("echo_run_vs_A (tok)", "echo", m),
        ("overlap_vs_A", "ov_a", m),
        ("overlap_vs_H", "ov_h", m),
    ]
    for name, key, fn in rows:
        b, a = fn(agg["before"][key]), fn(agg["after"][key])
        print(f"{name:22s} {b:10.4f} {a:10.4f} {a - b:+10.4f}")
    for name, key in (("echo_ge3_rate", "ge3"), ("start_ctx_rate", "ctx")):
        b = agg["before"][key] / counted if counted else 0.0
        a = agg["after"][key] / counted if counted else 0.0
        print(f"{name:22s} {b:10.4f} {a:10.4f} {a - b:+10.4f}")


if __name__ == "__main__":
    asyncio.run(main())
