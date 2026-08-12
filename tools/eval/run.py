"""Protocol run orchestration (doc 05 §1.3, §2).

Per configuration x per seed x per category loop over the fixed prompt set,
reusing the ``tools/eval_prod.py`` machinery (DB copying, trace-capturing
generator, verbatim checker). Reproducibility contract (spec):

- prompt selection is driven by ``random.Random(seed)`` per category;
- each generation gets ``random.Random(seed * 100_000 + index)`` — the
  ``tools/generation_hash.py`` pattern that pins RNG-consumption order;
- configurations whose resolved overrides are identical to an earlier one
  (CF while no V2 feature exists) share its records instead of re-running —
  the report says so explicitly.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import log_masking  # noqa: E402
from app.config.registry import RUNTIME_FIELDS  # noqa: E402
from app.core import gen_trace_log  # noqa: E402
from app.core.markov import content_tokens, tokenize  # noqa: E402
from app.core.response_generator import (  # noqa: E402
    CANDIDATE_TARGET,
    GenerationRequest,
    ResponseGenerator,
)
from app.core.text import sanitize_text  # noqa: E402
from app.infrastructure.database import Database  # noqa: E402
from app.services.learning_service import normalize_for_verbatim  # noqa: E402
from app.services.meme_analyzer import (  # noqa: E402
    MemeSettings,
    analyze_chat_memes,
)
from tools.eval_prod import (  # noqa: E402
    _ProdVerbatimChecker,
    _TraceCapturingGenerator,
    copy_database,
    load_messages,
    pick_chat_id,
)

from .config import MatrixConfig, resolve_overrides  # noqa: E402
from .metrics import (  # noqa: E402
    COPY_RUN_TOKENS,
    GenRecord,
    build_snapshot_idf,
    context_affinity,
    has_token_cycle,
    is_repetition_flagged,
    longest_common_run,
)
from .prompts import PromptSet  # noqa: E402
from .temporal_fixture import fresh_share  # noqa: E402

PROTOCOL_SEEDS = (42, 1337, 2026)  # doc 05 §1.3
DEFAULT_GENERATIONS = 500  # 125 x 4 categories
SMOKE_GENERATIONS = 40  # doc 05 §7


@dataclass(slots=True)
class ConfigRun:
    config_id: str
    records: list[GenRecord]
    shared_with: str | None = None  # set when records are aliased, not re-run
    # Per-seed generator telemetry snapshots (cache hit-rate, shadow order-4);
    # deliberately kept out of metrics_summary — telemetry describes the
    # machinery, not the content, and must not break cross-revision
    # content-identity comparisons.
    telemetry: list[dict[str, float | int | None]] = field(default_factory=list)


def _content_cf(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in content_tokens(tokenize(text)))


def _select_prompts(prompts: list[str], count: int, rng: random.Random) -> list[str]:
    """Seeded prompt choice (§1.3): a seeded shuffle defines the order, then
    prompts are cycled until ``count`` generations are covered."""
    order = list(range(len(prompts)))
    rng.shuffle(order)
    return [prompts[order[i % len(order)]] for i in range(count)]


async def run_config_seed(
    *,
    db_source: Path,
    chat_id: int | None,
    overrides: dict[str, Any],
    prompt_set: PromptSet,
    seed: int,
    generations: int,
    fresh_tokens: frozenset[str] | None = None,
    evaluation_moment: int | None = None,
) -> tuple[list[GenRecord], dict[str, float | int | None]]:
    """One (configuration, seed) protocol run — fresh DB copy, cold process state.

    Returns the generation records plus the generator's telemetry snapshot
    (cache hit-rate, shadow order-4 counters) for the report's machinery
    section."""
    log_masking.init_masking("markov2r-eval-protocol")
    per_category = max(1, generations // len(prompt_set.categories))
    db_copy, temp_dir = copy_database(db_source)
    saved_log_selection = gen_trace_log.log_selection
    saved_log_rejected = gen_trace_log.log_attempt_rejected
    records: list[GenRecord] = []
    try:
        resolved_chat = pick_chat_id(db_copy, chat_id)
        messages = load_messages(db_copy, resolved_chat)
        if len(messages) < 10:
            raise ValueError(f"chat {resolved_chat} has too few messages: {len(messages)}")
        verbatim_texts = {normalize_for_verbatim(message) for message in messages}
        idf, default_idf = build_snapshot_idf([list(_content_cf(m)) for m in messages])

        db = Database(str(db_copy))
        await db.init()
        try:
            # Phase 4 arms need the collocation registry a prod process would
            # have: its daily pass rides maintenance, which never runs inside
            # an eval. Populate it the same way, at the same fixed moment that
            # keeps the run reproducible; skipped entirely for arms that leave
            # every Phase 4 knob neutral.
            if (
                overrides.get("markov_collocation_bonus", 0.0) > 0.0
                or overrides.get("markov_collocation_break_penalty", 0.0) > 0.0
                or overrides.get("markov_hot_ngram_meme_ordering", False)
            ):
                meme_settings = MemeSettings()
                await analyze_chat_memes(
                    db.collocations,
                    resolved_chat,
                    now=(
                        evaluation_moment
                        if evaluation_moment is not None
                        else int(time.time())
                    ),
                    min_joint_count=meme_settings.min_joint_count,
                    min_support=meme_settings.min_support,
                    recency_days=meme_settings.recency_days,
                    max_entries=meme_settings.max_entries,
                )
            alltime_ngrams = frozenset(
                tuple(row) for row in await db.get_verbatim_ngrams(resolved_chat)
            )
            generator = _TraceCapturingGenerator(db.markov)
            pool_sizes: list[int] = [0]

            def _on_selection(_chat_id: int, candidates: Any, **_kw: Any) -> None:
                pool_sizes[0] = len(candidates or ())

            rg_rejected = [0]

            def _on_rejected(*_args: Any, **_kw: Any) -> None:
                rg_rejected[0] += 1

            gen_trace_log.log_selection = _on_selection  # type: ignore[assignment]
            gen_trace_log.log_attempt_rejected = _on_rejected  # type: ignore[assignment]

            runtime_state = SimpleNamespace(
                **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS}
            )
            runtime_state.recent_short_replies = {}
            runtime_state.recent_replies = {}
            for key, value in overrides.items():
                if not hasattr(runtime_state, key):
                    raise KeyError(f"matrix override references unknown knob {key!r}")
                setattr(runtime_state, key, value)
            response_generator = ResponseGenerator(
                generator=generator,
                learning_service=_ProdVerbatimChecker(
                    messages, ngram_index=alltime_ngrams, db=db
                ),
                runtime_state=runtime_state,
            )

            index = 0
            for category, prompts in prompt_set.categories.items():
                selector = random.Random(seed)
                for prompt in _select_prompts(prompts, per_category, selector):
                    rng = random.Random(seed * 100_000 + index)
                    index += 1
                    generator.reset_generation()
                    pool_sizes[0] = 0
                    rejected_before = sum(generator.rejections.values()) + rg_rejected[0]
                    prompt_tokens = tokenize(prompt)
                    started = time.perf_counter()
                    result = await response_generator.generate_with_result(
                        GenerationRequest(
                            chat_id=resolved_chat,
                            # Always a list: the pipeline's own convention is an
                            # empty list when there is no context (never None).
                            context_tokens=prompt_tokens,
                            seed=None,
                            current_message_normalized=sanitize_text(prompt).lower(),
                            # M2R-210: a fixed moment, so a time-dependent
                            # weight stays reproducible. On a reconstructed
                            # snapshot this is the corpus's own last moment —
                            # wall-clock "now" would be months later and the
                            # short layer would read as decayed to nothing.
                            now=evaluation_moment,
                        ),
                        rng=rng,
                        candidate_target=CANDIDATE_TARGET,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000
                    rejected = (
                        sum(generator.rejections.values()) + rg_rejected[0] - rejected_before
                    )
                    prompt_content = _content_cf(prompt)
                    if not result.text:
                        records.append(
                            GenRecord(
                                category=category,
                                prompt_content=prompt_content,
                                reply_text="",
                                reply_content=(),
                                success=False,
                                latency_ms=latency_ms,
                                pool_size=pool_sizes[0],
                                rejected_count=rejected,
                            )
                        )
                        continue
                    reply_content = _content_cf(result.text)
                    is_copy = (
                        normalize_for_verbatim(result.text) in verbatim_texts
                        or longest_common_run(reply_content, prompt_content)
                        >= COPY_RUN_TOKENS
                    )
                    meme_hits = frozenset(
                        i
                        for i, meme in enumerate(prompt_set.memes)
                        if longest_common_run(reply_content, tuple(meme)) >= len(meme)
                    )
                    records.append(
                        GenRecord(
                            category=category,
                            prompt_content=prompt_content,
                            reply_text=result.text,
                            reply_content=reply_content,
                            success=True,
                            latency_ms=latency_ms,
                            pool_size=pool_sizes[0],
                            rejected_count=rejected,
                            is_copy=is_copy,
                            has_repetition=is_repetition_flagged(reply_content),
                            has_cycle=has_token_cycle(reply_content),
                            affinity=context_affinity(
                                reply_content, prompt_content, idf, default_idf
                            ),
                            meme_hits=meme_hits,
                            fresh_share=(
                                None
                                if fresh_tokens is None
                                else fresh_share(reply_content, fresh_tokens)
                            ),
                        )
                    )
        finally:
            await db.close()
    finally:
        temp_dir.cleanup()
        gen_trace_log.log_selection = saved_log_selection  # type: ignore[assignment]
        gen_trace_log.log_attempt_rejected = saved_log_rejected  # type: ignore[assignment]
    return records, generator.telemetry.snapshot()


async def run_matrix(
    *,
    db_source: Path,
    chat_id: int | None,
    configs: dict[str, MatrixConfig],
    prompt_set: PromptSet,
    seeds: tuple[int, ...],
    generations: int,
    fresh_tokens: frozenset[str] | None = None,
    evaluation_moment: int | None = None,
) -> tuple[dict[str, ConfigRun], list[str]]:
    """Run every available configuration; alias ones resolving to identical
    overrides (spec: degenerate matrix before any V2 feature exists)."""
    runs: dict[str, ConfigRun] = {}
    skipped: list[str] = []
    resolved_cache: dict[str, dict[str, Any]] = {}
    for config_id, config in configs.items():
        if not config.available:
            skipped.append(config_id)
            continue
        resolved = resolve_overrides(config, configs)
        alias = next(
            (
                other_id
                for other_id, other in resolved_cache.items()
                if other == resolved
            ),
            None,
        )
        if alias is not None:
            runs[config_id] = ConfigRun(
                config_id=config_id,
                records=runs[alias].records,
                shared_with=alias,
            )
            continue
        records: list[GenRecord] = []
        snapshots: list[dict[str, float | int | None]] = []
        for seed in seeds:
            seed_records, snapshot = await run_config_seed(
                db_source=db_source,
                chat_id=chat_id,
                overrides=resolved,
                prompt_set=prompt_set,
                seed=seed,
                generations=generations,
                fresh_tokens=fresh_tokens,
                evaluation_moment=evaluation_moment,
            )
            records.extend(seed_records)
            snapshots.append(snapshot)
        runs[config_id] = ConfigRun(
            config_id=config_id, records=records, telemetry=snapshots
        )
        resolved_cache[config_id] = resolved
    return runs, skipped
