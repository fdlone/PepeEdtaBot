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
import sqlite3
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
from app.core.failure_taxonomy import UNMAPPED, classify_reason  # noqa: E402
from app.core.generation_telemetry import GenerationTelemetry  # noqa: E402
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
    DEFAULT_EDGE_OVERLAP_SIMILAR,
    GenRecord,
    build_snapshot_idf,
    context_affinity,
    distinct_trajectories,
    has_token_cycle,
    is_repetition_flagged,
    longest_common_run,
)
from .prompts import PromptSet  # noqa: E402
from .temporal_fixture import fresh_share  # noqa: E402

PROTOCOL_SEEDS = (42, 1337, 2026)  # doc 05 §1.3
DEFAULT_GENERATIONS = 500  # 125 x 4 categories
SMOKE_GENERATIONS = 40  # doc 05 §7

# M3R-140: the two modes every PRE gate is measured in. ``ctx`` is what every
# run before 2026-08-14 did — the prompt goes in as context. ``noctx`` sends the
# same prompt through prompt selection and the RNG but hands the generator no
# context tokens, which is how a self-initiated reply reaches it in production
# (`ReplyPipeline._context_tokens` returns an empty list there).
#
# ``current_message_normalized`` stays populated in BOTH modes on purpose: it
# feeds the echo gate and the seeded channel's anchors, neither of which the
# mode is about. Blanking it too would silently turn off seeded generation and
# make the noctx arm measure a different machine, not a different input.
CONTEXT_MODES = ("ctx", "noctx")

# M3R-145: the L1 seed draw of ``ReplyPipeline._hot_ngram_seed``, reproduced
# for the noctx mode. Its RNG is seeded per generation at this offset from the
# generation RNG's seed (``seed * 100_000 + index``) so the two streams never
# share a seed: a configuration with an empty hot selection stays byte-identical
# to a run without the draw, and the draw never shifts the walk's RNG.
HOT_SEED_RNG_OFFSET = 50_000_000


def draw_hot_seed(
    pool: list[tuple[str, ...]], chance: float, rng: random.Random
) -> tuple[bool, list[str] | None]:
    """One L1 seed draw the pipeline's way: roll first, then choose.

    Returns ``(rolled, seed)``. ``rolled`` is whether the roll asked for a
    seed — it is taken before looking at the pool, exactly as in production,
    so the draw counter keeps the same denominator there and here ("asked and
    got nothing", M3R-141). ``seed`` is ``None`` when the roll failed or the
    pool is empty.
    """
    if chance <= 0.0 or rng.random() >= chance:
        return False, None
    if not pool:
        return True, None
    return True, list(rng.choice(pool))


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


@dataclass(slots=True, frozen=True)
class DfCorpusFacts:
    """Prod-accumulated df of the SOURCE snapshot (gate-phase5-ndocs-floor).

    Read before the snapshot is copied and before ``_populate_token_df``
    window-populates the working copy (design D1): reading the copy would
    always see the retention window the runner itself wrote and turn the
    corpus precondition into a rubber stamp. ``singleton_share`` is ``None``
    when the chat has no df rows at all — "df is empty", not a zero division.
    ``error`` carries the reason when the source could not be read; the gate
    turns it into ``insufficient data``, never a crashed run.
    """

    n_docs: int | None = None
    singleton_share: float | None = None
    error: str | None = None


def read_df_corpus_facts(db_source: Path, chat_id: int | None) -> DfCorpusFacts:
    """n_docs and the df singleton share of ``pick_chat_id``'s chat (design D2).

    Share is over the distinct tokens of that one chat — other chats have
    ``n_docs = 0`` (learned before migration 020) and would only dilute the
    denominator with zeros.
    """
    try:
        resolved = pick_chat_id(db_source, chat_id)
        con = sqlite3.connect(f"file:{db_source}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT n_docs FROM chat_model_volume WHERE chat_id = ?",
                (resolved,),
            ).fetchone()
            n_docs = int(row[0]) if row is not None else 0
            total, singles = con.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(CASE WHEN messages_seen = 1 THEN 1 ELSE 0 END), 0) "
                "FROM markov_token_df WHERE chat_id = ?",
                (resolved,),
            ).fetchone()
        finally:
            con.close()
    except Exception as error:
        return DfCorpusFacts(error=str(error))
    if int(total) == 0:
        return DfCorpusFacts(n_docs=n_docs, singleton_share=None)
    return DfCorpusFacts(
        n_docs=n_docs, singleton_share=int(singles) / int(total)
    )


async def _populate_token_df(
    db: Database, chat_id: int, messages: list[str]
) -> None:
    """Window-approximated df for a Phase 5 arm (design D4).

    +1 per unique token per retained message, plus the message count as
    ``n_docs`` — the same arithmetic the learn path does, over the snapshot's
    retained window. Not prod-accumulated df; the gate accounts for that.
    """
    conn = await db._get_conn()
    for text in messages:
        tokens = set(tokenize(text))
        if not tokens:
            continue
        await conn.executemany(
            "INSERT INTO markov_token_df (chat_id, token, messages_seen) "
            "VALUES (?, ?, 1) ON CONFLICT(chat_id, token) DO UPDATE SET "
            "messages_seen = messages_seen + 1",
            [(chat_id, token) for token in tokens],
        )
    await conn.execute(
        "UPDATE chat_model_volume SET n_docs = ? WHERE chat_id = ?",
        (len(messages), chat_id),
    )
    await conn.commit()


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
    context_mode: str = "ctx",
    edge_overlap_similar: float = DEFAULT_EDGE_OVERLAP_SIMILAR,
) -> tuple[list[GenRecord], dict[str, float | int | None]]:
    """One (configuration, seed) protocol run — fresh DB copy, cold process state.

    Returns the generation records plus the generator's telemetry snapshot
    (cache hit-rate, shadow order-4 counters) for the report's machinery
    section."""
    log_masking.init_masking("markov2r-eval-protocol")
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"unknown context mode {context_mode!r}")
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
            # Phase 5 arms need a df aggregate the seed score can read. df is
            # accumulated on live prod and cannot be backfilled, but the copy's
            # retained messages give a window-approximation — enough to exercise
            # the seeded branch. The promotion gate stays `insufficient data`
            # over window-approximated df (report.py, design D4).
            if overrides.get("markov_seeded_candidate_ratio", 0.0) > 0.0:
                await _populate_token_df(db, resolved_chat, messages)
            alltime_ngrams = frozenset(
                tuple(row) for row in await db.get_verbatim_ngrams(resolved_chat)
            )
            generator = _TraceCapturingGenerator(db.markov)
            pool_sizes: list[int] = [0]
            # M3R-011: the hook only CAPTURES the pool — it runs inside the
            # section the latency metric times, and counting trajectories there
            # would bill the measurement to the generator (measured: +11 ms on
            # p95, enough to make latency incomparable with earlier reports).
            # Five references cost nothing; the counting happens after the timer
            # stops. The capture has to be here regardless: by the time a record
            # is written, only the winning text survives.
            captured_pool: list[tuple[tuple[str, ...], float]] = []
            captured_margin: list[float] = [0.0]
            # M3R-103 (reporting half): the route of every pool candidate, as
            # the generator attributed it. Same hook, same rule — capture only,
            # nothing is counted inside the timed section.
            captured_routes: list[str] = []

            def _on_selection(_chat_id: int, candidates: Any, **kwargs: Any) -> None:
                pool = list(candidates or ())
                pool_sizes[0] = len(pool)
                captured_pool[:] = [
                    (_content_cf(candidate.text), float(candidate.score.total))
                    for candidate in pool
                ]
                captured_routes[:] = [str(candidate.route) for candidate in pool]
                # The margin comes from the selection itself (design D3):
                # hardcoding SELECTION_SCORE_MARGIN would keep measuring an old
                # window the day that margin becomes a knob.
                captured_margin[0] = float(kwargs.get("margin", 0.0))

            def _structural_counts() -> tuple[int, int]:
                """(pool, window) trajectory counts of the last generation."""
                if not captured_pool:
                    return 0, 0
                texts = [text for text, _ in captured_pool]
                best = max(score for _, score in captured_pool)
                window = [
                    text
                    for text, score in captured_pool
                    if score >= best - captured_margin[0]
                ]
                return (
                    distinct_trajectories(texts, similar_at=edge_overlap_similar),
                    distinct_trajectories(window, similar_at=edge_overlap_similar),
                )

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
            # M3R-145: the hot selection a self-initiated reply would draw
            # from, read once per (arm, seed) like the pipeline's cache. noctx
            # only — the pipeline never seeds addressed replies, and addressed
            # replies are the ones that carry context (design D1).
            hot_seed_pool: list[tuple[str, ...]] = []
            if context_mode == "noctx" and runtime_state.hot_ngram_seed_chance > 0.0:
                hot_seed_pool = await db.chat_hot_ngrams.get_hot(
                    resolved_chat,
                    min_count=runtime_state.hot_ngram_min_count,
                    recency_share=runtime_state.hot_ngram_recency_share,
                    meme_ordering=runtime_state.markov_hot_ngram_meme_ordering,
                )

            index = 0
            for category, prompts in prompt_set.categories.items():
                selector = random.Random(seed)
                for prompt in _select_prompts(prompts, per_category, selector):
                    rng = random.Random(seed * 100_000 + index)
                    seed_tokens: list[str] | None = None
                    if context_mode == "noctx":
                        rolled, seed_tokens = draw_hot_seed(
                            hot_seed_pool,
                            runtime_state.hot_ngram_seed_chance,
                            random.Random(seed * 100_000 + index + HOT_SEED_RNG_OFFSET),
                        )
                        if rolled:
                            generator.telemetry.note_hot_ngram_draw(
                                empty=not hot_seed_pool
                            )
                    index += 1
                    generator.reset_generation()
                    pool_sizes[0] = 0
                    captured_pool.clear()
                    captured_routes.clear()
                    rejected_before = sum(generator.rejections.values()) + rg_rejected[0]
                    prompt_tokens = tokenize(prompt)
                    started = time.perf_counter()
                    result = await response_generator.generate_with_result(
                        GenerationRequest(
                            chat_id=resolved_chat,
                            # Always a list: the pipeline's own convention is an
                            # empty list when there is no context (never None).
                            # M3R-140: that empty list IS the noctx mode — the
                            # prompt still selects the generation and seeds the
                            # RNG, so the two modes stay paired per prompt and
                            # the affinity metrics keep measuring against the
                            # prompt (in noctx that is the question: how topical
                            # is a reply whose topic was never shown to it).
                            context_tokens=(
                                prompt_tokens if context_mode == "ctx" else []
                            ),
                            seed=seed_tokens,
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
                    # After the timer, never inside it (M3R-011).
                    pool_ecb, window_escape = _structural_counts()
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
                                pool_ecb=pool_ecb,
                                window_escape=window_escape,
                                pool_routes=tuple(captured_routes),
                                seed_drawn=seed_tokens is not None,
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
                            pool_ecb=pool_ecb,
                            window_escape=window_escape,
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
                            winner_route=result.winner_route,
                            pool_routes=tuple(captured_routes),
                            seed_drawn=seed_tokens is not None,
                            # The winner's walk as the generator traced it,
                            # looked up by text (the eval_prod convention).
                            # A post-processed reply that no longer matches
                            # its attempt reads as None — coverage is then
                            # undercounted, never inflated.
                            start_source=generator.attempt_sources.get(result.text),
                        )
                    )
        finally:
            await db.close()
    finally:
        temp_dir.cleanup()
        gen_trace_log.log_selection = saved_log_selection  # type: ignore[assignment]
        gen_trace_log.log_attempt_rejected = saved_log_rejected  # type: ignore[assignment]
    snapshot = generator.telemetry.snapshot()
    snapshot.update(route_telemetry(generator.telemetry))
    return records, snapshot


def route_telemetry(
    telemetry: GenerationTelemetry,
) -> dict[str, float | int | None]:
    """Per-route counters flattened for the report (M3R-103, reporting half).

    ``route_breakdown`` is nested and stays out of ``snapshot`` because
    ``/stats`` prints flat numbers; the report wants the same numbers per seed,
    so they travel as ``route_<route>_<key>``. Rejections are summed by failure
    class (M3R-021), not by raw reason: the class is the vocabulary features
    and rating rounds are compared in, and the raw reasons are already in the
    trace. An unmapped reason lands in ``unmapped`` rather than vanishing.
    """
    flat: dict[str, float | int | None] = {}
    for route, numbers in telemetry.route_breakdown().items():
        for key, value in numbers.items():
            flat[f"route_{route}_{key}"] = value
    for route, reasons in telemetry.route_rejection_reasons().items():
        for reason, count in reasons.items():
            failure_class = classify_reason(reason)
            key = (
                f"route_{route}_rejected_"
                f"{UNMAPPED if failure_class is None else failure_class.value}"
            )
            flat[key] = int(flat.get(key) or 0) + count
    return flat


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
    context_mode: str = "ctx",
    edge_overlap_similar: float = DEFAULT_EDGE_OVERLAP_SIMILAR,
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
                context_mode=context_mode,
                edge_overlap_similar=edge_overlap_similar,
            )
            records.extend(seed_records)
            snapshots.append(snapshot)
        runs[config_id] = ConfigRun(
            config_id=config_id, records=records, telemetry=snapshots
        )
        resolved_cache[config_id] = resolved
    return runs, skipped
