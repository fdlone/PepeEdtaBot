"""Offline generation evaluation against a real (prod-copy) SQLite database.

Unlike ``eval_generation`` (synthetic corpus, temp DB), this harness runs the
full production pipeline (``ResponseGenerator`` best-of-N) against an existing
database file so model changes can be compared before/after on real data.

The database file is copied to a temp location first (together with its WAL/SHM
sidecars) so the source copy stays pristine and is never mutated by migrations.

Headline metric is ``verbatim_run`` — the longest contiguous run of content
tokens in a reply that also appears verbatim in some training message,
normalized by reply length. High values mean the model is replaying memorized
sentences rather than generating.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import log_masking  # noqa: E402
from app.config.registry import RUNTIME_FIELDS  # noqa: E402
from app.core import candidate_scorer as _cs  # noqa: E402
from app.core import context_state_matcher, gen_trace_log  # noqa: E402
from app.core import response_generator as _rg  # noqa: E402
from app.core.candidate_scorer import build_token_idf  # noqa: E402
from app.core.markov import (  # noqa: E402
    MarkovGenerator,
    content_tokens,
    tokenize,
)
from app.core.response_generator import (  # noqa: E402
    CANDIDATE_TARGET,
    GenerationRequest,
    ResponseGenerator,
)
from app.core.text import sanitize_text  # noqa: E402
from app.infrastructure.database import Database  # noqa: E402
from app.services.learning_service import (  # noqa: E402
    content_ngram_windows,
    normalize_for_verbatim,
)
from tools.eval_generation import (  # noqa: E402
    context_token_overlap,
    distinct_ratio,
    repeated_ngram_ratio,
)

DEFAULT_SEED = 20260622
DEFAULT_GENERATIONS = 300
VERBATIM_MIN_N = 4
VERBATIM_MAX_N = 12
# Trace ``start_source`` values meaning the walk was anchored on the reply
# context (visible emission or hidden). Only the eval needs to know these.
_CONTEXT_START_SOURCES = frozenset({"context", "hidden_context"})


class _ProdVerbatimChecker:
    """Mirrors LearningService: verbatim gate + corpus 4-gram index."""

    def __init__(self, messages: list[str]) -> None:
        self._texts = {normalize_for_verbatim(m) for m in messages}
        self._ngram_index = frozenset(
            window
            for message in messages
            for window in content_ngram_windows(message)
        )
        self._token_idf = build_token_idf(tokenize(m) for m in messages)

    async def is_verbatim_copy(self, chat_id: int, text: str) -> bool:
        normalized = normalize_for_verbatim(text)
        return bool(normalized) and normalized in self._texts

    async def get_emoji_stats(self, chat_id: int) -> dict[str, int]:
        # The emoji channel is disabled in evals (emoji_append_chance=0.0);
        # present only to satisfy the VerbatimCopyChecker protocol.
        return {}

    async def get_verbatim_ngram_index(
        self, chat_id: int
    ) -> frozenset[tuple[str, ...]]:
        return self._ngram_index

    async def get_context_idf(self, chat_id: int) -> dict[str, float]:
        return self._token_idf


class _TraceCapturingGenerator(MarkovGenerator):
    """Accumulates per-attempt trace stats across the whole eval run.

    ``attempt_sources``/``attempt_orders`` map each attempt's candidate text to
    the walk's ``start_source`` and ``markov_order_used``. ``ResponseGenerator``
    does not report which candidate won the softmax, so the eval recovers it by
    looking the returned reply up here. Both maps are cleared per generation to
    keep texts from colliding across them.
    """

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.order_used: Counter[int] = Counter()
        self.rejections: Counter[str] = Counter()
        self.context_exact = 0
        self.context_casefold = 0
        self.context_prefix = 0
        self.hidden_context_fallbacks = 0
        self.attempt_sources: dict[str, str] = {}
        self.attempt_orders: dict[str, int] = {}
        self.attempt_jumps: dict[str, int] = {}

    def reset_generation(self) -> None:
        self.attempt_sources.clear()
        self.attempt_orders.clear()
        self.attempt_jumps.clear()

    async def generate_text_with_trace(
        self, *args: Any, **kwargs: Any
    ) -> tuple[str, Any]:
        text, trace = await super().generate_text_with_trace(*args, **kwargs)
        if text:
            self.order_used[trace.markov_order_used] += 1
            self.attempt_sources[text] = trace.start_source
            self.attempt_orders[text] = trace.markov_order_used
            self.attempt_jumps[text] = trace.jump_count
        elif trace.rejection_reason is not None:
            self.rejections[trace.rejection_reason] += 1
        self.context_exact += trace.context_exact_matches
        self.context_casefold += trace.context_casefold_matches
        self.context_prefix += trace.context_prefix_matches
        self.hidden_context_fallbacks += trace.hidden_context_fallbacks
        return text, trace


class _PoolCollector:
    """Captures each generation's scored candidate pool and the softmax pick.

    ``ResponseGenerator`` only returns the winning text, so the conditional
    metric below needs the pool. ``gen_trace_log.log_selection`` is the one hook
    that already receives both; the eval swaps it out (restoring afterwards).
    Candidate start_source comes from the generator's per-attempt text map.
    """

    def __init__(self, sources: dict[str, str]) -> None:
        self._sources = sources
        # (context candidate was top-scoring, the selected one was context)
        self.context_top = 0
        self.context_top_and_won = 0
        self.pools = 0

    def _is_context(self, text: str) -> bool:
        return self._sources.get(text) in _CONTEXT_START_SOURCES

    def on_selection(self, candidates: Sequence[Any], selected: Any) -> None:
        if not candidates or selected is None:
            return
        self.pools += 1
        best = max(candidate.score.total for candidate in candidates)
        top = [c for c in candidates if c.score.total >= best - 1e-9]
        if not any(self._is_context(c.text) for c in top):
            return
        self.context_top += 1
        if self._is_context(selected.text):
            self.context_top_and_won += 1


def copy_database(source: Path) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    temp_dir = tempfile.TemporaryDirectory(prefix="pepe_eval_prod_")
    target = Path(temp_dir.name) / "markov.db"
    shutil.copyfile(source, target)
    for suffix in ("-wal", "-shm"):
        sidecar = source.with_name(source.name + suffix)
        if sidecar.exists():
            shutil.copyfile(sidecar, target.with_name(target.name + suffix))
    return target, temp_dir


def load_messages(db_path: Path, chat_id: int) -> list[str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT normalized_text FROM messages "
            "WHERE chat_id = ? AND normalized_text IS NOT NULL AND normalized_text != ''",
            (chat_id,),
        ).fetchall()
    finally:
        con.close()
    return [row[0] for row in rows]


def pick_chat_id(db_path: Path, requested: int | None) -> int:
    if requested is not None:
        return requested
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT chat_id, COUNT(*) AS c FROM messages "
            "GROUP BY chat_id ORDER BY c DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    if rows is None:
        raise ValueError("no chats found in messages table")
    return int(rows[0])


def build_verbatim_index(
    messages: list[str],
    min_n: int = VERBATIM_MIN_N,
    max_n: int = VERBATIM_MAX_N,
) -> dict[int, set[tuple[str, ...]]]:
    index: dict[int, set[tuple[str, ...]]] = {n: set() for n in range(min_n, max_n + 1)}
    for message in messages:
        tokens = [token.casefold() for token in content_tokens(tokenize(message))]
        for size in index:
            for start in range(len(tokens) - size + 1):
                index[size].add(tuple(tokens[start : start + size]))
    return index


def longest_verbatim_run(
    content_tokens_cf: list[str],
    index: dict[int, set[tuple[str, ...]]],
    min_n: int = VERBATIM_MIN_N,
    max_n: int = VERBATIM_MAX_N,
) -> int:
    """Longest content-token run (>= min_n, capped at max_n) found verbatim."""
    length = len(content_tokens_cf)
    for size in range(min(max_n, length), min_n - 1, -1):
        grams = index[size]
        for start in range(length - size + 1):
            if tuple(content_tokens_cf[start : start + size]) in grams:
                return size
    return 0


async def evaluate(
    *,
    db_source: Path,
    chat_id: int | None,
    seed: int,
    generations: int,
    overrides: dict[str, Any] | None = None,
    selection_margin: float | None = None,
    prefix_confidence: float | None = None,
    context_weight: float | None = None,
    context_cap: float | None = None,
) -> dict[str, Any]:
    """Run the prod pipeline ``generations`` times and report content metrics.

    ``overrides`` sets ``runtime_state`` knobs. The remaining arguments patch
    module constants that are not runtime-configurable
    (``SELECTION_SCORE_MARGIN``, ``_MIN_PREFIX_CONFIDENCE``,
    ``CONTEXT_RELEVANCE_WEIGHT``, ``CONTEXT_RELEVANCE_CAP``); all are restored
    before returning so a sweep can call this repeatedly in one process.
    """
    log_masking.init_masking("prod-generation-evaluation")
    saved_margin = _rg.SELECTION_SCORE_MARGIN
    saved_confidence = context_state_matcher._MIN_PREFIX_CONFIDENCE
    saved_weight = _cs.CONTEXT_RELEVANCE_WEIGHT
    saved_cap = _cs.CONTEXT_RELEVANCE_CAP
    saved_log_selection = gen_trace_log.log_selection
    saved_log_rejected = gen_trace_log.log_attempt_rejected
    saved_log_extended = gen_trace_log.log_attempt_extended
    if selection_margin is not None:
        _rg.SELECTION_SCORE_MARGIN = float(selection_margin)
    if prefix_confidence is not None:
        context_state_matcher._MIN_PREFIX_CONFIDENCE = float(prefix_confidence)
    if context_weight is not None:
        _cs.CONTEXT_RELEVANCE_WEIGHT = float(context_weight)
    if context_cap is not None:
        _cs.CONTEXT_RELEVANCE_CAP = float(context_cap)
    db_copy, temp_dir = copy_database(db_source)
    try:
        resolved_chat = pick_chat_id(db_copy, chat_id)
        messages = load_messages(db_copy, resolved_chat)
        if len(messages) < 10:
            raise ValueError(f"chat {resolved_chat} has too few messages: {len(messages)}")
        verbatim_index = build_verbatim_index(messages)

        sampler = random.Random(seed)
        context_pool = [m for m in messages if len(content_tokens(tokenize(m))) >= 3]

        db = Database(str(db_copy))
        await db.init()
        generator = _TraceCapturingGenerator(db)
        pool = _PoolCollector(generator.attempt_sources)
        gen_trace_log.log_selection = (  # type: ignore[assignment]
            lambda _chat_id, candidates, *, selected=None, **_kw: pool.on_selection(
                candidates, selected
            )
        )
        # Response-generator-level events: gate rejections (the markov-level
        # ones live in generator.rejections) and verbatim-copy extensions.
        rg_rejections: Counter[str] = Counter()
        extended_texts: set[str] = set()
        extension_count = 0

        def _on_rejected(_chat_id, attempt, *, context_used, reason, text):
            rg_rejections[reason] += 1

        def _on_extended(_chat_id, attempt, *, original, extended):
            nonlocal extension_count
            extension_count += 1
            extended_texts.add(extended)

        gen_trace_log.log_attempt_rejected = _on_rejected  # type: ignore[assignment]
        gen_trace_log.log_attempt_extended = _on_extended  # type: ignore[assignment]
        # Built from the registry defaults (app/config/registry.py) so the eval
        # always measures the pipeline the bot actually runs -- a hand-copied
        # namespace silently drifts when a default is retuned. Deviations are
        # deliberate: reply_flavor_strength=0 and emoji_append_chance=0 keep the
        # surface layers out of content metrics.
        runtime_state = SimpleNamespace(
            **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS}
        )
        runtime_state.reply_flavor_strength = 0.0
        runtime_state.emoji_append_chance = 0.0
        runtime_state.recent_short_replies = {}
        runtime_state.recent_replies = {}
        for key, value in (overrides or {}).items():
            setattr(runtime_state, key, value)
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=_ProdVerbatimChecker(messages),
            runtime_state=runtime_state,
        )

        rng = random.Random(seed)
        outputs: list[list[str]] = []
        verbatim_ratios: list[float] = []
        verbatim_runs: list[int] = []
        context_overlaps: list[float] = []
        latencies_ms: list[float] = []
        empty = 0
        # start_source / markov order of the candidate that won selection.
        winner_sources: Counter[str] = Counter()
        winner_orders: Counter[int] = Counter()
        # Joint (start_source, markov order) of winners: the marginals alone
        # cannot say whether relief buys on-topic replies or on-topic salad.
        winner_source_order: Counter[tuple[str, int]] = Counter()
        # M4 jumps of the winning walk (-1 = winner not found in the attempt
        # map). Multi-jump winners are the "long salad" replies; per-bucket
        # length/verbatim slices show what each extra jump costs.
        winner_jumps: Counter[int] = Counter()
        extension_wins = 0
        length_by_jumps: dict[int, list[int]] = {}
        verbatim_by_jumps: dict[int, list[float]] = {}
        try:
            for _ in range(generations):
                generator.reset_generation()
                context_message = sampler.choice(context_pool)
                context_tokens = tokenize(context_message)
                started = time.perf_counter()
                result = await response_generator.generate_with_result(
                    GenerationRequest(
                        chat_id=resolved_chat,
                        context_tokens=context_tokens,
                        seed=None,
                        current_message_normalized=sanitize_text(context_message).lower(),
                    ),
                    rng=rng,
                    candidate_target=CANDIDATE_TARGET,
                )
                latencies_ms.append((time.perf_counter() - started) * 1000)
                if not result.text:
                    empty += 1
                    continue
                # Surface layers are disabled in evals, so the reply is the
                # winning candidate verbatim. "unknown" means that stopped
                # holding and the win-rate below is no longer trustworthy.
                winner_sources[
                    generator.attempt_sources.get(result.text, "unknown")
                ] += 1
                winner_orders[generator.attempt_orders.get(result.text, 0)] += 1
                winner_source_order[
                    (
                        generator.attempt_sources.get(result.text, "unknown"),
                        generator.attempt_orders.get(result.text, 0),
                    )
                ] += 1
                reply_content = content_tokens(tokenize(result.text))
                outputs.append(reply_content)
                content_cf = [token.casefold() for token in reply_content]
                run = longest_verbatim_run(content_cf, verbatim_index)
                verbatim_runs.append(run)
                verbatim_ratios.append(run / len(content_cf) if content_cf else 0.0)
                jumps = generator.attempt_jumps.get(result.text, -1)
                if result.text in extended_texts:
                    extension_wins += 1
                winner_jumps[jumps] += 1
                length_by_jumps.setdefault(jumps, []).append(len(reply_content))
                verbatim_by_jumps.setdefault(jumps, []).append(verbatim_ratios[-1])
                context_overlaps.append(
                    context_token_overlap(reply_content, content_tokens(context_tokens))
                )
        finally:
            await db.close()
    finally:
        temp_dir.cleanup()
        _rg.SELECTION_SCORE_MARGIN = saved_margin
        context_state_matcher._MIN_PREFIX_CONFIDENCE = saved_confidence
        _cs.CONTEXT_RELEVANCE_WEIGHT = saved_weight
        _cs.CONTEXT_RELEVANCE_CAP = saved_cap
        gen_trace_log.log_selection = saved_log_selection  # type: ignore[assignment]
        gen_trace_log.log_attempt_rejected = saved_log_rejected  # type: ignore[assignment]
        gen_trace_log.log_attempt_extended = saved_log_extended  # type: ignore[assignment]

    produced = len(outputs)
    lengths = [len(output) for output in outputs]
    total_context = (
        generator.context_exact
        + generator.context_casefold
        + generator.context_prefix
        + generator.hidden_context_fallbacks
    )
    return {
        "db_source": str(db_source),
        "chat_id": resolved_chat,
        "seed": seed,
        "generations": generations,
        "overrides": overrides or {},
        "selection_margin": saved_margin if selection_margin is None else selection_margin,
        "prefix_confidence": (
            saved_confidence if prefix_confidence is None else prefix_confidence
        ),
        "context_relevance_weight": saved_weight if context_weight is None else context_weight,
        "context_relevance_cap": saved_cap if context_cap is None else context_cap,
        # How often a context-anchored candidate is the best-scoring one, and --
        # the metric that actually governs "does the bot answer on topic" --
        # how often it then survives the softmax roll.
        "context_top_rate": round(pool.context_top / pool.pools, 4) if pool.pools else 0.0,
        "context_win_given_top": round(
            pool.context_top_and_won / pool.context_top, 4
        ) if pool.context_top else 0.0,
        "context_top_n": pool.context_top,
        "empty_result_rate": round(empty / generations, 4),
        # Headline metric for context anchoring: share of replies whose winning
        # candidate started from the reply context rather than a global start.
        "context_anchored_win_rate": round(
            sum(winner_sources[source] for source in _CONTEXT_START_SOURCES) / produced,
            4,
        ) if produced else 0.0,
        "winner_start_sources": dict(winner_sources.most_common()),
        "winner_orders": dict(sorted(winner_orders.items())),
        "winner_source_order": {
            f"{source}/order{order}": count
            for (source, order), count in sorted(winner_source_order.items())
        },
        # M4 jump load of the winners. The multijump rate is the headline: those
        # replies are the incoherent "three topics per message" tail, and the
        # per-bucket slice shows length/verbatim per extra jump.
        "winner_jump_histogram": {
            str(jumps): count for jumps, count in sorted(winner_jumps.items())
        },
        "winner_multijump_rate": round(
            sum(count for jumps, count in winner_jumps.items() if jumps >= 2)
            / produced,
            4,
        ) if produced else 0.0,
        "jump_bucket_stats": {
            f"jumps{jumps}": {
                "n": len(length_by_jumps[jumps]),
                "avg_len": round(mean(length_by_jumps[jumps]), 2),
                "verbatim_mean": round(mean(verbatim_by_jumps[jumps]), 4),
            }
            for jumps in sorted(length_by_jumps)
        },
        # Verbatim-copy extension channel: how often a 1:1 training-sample walk
        # was extended with novel continuation, and how often that won.
        "verbatim_extension_count": extension_count,
        "verbatim_extension_win_rate": round(extension_wins / produced, 4) if produced else 0.0,
        "rg_rejections": dict(rg_rejections.most_common()),
        "verbatim_run_ratio_mean": round(mean(verbatim_ratios), 4) if verbatim_ratios else 0.0,
        "verbatim_run_ratio_median": round(median(verbatim_ratios), 4) if verbatim_ratios else 0.0,
        "verbatim_run_len_median": median(verbatim_runs) if verbatim_runs else 0,
        "reply_ge90pct_verbatim_rate": round(
            sum(1 for r in verbatim_ratios if r >= 0.9) / produced, 4
        ) if produced else 0.0,
        "distinct_1": round(distinct_ratio(outputs, 1), 4),
        "distinct_2": round(distinct_ratio(outputs, 2), 4),
        "repeated_bigram_ratio": round(repeated_ngram_ratio(outputs, 2), 4),
        "repeated_trigram_ratio": round(repeated_ngram_ratio(outputs, 3), 4),
        "avg_length_tokens": round(mean(lengths), 2) if lengths else 0.0,
        "median_length_tokens": median(lengths) if lengths else 0,
        "context_token_overlap_mean": round(mean(context_overlaps), 4) if context_overlaps else 0.0,
        "order_used": dict(sorted(generator.order_used.items())),
        "rejections": dict(generator.rejections.most_common()),
        "context_exact": generator.context_exact,
        "context_casefold": generator.context_casefold,
        "context_prefix": generator.context_prefix,
        "hidden_context_fallbacks": generator.hidden_context_fallbacks,
        "hidden_context_fallback_rate": round(
            generator.hidden_context_fallbacks / total_context, 4
        ) if total_context else 0.0,
        "avg_generation_latency_ms": round(mean(latencies_ms), 2) if latencies_ms else 0.0,
        "median_generation_latency_ms": round(median(latencies_ms), 2) if latencies_ms else 0.0,
    }


def parse_override(item: str) -> tuple[str, Any]:
    """Parse a ``key=value`` runtime_state override, coercing bool/int/float."""
    key, separator, raw = item.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"expected key=value, got {item!r}")
    value: Any
    if raw.lower() in ("true", "false"):
        value = raw.lower() == "true"
    else:
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                value = raw
    return key, value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="path to the SQLite database file")
    parser.add_argument("--chat-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--out", default=None, help="optional path to write JSON report")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        type=parse_override,
        default=[],
        metavar="KEY=VALUE",
        help="override a runtime_state knob, repeatable "
             "(e.g. --set candidate_selection_temperature=0.7)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="override SELECTION_SCORE_MARGIN (module constant)",
    )
    parser.add_argument(
        "--prefix-confidence",
        type=float,
        default=None,
        help="override _MIN_PREFIX_CONFIDENCE (module constant)",
    )
    parser.add_argument(
        "--context-weight",
        type=float,
        default=None,
        help="override CONTEXT_RELEVANCE_WEIGHT (module constant)",
    )
    parser.add_argument(
        "--context-cap",
        type=float,
        default=None,
        help="override CONTEXT_RELEVANCE_CAP (module constant)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(
        evaluate(
            db_source=Path(args.db),
            chat_id=args.chat_id,
            seed=args.seed,
            generations=args.generations,
            overrides=dict(args.overrides),
            selection_margin=args.margin,
            prefix_confidence=args.prefix_confidence,
            context_weight=args.context_weight,
            context_cap=args.context_cap,
        )
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
