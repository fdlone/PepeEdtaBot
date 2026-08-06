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
from app.core import gen_trace_log  # noqa: E402
from app.core import response_generator as _rg  # noqa: E402
from app.core.candidate_scorer import build_token_idf  # noqa: E402
from app.core.intonation import (  # noqa: E402
    IntonationProfile,
    build_intonation_profile,
)
from app.core.markov import (  # noqa: E402
    JUMP_CONNECTIVE_TOKENS,
    PUNCT_SET,
    MarkovGenerator,
    build_windows,
    content_tokens,
    tokenize,
)
from app.core.response_generator import (  # noqa: E402
    CANDIDATE_TARGET,
    GenerationRequest,
    ResponseGenerator,
)
from app.core.slot_mutation import (  # noqa: E402
    MIN_MUTABLE_WORD_LEN,
    frequencies_by_ending,
)
from app.core.text import sanitize_text  # noqa: E402
from app.infrastructure.database import Database  # noqa: E402
from app.services.learning_service import (  # noqa: E402
    normalize_for_verbatim,
)
from tools.eval_generation import (  # noqa: E402
    context_token_overlap,
    distinct_ratio,
    repeated_ngram_ratio,
)


def content_ngram_windows(
    text: str, size: int = 4
) -> list[tuple[str, ...]]:
    """Casefolded content-token ``size``-grams of one message.

    Жила в LearningService, но рантайм строит окна из токенов уже выученного
    сообщения (``verbatim_ngram_windows``), а этот вариант — из готового
    текста, и нужен только здесь: eval читает сообщения из базы, а не через
    путь обучения.
    """
    content = [token.casefold() for token in content_tokens(tokenize(text))]
    return build_windows(content, size)


DEFAULT_SEED = 20260622
# 400 (2026-07-21): the floor for telling a knob's effect from sampling noise.
# At 200 generations the per-seed spread of context_anchored_win_rate is sd
# ~0.040 -- wider than most real knob effects (+-0.02..0.04), which is how the
# 2026-07-20 pass A produced three false "noisy/non-monotonic" verdicts that
# pass B overturned. At 400 the spread drops to sd ~0.015 (SE of a paired
# difference ~0.005), enough to resolve effects down to ~0.02. Comparative
# sweeps should not go below this; single smoke runs may pass --generations.
DEFAULT_GENERATIONS = 400
VERBATIM_MIN_N = 4
VERBATIM_MAX_N = 12
# Trace ``start_source`` values meaning the walk was anchored on the reply
# context (visible emission or hidden). Only the eval needs to know these.
_CONTEXT_START_SOURCES = frozenset(
    {"context", "hidden_context", "context_spliced"}
)


class _ProdVerbatimChecker:
    """Mirrors LearningService: verbatim gate + corpus 4-gram index.

    ``ngram_index`` mirrors the cumulative ``chat_verbatim_ngrams`` table (O4)
    when provided; the message-window fallback exists only for callers without
    a migrated database at hand.
    """

    def __init__(
        self,
        messages: list[str],
        ngram_index: frozenset[tuple[str, ...]] | None = None,
        db: Database | None = None,
    ) -> None:
        self._texts = {normalize_for_verbatim(m) for m in messages}
        self._ngram_index = ngram_index if ngram_index is not None else frozenset(
            window
            for message in messages
            for window in content_ngram_windows(message)
        )
        self._token_idf = build_token_idf(tokenize(m) for m in messages)
        # P4 intonation: built lazily from the same message window the runtime
        # LearningService profiles; only read when the knob is on.
        self._messages = list(messages)
        self._intonation_built = False
        self._intonation: IntonationProfile | None = None
        # Slot mutations read the chat frequency dictionary and hot n-grams;
        # the eval delegates to the migrated DB copy and caches (the runtime
        # LearningService caches too, so per-request cost matches prod).
        self._db = db
        self._word_frequencies: dict[str, int] | None = None
        self._frequencies_by_ending: dict[str, dict[str, int]] | None = None
        self._hot_ngrams: list[tuple[str, ...]] | None = None

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

    async def get_intonation_profile(
        self, chat_id: int
    ) -> IntonationProfile | None:
        if not self._intonation_built:
            self._intonation = build_intonation_profile(self._messages)
            self._intonation_built = True
        return self._intonation
    async def get_word_frequencies(self, chat_id: int) -> dict[str, int]:
        if self._db is None:
            return {}
        if self._word_frequencies is None:
            self._word_frequencies = await self._db.get_word_frequencies(
                chat_id, min_word_len=MIN_MUTABLE_WORD_LEN
            )
        return self._word_frequencies

    async def get_word_frequencies_by_ending(
        self, chat_id: int
    ) -> dict[str, dict[str, int]]:
        if self._frequencies_by_ending is None:
            self._frequencies_by_ending = frequencies_by_ending(
                await self.get_word_frequencies(chat_id)
            )
        return self._frequencies_by_ending

    async def get_hot_ngrams(
        self, chat_id: int, *, min_count: int, recency_share: float
    ) -> list[tuple[str, ...]]:
        if self._db is None:
            return []
        if self._hot_ngrams is None:
            self._hot_ngrams = await self._db.get_hot_chat_ngrams(
                chat_id, min_count=min_count, recency_share=recency_share
            )
        return self._hot_ngrams


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


# Connective phrases that contain at least one word: the silent «.» splice is
# invisible by design and must not count as a connective.
_WORDY_CONNECTIVES: tuple[tuple[str, ...], ...] = tuple(
    phrase
    for phrase in JUMP_CONNECTIVE_TOKENS
    if any(token not in PUNCT_SET for token in phrase)
)


def contains_splice_connective(tokens: list[str]) -> bool:
    """True if the reply contains a splice connective phrase as a contiguous
    token run.

    Slight overcount by design: the connective words also occur naturally in
    chat speech. The bias is identical on both sides of any comparison, and
    the metric exists to compare configurations, not to report an absolute.
    """
    lowered = [token.casefold() for token in tokens]
    for phrase in _WORDY_CONNECTIVES:
        size = len(phrase)
        for start in range(len(lowered) - size + 1):
            if tuple(lowered[start : start + size]) == phrase:
                return True
    return False


def novel_ngram_share(
    content_tokens_cf: list[str],
    index: dict[int, set[tuple[str, ...]]],
    size: int = VERBATIM_MIN_N,
) -> float | None:
    """Share of the reply's content ``size``-grams NOT found in the corpus.

    The direct measure of «отсебятина»: 0.0 means every window of the reply
    exists verbatim in some training message (pure corpus recombination at
    best), 1.0 means no window does. ``verbatim_run_ratio`` cannot see this —
    it measures the single longest run, so a reply spliced from two long
    quotes scores the same as one quote half its length. Replies shorter than
    ``size`` content tokens return None and must be excluded from means (they
    are governed by the short-reply anti-repeat, not by novelty).
    """
    grams = index[size]
    windows = [
        tuple(content_tokens_cf[start : start + size])
        for start in range(len(content_tokens_cf) - size + 1)
    ]
    if not windows:
        return None
    hits = sum(1 for window in windows if window in grams)
    return 1.0 - hits / len(windows)


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


def incoming_length_bucket(incoming_tokens: int) -> str:
    """Bucket the answered message by length, matching the scorer's own bands.

    The cut points are the short/long thresholds the length conditioning uses
    (candidate_scorer.CONTEXT_LENGTH_SHORT/LONG_TOKENS), so the buckets show
    exactly the populations the knob is supposed to pull apart.
    """
    if incoming_tokens <= _cs.CONTEXT_LENGTH_SHORT_TOKENS:
        return "short_in"
    if incoming_tokens >= _cs.CONTEXT_LENGTH_LONG_TOKENS:
        return "long_in"
    return "mid_in"


def _length_shares(lengths: list[int]) -> dict[str, float]:
    """Short/medium/long shares of content-token lengths (scorer's bands)."""
    counted = [length for length in lengths if length > 0]
    if not counted:
        return {"short": 0.0, "medium": 0.0, "long": 0.0}
    short_max = _cs.LENGTH_MODE_BANDS["short"][1]
    long_min = _cs.LENGTH_MODE_BANDS["long"][0]
    total = len(counted)
    short = sum(1 for length in counted if length <= short_max)
    long_ = sum(1 for length in counted if length >= long_min)
    return {
        "short": round(short / total, 4),
        "medium": round((total - short - long_) / total, 4),
        "long": round(long_ / total, 4),
    }


async def evaluate(
    *,
    db_source: Path,
    chat_id: int | None,
    seed: int,
    generations: int,
    overrides: dict[str, Any] | None = None,
    selection_margin: float | None = None,
    context_weight: float | None = None,
    context_cap: float | None = None,
) -> dict[str, Any]:
    """Run the prod pipeline ``generations`` times and report content metrics.

    ``overrides`` sets ``runtime_state`` knobs. The remaining arguments patch
    module constants that are not runtime-configurable
    (``SELECTION_SCORE_MARGIN``, ``CONTEXT_RELEVANCE_WEIGHT``,
    ``CONTEXT_RELEVANCE_CAP``); all are restored
    before returning so a sweep can call this repeatedly in one process.
    """
    log_masking.init_masking("prod-generation-evaluation")
    saved_margin = _rg.SELECTION_SCORE_MARGIN
    saved_weight = _cs.CONTEXT_RELEVANCE_WEIGHT
    saved_cap = _cs.CONTEXT_RELEVANCE_CAP
    saved_log_selection = gen_trace_log.log_selection
    saved_log_rejected = gen_trace_log.log_attempt_rejected
    saved_log_extended = gen_trace_log.log_attempt_extended
    saved_log_mutated = gen_trace_log.log_attempt_mutated
    if selection_margin is not None:
        _rg.SELECTION_SCORE_MARGIN = float(selection_margin)
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
        # db.init() ran migration 016 on the copy, so the cumulative all-time
        # index is available; both the runtime checker and the metric index
        # below use it — the message-window index understated verbatim ~3x.
        alltime_ngrams = frozenset(
            tuple(row) for row in await db.get_verbatim_ngrams(resolved_chat)
        )
        verbatim_index[VERBATIM_MIN_N] |= set(alltime_ngrams)
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
        # An extended reply is a NEW string (base + connective + tail), absent
        # from attempt_sources, so its winner would be attributed "unknown" and
        # silently leave the context_anchored bucket even though its head is the
        # original context-anchored candidate. Map it back to the base so the
        # start-source metric follows the walk that actually opened the reply.
        extended_to_original: dict[str, str] = {}
        extension_count = 0

        def _on_rejected(_chat_id, attempt, *, context_used, reason, text):
            rg_rejections[reason] += 1

        def _on_extended(_chat_id, attempt, *, original, extended):
            nonlocal extension_count
            extension_count += 1
            extended_texts.add(extended)
            extended_to_original[extended] = original

        # Slot mutations (P2): every fielded mutated copy as (original,
        # mutated) — the pairs feed the manual morphology review; the set
        # recovers which winners were mutations.
        mutation_pairs: list[tuple[str, str]] = []
        mutated_to_original: dict[str, str] = {}

        def _on_mutated(_chat_id, attempt, *, original, mutated, **_kw):
            mutation_pairs.append((original, mutated))
            mutated_to_original[mutated] = original

        gen_trace_log.log_attempt_rejected = _on_rejected  # type: ignore[assignment]
        gen_trace_log.log_attempt_extended = _on_extended  # type: ignore[assignment]
        gen_trace_log.log_attempt_mutated = _on_mutated  # type: ignore[assignment]
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
            learning_service=_ProdVerbatimChecker(
                messages, ngram_index=alltime_ngrams, db=db
            ),
            runtime_state=runtime_state,
        )

        rng = random.Random(seed)
        outputs: list[list[str]] = []
        verbatim_ratios: list[float] = []
        verbatim_runs: list[int] = []
        novel_shares: list[float] = []
        connective_replies = 0
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
        mutation_wins = 0
        # (original, mutated) of replies that actually won selection: the
        # perceptual quality bar — fielded-but-losing mutations never surface.
        mutation_winner_pairs: list[tuple[str, str]] = []
        length_by_jumps: dict[int, list[int]] = {}
        verbatim_by_jumps: dict[int, list[float]] = {}
        # Reply length sliced by the length of the message being answered: a
        # bot that mirrors its interlocutor answers a two-word question with a
        # short reply. Without conditioning, every bucket has the same mean.
        length_by_incoming: dict[str, list[int]] = {}
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
                # A mutated winner keeps its original's walk attribution —
                # the mutation changes one word, not the start source. An
                # extended winner (base + connective + tail) likewise keeps its
                # base's start source: the head is the walk that opened it.
                source_text = extended_to_original.get(result.text, result.text)
                source_text = mutated_to_original.get(source_text, source_text)
                winner_sources[
                    generator.attempt_sources.get(source_text, "unknown")
                ] += 1
                winner_orders[generator.attempt_orders.get(source_text, 0)] += 1
                winner_source_order[
                    (
                        generator.attempt_sources.get(source_text, "unknown"),
                        generator.attempt_orders.get(source_text, 0),
                    )
                ] += 1
                reply_content = content_tokens(tokenize(result.text))
                outputs.append(reply_content)
                content_cf = [token.casefold() for token in reply_content]
                run = longest_verbatim_run(content_cf, verbatim_index)
                verbatim_runs.append(run)
                verbatim_ratios.append(run / len(content_cf) if content_cf else 0.0)
                novelty = novel_ngram_share(content_cf, verbatim_index)
                if novelty is not None:
                    novel_shares.append(novelty)
                if contains_splice_connective(tokenize(result.text)):
                    connective_replies += 1
                jumps = generator.attempt_jumps.get(result.text, -1)
                if result.text in extended_texts:
                    extension_wins += 1
                if result.text in mutated_to_original:
                    mutation_wins += 1
                    mutation_winner_pairs.append(
                        (mutated_to_original[result.text], result.text)
                    )
                winner_jumps[jumps] += 1
                length_by_jumps.setdefault(jumps, []).append(len(reply_content))
                verbatim_by_jumps.setdefault(jumps, []).append(verbatim_ratios[-1])
                incoming_content = content_tokens(context_tokens)
                length_by_incoming.setdefault(
                    incoming_length_bucket(len(incoming_content)), []
                ).append(len(reply_content))
                context_overlaps.append(
                    context_token_overlap(reply_content, incoming_content)
                )
        finally:
            await db.close()
    finally:
        temp_dir.cleanup()
        _rg.SELECTION_SCORE_MARGIN = saved_margin
        _cs.CONTEXT_RELEVANCE_WEIGHT = saved_weight
        _cs.CONTEXT_RELEVANCE_CAP = saved_cap
        gen_trace_log.log_selection = saved_log_selection  # type: ignore[assignment]
        gen_trace_log.log_attempt_rejected = saved_log_rejected  # type: ignore[assignment]
        gen_trace_log.log_attempt_extended = saved_log_extended  # type: ignore[assignment]
        gen_trace_log.log_attempt_mutated = saved_log_mutated  # type: ignore[assignment]

    produced = len(outputs)
    lengths = [len(output) for output in outputs]
    total_context = (
        generator.context_exact
        + generator.context_casefold
        + generator.hidden_context_fallbacks
    )
    return {
        "db_source": str(db_source),
        "chat_id": resolved_chat,
        "seed": seed,
        "generations": generations,
        "overrides": overrides or {},
        "selection_margin": saved_margin if selection_margin is None else selection_margin,
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
        # Slot mutations (P2): fielded mutated copies, how often one won, and
        # (original, mutated) pairs for the manual morphology review.
        "slot_mutation_candidates": len(mutation_pairs),
        "slot_mutation_win_rate": round(mutation_wins / produced, 4) if produced else 0.0,
        "slot_mutation_samples": [
            {"original": original, "mutated": mutated}
            for original, mutated in mutation_pairs[:100]
        ],
        "slot_mutation_winner_samples": [
            {"original": original, "mutated": mutated}
            for original, mutated in mutation_winner_pairs[:100]
        ],
        "rg_rejections": dict(rg_rejections.most_common()),
        # Отсебятина: share of the reply's content 4-grams absent from the
        # corpus, averaged over replies long enough to have 4-gram windows.
        # pure_corpus_reply_rate is its tail: replies where EVERY window is a
        # known corpus 4-gram (novelty exactly 0) — recombination indistin-
        # guishable from quoting at this n-gram size.
        # Formularity of splices: share of replies carrying a wordy connective
        # phrase («, кстати» ...). The six-phrase pool at 82% saturation is the
        # documented perceptual risk this metric was added to track.
        "connective_reply_rate": round(connective_replies / produced, 4)
        if produced else 0.0,
        "novel_ngram_share_mean": round(mean(novel_shares), 4) if novel_shares else 0.0,
        "novel_ngram_share_median": round(median(novel_shares), 4) if novel_shares else 0.0,
        "novel_evaluable_n": len(novel_shares),
        "pure_corpus_reply_rate": round(
            sum(1 for share in novel_shares if share == 0.0) / len(novel_shares), 4
        ) if novel_shares else 0.0,
        "verbatim_run_ratio_mean": round(mean(verbatim_ratios), 4) if verbatim_ratios else 0.0,
        "verbatim_run_ratio_median": round(median(verbatim_ratios), 4) if verbatim_ratios else 0.0,
        "verbatim_run_len_median": median(verbatim_runs) if verbatim_runs else 0,
        "reply_ge90pct_verbatim_rate": round(
            sum(1 for r in verbatim_ratios if r >= 0.9) / produced, 4
        ) if produced else 0.0,
        "distinct_1": round(distinct_ratio(outputs, 1), 4),
        "distinct_2": round(distinct_ratio(outputs, 2), 4),
        # Denominator behind distinct_*: that metric is a type/token ratio and
        # sinks as more text is pooled, so it is only comparable between arms
        # measured at the same volume. Recorded so a cross-run comparison can
        # be checked against the data instead of remembered.
        "distinct_basis_tokens": sum(len(output) for output in outputs),
        "repeated_bigram_ratio": round(repeated_ngram_ratio(outputs, 2), 4),
        "repeated_trigram_ratio": round(repeated_ngram_ratio(outputs, 3), 4),
        "avg_length_tokens": round(mean(lengths), 2) if lengths else 0.0,
        "median_length_tokens": median(lengths) if lengths else 0,
        # P4 intonation: reply length-mode shares next to the chat's own
        # (classified by the scorer's bands) — the knob is supposed to pull
        # the former toward the latter.
        "reply_length_shares": _length_shares(lengths),
        "chat_length_shares": _length_shares(
            [len(content_tokens(tokenize(m))) for m in messages]
        ),
        # Length mirroring: reply length per bucket of the answered message.
        # length_mirror_gap (long bucket mean - short bucket mean) is the
        # headline -- it is ~0 when the length mode ignores the interlocutor.
        "length_by_incoming": {
            bucket: {
                "n": len(length_by_incoming[bucket]),
                "avg_len": round(mean(length_by_incoming[bucket]), 2),
            }
            for bucket in sorted(length_by_incoming)
        },
        "length_mirror_gap": round(
            mean(length_by_incoming["long_in"]) - mean(length_by_incoming["short_in"]),
            2,
        ) if length_by_incoming.get("long_in") and length_by_incoming.get("short_in")
        else 0.0,
        "context_token_overlap_mean": round(mean(context_overlaps), 4) if context_overlaps else 0.0,
        "order_used": dict(sorted(generator.order_used.items())),
        "rejections": dict(generator.rejections.most_common()),
        "context_exact": generator.context_exact,
        "context_casefold": generator.context_casefold,
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
