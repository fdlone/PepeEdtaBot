"""Metric implementations for the eval protocol.

Every public function references the section of
``docs/v2/05_MARKOV_2_0R_EVAL_PROTOCOL.md`` it implements (doc 05 §3 is the
normative source; numbers in comments below are its subsections). Metrics that
cannot be computed in the current phase return ``None`` and are reported as
``insufficient data`` — never silently omitted (spec: generation-eval).

Interpretations fixed here (doc 05 leaves them to the scorer configuration):

- §3.2 ``exact_context_copy_rate``: K = 4 content tokens (the scorer's
  ``VERBATIM_NGRAM_SIZE``). An answer is a copy when its verbatim-normalized
  text equals a training message (the runtime ``is_verbatim_copy`` convention)
  OR it shares a contiguous casefolded content-token run of length >= K with
  the prompt.
- §3.2 ``repetition_rate``: an answer repeats when it contains a repeated
  content trigram, or repeated content bigrams above 0.2 of all its bigrams
  (the scorer penalizes gradually; the flag marks the clearly degraded tail).
- §3.2 ``cycle_detection_rate`` (shadow, pre-Phase-6): a period-2 or period-3
  token cycle — some position where the next 2 (or 3) tokens repeat the
  previous 2 (or 3) verbatim, i.e. the walk revisited the same transition.
- §3.2 ``cycle_harm_rate``: automatic component only until a manual round
  (doc 05 §5) exists — a generation counts when a cycle was detected AND the
  answer is also flagged by ``repetition_rate``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

COPY_RUN_TOKENS = 4  # K of §3.2 = scorer's VERBATIM_NGRAM_SIZE


@dataclass(slots=True)
class GenRecord:
    """One generation's outcome; the unit every metric aggregates over."""

    category: str
    prompt_content: tuple[str, ...]  # casefolded content tokens of the prompt
    reply_text: str
    reply_content: tuple[str, ...]  # casefolded content tokens of the reply
    success: bool
    latency_ms: float
    pool_size: int = 0
    rejected_count: int = 0
    is_copy: bool = False
    has_repetition: bool = False
    has_cycle: bool = False
    affinity: float | None = None
    meme_hits: frozenset[int] = field(default_factory=frozenset)
    # §3.5: share of the reply's content tokens that belong to the snapshot's
    # fresh slice. ``None`` on a snapshot without observation times — the
    # metric then reports insufficient data rather than a fabricated zero.
    fresh_share: float | None = None
    # M3R-011: how many substantially different trajectories this generation
    # produced — over the whole pool, and over the selection window (the
    # candidates the final draw could actually have returned). Always a pair:
    # the gap between them is the finding, and either number alone reads as its
    # opposite.
    pool_ecb: int = 0
    window_escape: int = 0
    # M3R-103 (reporting half): the mechanism that built the winner and the
    # route of every candidate in the pool, as the generator attributed them at
    # creation — never inferred from text afterwards. ``None`` / empty on a
    # generation that produced no pool. Inputs of the per-route table only:
    # they stay out of ``metrics_summary`` on purpose, because that object is
    # compared bit-for-bit across runs and revisions and a new key there would
    # break every comparison (design D5).
    winner_route: str | None = None
    pool_routes: tuple[str, ...] = ()
    # M3R-145: whether the harness drew an L1 hot-n-gram seed for this
    # generation (noctx only, the pipeline's way), and the walk's start source
    # of the winner as the generator traced it. The gate's coverage reads the
    # second: a drawn seed anchors a walk only when it opens a stored start.
    seed_drawn: bool = False
    start_source: str | None = None


# M3R-011: the similarity threshold lives in eval_thresholds.yaml
# (`structural_escape.edge_overlap_similar`); this default exists only so the
# helpers are usable from a test or a scratch script without loading the file.
DEFAULT_EDGE_OVERLAP_SIMILAR = 0.5


# M3R-100: the trajectory identity lives in app.core.trajectory so the
# diversity bonus and this gate cannot drift apart on what "the same
# trajectory" means; re-exported here for the existing callers.
from app.core.trajectory import edge_overlap, trajectory_edges  # noqa: E402


def distinct_trajectories(
    candidates: list[tuple[str, ...]],
    *,
    similar_at: float = DEFAULT_EDGE_OVERLAP_SIMILAR,
) -> int:
    """How many substantially different trajectories a candidate set holds.

    Similar candidates are merged transitively: if A is similar to B and B to
    C, all three are one trajectory even when A and C have drifted apart. The
    alternative — the largest pairwise-different subset — is the independent-set
    problem, and its answer depends on enumeration order at ties; this report is
    compared bit-for-bit, so an order-dependent number is not an option.

    The trade is a possible undercount when near-similar candidates chain into
    one group. Undercounting is the safe side for a gate that demands ">= 2
    substantially different": it withholds a pass it should have given rather
    than granting one it should not.
    """
    edges = [trajectory_edges(tokens) for tokens in candidates]
    parent = list(range(len(edges)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if edge_overlap(edges[i], edges[j]) >= similar_at:
                parent[find(j)] = find(i)
    return len({find(i) for i in range(len(edges))})


def longest_common_run(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Longest contiguous token run present in both sequences (§3.2 helper)."""
    if not a or not b:
        return 0
    b_positions: dict[str, list[int]] = {}
    for j, token in enumerate(b):
        b_positions.setdefault(token, []).append(j)
    best = 0
    for i in range(len(a)):
        for j in b_positions.get(a[i], ()):
            length = 0
            while i + length < len(a) and j + length < len(b) and a[i + length] == b[j + length]:
                length += 1
            best = max(best, length)
    return best


def has_token_cycle(tokens: tuple[str, ...]) -> bool:
    """Shadow 2/3-cycle detector (§3.2): the same 2- or 3-token unit repeated
    back-to-back means the walk revisited its own transition."""
    for period in (2, 3):
        for i in range(len(tokens) - 2 * period + 1):
            if tokens[i : i + period] == tokens[i + period : i + 2 * period]:
                return True
    return False


def repeated_ngram_share(tokens: tuple[str, ...], n: int) -> float:
    """Share of n-grams occurring more than once (§3.2 repetition input)."""
    if len(tokens) < n:
        return 0.0
    grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    total = sum(grams.values())
    repeated = sum(count for count in grams.values() if count > 1)
    return repeated / total if total else 0.0


def is_repetition_flagged(tokens: tuple[str, ...]) -> bool:
    """§3.2 repetition_rate flag (interpretation in the module docstring)."""
    return repeated_ngram_share(tokens, 3) > 0.0 or repeated_ngram_share(tokens, 2) > 0.2


def build_snapshot_idf(messages_content: list[tuple[str, ...]]) -> tuple[dict[str, float], float]:
    """IDF over the snapshot's df-aggregate (§3.3; formula from TZ §9.3).

    Document = one normalized message; ``df(t)`` counts messages containing the
    token; ``idf(t) = log(N_docs / (1 + df(t)))``. Returns the idf map and the
    default idf for tokens unseen in the snapshot (df = 0). Honest limitation,
    recorded in every report: the snapshot retains only the message-retention
    window, so df is window-relative (audit §3); deltas between configurations
    remain valid because all configurations share the same idf.
    """
    n_docs = len(messages_content)
    df: Counter[str] = Counter()
    for content in messages_content:
        df.update(set(content))
    if n_docs == 0:
        return {}, 0.0
    idf = {token: math.log(n_docs / (1 + count)) for token, count in df.items()}
    return idf, math.log(n_docs / 1)


def context_affinity(
    reply_content: tuple[str, ...],
    prompt_content: tuple[str, ...],
    idf: dict[str, float],
    default_idf: float,
) -> float | None:
    """§3.3 ``context_affinity``: IDF-weighted token-set intersection of the
    answer with the prompt, normalized by the prompt's IDF mass. ``None`` when
    the prompt carries no IDF mass (nothing to be topical about)."""
    prompt_set = set(prompt_content)
    denom = sum(max(idf.get(token, default_idf), 0.0) for token in prompt_set)
    if denom <= 0:
        return None
    shared = prompt_set & set(reply_content)
    num = sum(max(idf.get(token, default_idf), 0.0) for token in shared)
    return num / denom


def proportion_samples(records: list[GenRecord], flag: str) -> list[float]:
    """Per-generation 0/1 samples for a boolean attribute — bootstrap input."""
    return [1.0 if getattr(record, flag) else 0.0 for record in records]


def _aligned(
    records: list[GenRecord], value_of: Callable[[GenRecord], float | None]
) -> list[float] | None:
    """Значение на каждую запись, с ``nan`` там, где записи в метрике нет.

    Раньше такие метрики просто отфильтровывались (``for r in successful``), и
    список выходил короче исходного на неотвеченные записи. Для одиночного
    интервала это безразлично, а для дельты — нет: армы не отвечают на *разных*
    промптах, поэтому после фильтрации позиция перестаёт быть идентификатором
    наблюдения, и парный ресэмплинг сшивал бы разные промпты.

    ``nan`` вместо пропуска сохраняет выравнивание по исходному списку записей,
    а потребители его отбрасывают: одиночный интервал — из выборки, парная
    дельта — вместе со всей парой. ``None`` был бы честнее по типу, но
    протащил бы `float | None` через все сигнатуры бутстрапа ради того же
    результата.
    """
    values = [
        math.nan if (value := value_of(record)) is None else value
        for record in records
    ]
    return values if any(not math.isnan(value) for value in values) else None


def metric_values(records: list[GenRecord]) -> dict[str, list[float] | None]:
    """Per-generation sample arrays for every §3 metric.

    Values are lists (bootstrap resamples them) or ``None`` = insufficient
    data in the current phase. Config-level ratios (distinct-N) are computed
    by :func:`distinct_n` on resampled reply sets instead.
    """
    return {
        # §3.1
        "generation_success_rate": proportion_samples(records, "success"),
        # Через `_aligned`, как остальные фильтрованные: фильтр здесь
        # арм-зависимый (`pool_size` и `rejected_count` — результат
        # генерации), поэтому без выравнивания парная дельта сшивала бы
        # разные промпты. Единственная метрика `METRIC_ORDER`, которую первая
        # редакция A2-4 пропустила.
        "candidate_accept_rate": _aligned(
            records,
            lambda r: r.pool_size / (r.pool_size + r.rejected_count)
            if r.pool_size + r.rejected_count > 0
            else None,
        ),
        "mean_response_length": _aligned(
            records, lambda r: float(len(r.reply_content)) if r.success else None
        ),
        "unique_token_ratio": _aligned(
            records,
            lambda r: len(set(r.reply_content)) / len(r.reply_content)
            if r.success and r.reply_content
            else None,
        ),
        # §3.2
        "exact_context_copy_rate": _aligned(
            records, lambda r: float(r.is_copy) if r.success else None
        ),
        "repetition_rate": _aligned(
            records, lambda r: float(r.has_repetition) if r.success else None
        ),
        "cycle_detection_rate": _aligned(
            records, lambda r: float(r.has_cycle) if r.success else None
        ),
        "cycle_harm_rate": _aligned(
            records,
            lambda r: float(r.has_cycle and r.has_repetition) if r.success else None,
        ),
        # §3.3
        "context_affinity": _aligned(
            records, lambda r: r.affinity if r.success else None
        ),
        "context_affinity_without_copy": _aligned(
            records, lambda r: r.affinity if r.success and not r.is_copy else None
        ),
        # §3.4 — seeded generation does not exist before Phase 5.
        "seeded_present_rate": None,
        "seeded_win_rate_given_present": None,
        # §3.5 — both require a temporal snapshot; ``None`` on any snapshot
        # without observation times (audit §10.1).
        "freshness_reflection": _aligned(
            records, lambda r: r.fresh_share if r.success else None
        ),
        "historical_meme_rate": _aligned(
            records,
            lambda r: float(bool(r.meme_hits))
            if r.success and r.category == "meme-bait"
            else None,
        ),
        # M3R-011 — the pair, over ALL records rather than the successful ones:
        # a generation that collected nothing produced zero trajectories, and
        # dropping it would measure diversity only where diversity happened.
        "structural_pool_ecb": [float(r.pool_ecb) for r in records] or None,
        "structural_window_escape": [float(r.window_escape) for r in records] or None,
        # Доля, а не счёт. Порог `pool_ecb_min` пре-регистрирован как ЧИСЛО
        # различных траекторий, а знаменатель 5 держит только проза CLAUDE.md
        # («ECB ≥ 4/5 по пулу»). Пул уже не равен пяти: seeded добавляет
        # кандидатов сверх `effective_target`, а каждый маршрут Track B
        # добавит ещё, и на пуле 11 порог 4 берётся тривиально. Порог не
        # двигается — пре-регистрация, — но рядом с ним печатается величина,
        # которую он должен был мерить. Тем же приёмом M3R-011 завёл «долю
        # входов ниже 2» рядом со средним по окну.
        "structural_pool_ecb_share": [
            float(r.pool_ecb) / r.pool_size for r in records if r.pool_size > 0
        ]
        or None,
    }


def distinct_n(replies: list[tuple[str, ...]], n: int) -> tuple[float | None, int]:
    """§3.1 distinct-N over a set of final answers: unique n-grams / total
    n-grams, plus the token basis (type/token ratios are only comparable at
    equal basis — the ``distinct_basis_tokens`` lesson)."""
    grams: list[tuple[str, ...]] = []
    for reply in replies:
        grams.extend(tuple(reply[i : i + n]) for i in range(len(reply) - n + 1))
    if not grams:
        return None, 0
    return len(set(grams)) / len(grams), len(grams)


def latency_percentiles(records: list[GenRecord]) -> dict[str, float | None]:
    """§3.6 latency_p50 / latency_p95 (no bootstrap — not proportions)."""
    values = sorted(record.latency_ms for record in records)
    if not values:
        return {"latency_p50": None, "latency_p95": None}

    def pct(p: float) -> float:
        index = min(len(values) - 1, max(0, round(p * (len(values) - 1))))
        return values[index]

    return {"latency_p50": pct(0.50), "latency_p95": pct(0.95)}


def meme_regression(
    records: list[GenRecord], meme_count: int
) -> tuple[int, list[int]]:
    """§3.5 meme regression: how many memes of the fixed list one configuration
    reproduced, and which ones it never did.

    Returns ``(reproduced_count, missing_indices)`` — a **share numerator**, not
    a verdict (M3R-130). The old binary form ("every meme at least once") asked
    for luck: the list had no support floor, so most of it was n-grams seen in a
    single message and the set of misses drifted between runs. The verdict is
    now formed by the report, relative to the baseline, over every arm — this
    function only counts.
    """
    if meme_count == 0:
        return 0, []
    reproduced: set[int] = set()
    for record in records:
        if record.category == "meme-bait":
            reproduced |= record.meme_hits
    missing = [index for index in range(meme_count) if index not in reproduced]
    return len(reproduced), missing
