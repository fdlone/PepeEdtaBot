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


def metric_values(records: list[GenRecord]) -> dict[str, list[float] | None]:
    """Per-generation sample arrays for every §3 metric.

    Values are lists (bootstrap resamples them) or ``None`` = insufficient
    data in the current phase. Config-level ratios (distinct-N) are computed
    by :func:`distinct_n` on resampled reply sets instead.
    """
    successful = [record for record in records if record.success]
    accept_rates = [
        record.pool_size / (record.pool_size + record.rejected_count)
        for record in records
        if record.pool_size + record.rejected_count > 0
    ]
    non_copy = [record for record in successful if not record.is_copy]
    return {
        # §3.1
        "generation_success_rate": proportion_samples(records, "success"),
        "candidate_accept_rate": accept_rates or None,
        "mean_response_length": [float(len(r.reply_content)) for r in successful] or None,
        "unique_token_ratio": [
            len(set(r.reply_content)) / len(r.reply_content)
            for r in successful
            if r.reply_content
        ]
        or None,
        # §3.2
        "exact_context_copy_rate": proportion_samples(successful, "is_copy") or None,
        "repetition_rate": proportion_samples(successful, "has_repetition") or None,
        "cycle_detection_rate": proportion_samples(successful, "has_cycle") or None,
        "cycle_harm_rate": [
            1.0 if (r.has_cycle and r.has_repetition) else 0.0 for r in successful
        ]
        or None,
        # §3.3
        "context_affinity": [r.affinity for r in successful if r.affinity is not None] or None,
        "context_affinity_without_copy": [
            r.affinity for r in non_copy if r.affinity is not None
        ]
        or None,
        # §3.4 — seeded generation does not exist before Phase 5.
        "seeded_present_rate": None,
        "seeded_win_rate_given_present": None,
        # §3.5 — no temporal snapshot before Phase 3 (audit §10.1).
        "freshness_reflection": None,
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
) -> tuple[bool | None, list[int]]:
    """§3.5 ``meme_regression_pass``: every meme in the fixed list must be
    reproduced in at least one meme-bait generation. Returns the verdict and
    the indices of memes never reproduced (``None`` verdict when the list is
    empty — gate not applicable)."""
    if meme_count == 0:
        return None, []
    reproduced: set[int] = set()
    for record in records:
        if record.category == "meme-bait":
            reproduced |= record.meme_hits
    missing = [index for index in range(meme_count) if index not in reproduced]
    return not missing, missing
