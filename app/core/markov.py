from __future__ import annotations

import bisect
import logging
import math
import random
import re
from collections import Counter, OrderedDict
from collections.abc import Container, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

from app.core.context_state_matcher import ContextStateMatcher
from app.core.generation_telemetry import GenerationTelemetry
from app.core.interpolation import InterpolatedPool, OrderInterpolation
from app.core.lexicon import BAD_ENDING_WORDS, STOPWORDS
from app.core.markov_port import MarkovReadPort
from app.core.morphology import stem_folded, stem_token
from app.core.seed import SeedScore, is_scorable, score_seeds
from app.core.temporal import (
    BlendedPool,
    TemporalBlend,
    TransitionRow,
    short_observed,
)

logger = logging.getLogger("chat_markov")

TOKEN_RE = re.compile(r"\w+|[.,!?;:]", re.UNICODE)
PUNCT_SET = {".", ",", "!", "?", ";", ":"}
DEFAULT_GENERATION_ATTEMPT_BUDGET = 1
EXPLORATION_FLATTENING = 1.5
EXPLORATION_POWER_FLOOR = 0.02
# Floor under the entropy-adjusted temperature: a large negative gain can drive
# T to zero or below, and 1/T is taken right after.
_MIN_SAMPLING_TEMPERATURE = 0.01
SHORT_REPLY_MAX_CONTENT_TOKENS = 3
REJECTION_NO_STARTS = "no_starts"
REJECTION_NO_START_TRANSITION = "no_start_transition"
REJECTION_RESULT_TOO_SHORT = "result_too_short"
REJECTION_LOW_DIVERSITY = "low_diversity"
REJECTION_SHORT_CONTEXT_COPY = "short_context_copy"
REJECTION_CONTEXT_HEAVY = "context_heavy"
# M4 jumps become possible once the reply has this many tokens (lowered from 9
# so shorter replies can still drift to a new topic mid-generation).
JUMP_MIN_GENERATED_TOKENS = 5

# M4: connective phrases spliced in when a mid-generation jump drifts to a new
# topic, so the shift reads as a deliberate aside rather than a non-sequitur.
# Each is a token sequence; the leading comma binds it to the preceding text
# (detokenize attaches punctuation without a space). The words are common
# discourse markers, so they neither echo context nor look out of register.
JUMP_CONNECTIVE_TOKENS: tuple[tuple[str, ...], ...] = (
    (",", "кстати"),
    (",", "короче"),
    (",", "слушай"),
    (",", "хотя"),
    (",", "а", "вообще"),
    (",", "ну", "и"),
    (",", "да", "и"),
    (",", "а", "ещё"),
    (",", "причём"),
    (",", "ладно"),
    (",", "но", "вообще-то"),
    (",", "и", "да"),
)

# Silent splice: the aside starts a new sentence instead of hanging off a
# connective. Reads as a plain topic change («... . новая мысль») and is
# invisible to connective-frequency detection — the whole point: with the
# boosted context jumps 82% of winners carried a connective phrase from a
# six-item pool, a recognisable tic in a live chat (2026-07-15 measurement).
SILENT_SPLICE: tuple[str, ...] = (".",)
SILENT_SPLICE_PROBABILITY = 0.35

# One aside per reply reads as deliberate topic drift; two or more read as
# salad. Uncapped, the 0.12/step hazard gave 18% of selection winners >=2
# jumps, and those were exactly the incoherent long replies (avg 18-23 content
# tokens vs 6.3 without jumps). Extra length must come from the chain itself,
# not from splicing more topics.
JUMP_MAX_PER_REPLY = 1

# Insurance for a raised cap: a further jump may not fire until the reply has
# grown this many tokens past the previous splice, so asides cannot chain
# back-to-back. Dead while JUMP_MAX_PER_REPLY is 1.
JUMP_MIN_TOKENS_BETWEEN = 6

# Tokens dropped from the tail of the walk right before a connective splice: a
# dangling comma, conjunction or preposition at the splice point yields ",,",
# "хотя, хотя" and "писал на, а вообще"-style fragments. Includes the
# connective words themselves so "<...> хотя" + ", хотя <...>" cannot double
# up. Prepositions were missed in the first pass — live traces showed jumps
# firing mid-phrase on "на"/"с".
_JUMP_SPLICE_TRIM: frozenset[str] = frozenset(
    {",", "и", "а", "но", "ну", "что", "как"}
    | {"в", "на", "с", "со", "по", "у", "к", "ко", "о", "об", "за", "до",
       "из", "от", "под", "над", "при", "про", "без", "для", "через"}
    | {token for phrase in JUMP_CONNECTIVE_TOKENS for token in phrase}
)


def trim_splice_tail(generated: list[str]) -> None:
    """Drop dangling comma/conjunction tokens so a connective splices cleanly.

    Mutates ``generated`` in place; always leaves at least one token."""
    while len(generated) > 1 and generated[-1] in _JUMP_SPLICE_TRIM:
        generated.pop()


def pick_jump_connective(
    rng: random.Random,
    exclude: Iterable[tuple[str, ...]] = (),
) -> tuple[str, ...]:
    """A connective outside ``exclude``, so one reply does not reuse an aside
    marker (and the splice does not stutter into the jump target's first word).
    Falls back to the full pool when everything is excluded."""
    excluded = set(exclude)
    fresh = [
        phrase for phrase in JUMP_CONNECTIVE_TOKENS if phrase not in excluded
    ]
    return rng.choice(fresh or list(JUMP_CONNECTIVE_TOKENS))


def pick_splice_connective(
    rng: random.Random,
    exclude: Iterable[tuple[str, ...]] = (),
) -> tuple[str, ...]:
    """A splice marker: the silent sentence break with a fixed share, else a
    wordy connective outside ``exclude``. Both splice paths (M4 drift and the
    verbatim extension) draw from here so the connective tic dilutes evenly."""
    if rng.random() < SILENT_SPLICE_PROBABILITY:
        return SILENT_SPLICE
    return pick_jump_connective(rng, exclude=exclude)


def splice_marker_tokens(
    generated: list[str], connective: tuple[str, ...]
) -> list[str]:
    """Tokens to actually splice: the silent «.» is dropped when the tail
    already ends the sentence (trim_splice_tail removes commas but keeps
    terminal punctuation — «слово!.» must not happen)."""
    if (
        connective == SILENT_SPLICE
        and generated
        and generated[-1] in PUNCT_SET
    ):
        return []
    return list(connective)


def context_emission_tokens(state: tuple[str, str, str]) -> list[str]:
    """Tokens of a contextual start state worth emitting into the reply.

    At most the last two state tokens are emitted (caps the visible echo of the
    prompt), and leading stopwords/punctuation are dropped so the reply opens on
    the content part of the matched window: a context match on
    ``("кто", "гнойный", "пидор")`` emits ``["гнойный", "пидор"]`` and the chain
    continues from the full state. An empty result means the start stays hidden
    (the pre-emission Phase 4.1d behaviour).
    """
    tail = list(state[-2:])
    while tail and (tail[0] in PUNCT_SET or tail[0].casefold() in STOPWORDS):
        tail.pop(0)
    return tail

def pool_diagnostics(counts: Sequence[float]) -> tuple[float, float, int, float]:
    """Distribution diagnostics of one transition pool (M2R-010, TZ §6).

    Returns ``(entropy_bits, normalized_entropy, branching_factor,
    confidence)`` computed over the proportions of the weights the sampler
    actually uses — raw counts while the temporal blend is off, blended weights
    once it is on (M2R-210, design D5). Deliberately *not* the
    temperature-adjusted weights: the temperature is a response to this number,
    not an input to it.

    ``normalized_entropy`` is 0 when branching <= 1 (no uncertainty to
    normalize); ``confidence = 1 - normalized_entropy``. Pure math on the
    pool list: no SQL, no RNG.

    Blended weights are probabilities below 1, so the old ``max(count, 1)``
    floor would have flattened every pool to uniform. It is now a floor at 0,
    which leaves the integer path bit-identical: no chain row has ``cnt <= 0``
    (verified on the prod copy — min is 1 in all four tables) and nothing in
    the codebase deletes counts.
    """
    branching = len(counts)
    if branching <= 1:
        return 0.0, 0.0, branching, 1.0
    total = 0.0
    for count in counts:
        total += max(count, 0.0)
    if total <= 0.0:
        # No weight anywhere: every candidate is equally unsupported, which is
        # maximum uncertainty, not certainty.
        return math.log2(branching), 1.0, branching, 0.0
    entropy = 0.0
    for count in counts:
        p = max(count, 0.0) / total
        if p <= 0.0:
            continue
        entropy -= p * math.log2(p)
    normalized = entropy / math.log2(branching)
    # Float rounding can push H marginally past log2(B); clamp keeps the
    # confidence invariant 0 <= C <= 1 exact for property tests.
    normalized = max(0.0, min(1.0, normalized))
    return entropy, normalized, branching, 1.0 - normalized


@dataclass(frozen=True, slots=True)
class EntropySampling:
    """Entropy-aware sampling temperature (M2R-100, TZ §6).

    The sampler weights candidates as ``cnt ** power``, so ``power`` is an
    inverse temperature: ``T_base = 1/power``, and ``randomness_strength``
    keeps its established meaning as the scale of ``T_base``. This applies the
    per-pool term of TZ §6 — ``T = T_base·(1 + gain·(H_norm − pivot))``,
    clamped — and hands back a power for the existing machinery.

    The default instance is the neutral one: a generation that never builds a
    tuned instance behaves exactly like Markov 1.x.
    """

    gain: float = 0.0
    pivot: float = 0.5
    temp_min: float = 0.5
    temp_max: float = 12.0

    def power_for(self, power: float, normalized_entropy: float) -> float:
        """Entropy-adjusted frequency power for one pool.

        Zero gain returns the input unchanged through an explicit branch taken
        before any arithmetic. Relying on ``x * 1.0 == x`` would still be exact,
        but the clamp is not: a ``T_base`` outside ``[temp_min, temp_max]``
        would be pulled to a bound even at zero gain, and this phase's contract
        is that zero gain is byte-identical to 1.x. The early return makes that
        structural instead of a float coincidence. No random draw either way,
        so the RNG stream is untouched.
        """
        if self.gain == 0.0 or power <= 0.0:
            return power
        # Two individually valid /set calls can leave min above max; ordering
        # the pair beats pinning every step to a nonsense constant.
        low, high = (
            (self.temp_min, self.temp_max)
            if self.temp_min <= self.temp_max
            else (self.temp_max, self.temp_min)
        )
        low = max(low, _MIN_SAMPLING_TEMPERATURE)
        high = max(high, low)
        temperature = (1.0 / power) * (
            1.0 + self.gain * (normalized_entropy - self.pivot)
        )
        return 1.0 / max(low, min(high, temperature))


@dataclass(slots=True)
class _DiagnosticsAccumulator:
    """Per-attempt sums of step-pool diagnostics (M2R-010)."""

    entropy_bits_sum: float = 0.0
    normalized_entropy_sum: float = 0.0
    branching_sum: float = 0.0
    steps: int = 0
    min_confidence: float = 1.0
    # M2R-100: the temperature actually applied, so "the knob is doing nothing"
    # is readable from the numbers instead of inferred from the config.
    applied_temperature_sum: float = 0.0
    # M2R-210: how often the short layer had anything to say at all, and how far
    # the blend actually moved the distribution. A configured alpha proves
    # intent; these two prove effect.
    blend_covered_steps: int = 0
    blend_displacement_sum: float = 0.0
    # M2R-901: the same intent-versus-effect pair for the order interpolation.
    # A step counts as covered only when the projection actually ADDED a token:
    # "beta is set but the projection is just as sparse" and "beta is not set"
    # are different findings, and only this counter separates them.
    interp_covered_steps: int = 0
    interp_displacement_sum: float = 0.0

    def note_interp(self, merged: InterpolatedPool | None) -> None:
        """Record what the interpolation did at this step (nothing, when off)."""
        if merged is None:
            return
        if merged.added > 0:
            self.interp_covered_steps += 1
        self.interp_displacement_sum += merged.displacement

    def note_blend(self, blended: BlendedPool | None) -> None:
        """Record what the blend did at this step (nothing, when it is off)."""
        if blended is None:
            return
        if blended.displacement > 0.0:
            self.blend_covered_steps += 1
        self.blend_displacement_sum += blended.displacement

    def note_pool(self, counts: Sequence[float]) -> float:
        """Accumulate one pool's diagnostics; returns its normalized entropy.

        Returning the value is what lets M2R-100 sample from the same numbers
        the telemetry reports without computing them a second time per step.
        """
        entropy, normalized, branching, confidence = pool_diagnostics(counts)
        self.entropy_bits_sum += entropy
        self.normalized_entropy_sum += normalized
        self.branching_sum += branching
        self.steps += 1
        self.min_confidence = min(self.min_confidence, confidence)
        return normalized

    def note_temperature(self, power: float) -> None:
        """Record the temperature this step sampled at (``T = 1/power``).

        This is the entropy-adjusted temperature *before* the exploration roll
        flattens it further — deliberately, because it is the knob's own effect
        that needs to be readable, not the exploration channel's.
        """
        if power > 0.0:
            self.applied_temperature_sum += 1.0 / power


def _step_power(
    diagnostics: _DiagnosticsAccumulator | None,
    pool: Sequence[TransitionRow],
    base_power: float,
    entropy_sampling: EntropySampling,
    blended: BlendedPool | None = None,
) -> float:
    """Frequency power for one walk step: telemetry in, temperature out.

    Keeping both here is deliberate — the entropy the telemetry reports and the
    entropy the sampler consumes are the same number by construction, and the
    temperature actually applied is recorded rather than inferred.

    With the temporal blend on, both read the blended weights rather than the
    raw counts (design D5): entropy has to describe the distribution actually
    being sampled, otherwise Phase 2's temperature responds to a distribution
    that no longer exists.

    Telemetry (M2R-010) needs the diagnostics for every step anyway, so the
    accumulator is the source. Without one (test doubles pass no accumulator)
    entropy is computed only when sampling actually consumes it.
    """
    if diagnostics is None and entropy_sampling.gain == 0.0:
        normalized = 0.0
    else:
        weights: Sequence[float] = (
            blended.weights if blended is not None else [row[1] for row in pool]
        )
        normalized = (
            diagnostics.note_pool(weights)
            if diagnostics is not None
            else pool_diagnostics(weights)[1]
        )
    power = entropy_sampling.power_for(base_power, normalized)
    if diagnostics is not None:
        diagnostics.note_temperature(power)
    return power


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    attempts_used: int
    markov_order_used: int
    jump_count: int
    rejection_reason: str | None
    token_count: int
    start_source: str
    leading_punctuation_stripped: int = 0
    context_exact_matches: int = 0
    context_casefold_matches: int = 0
    hidden_context_fallbacks: int = 0
    # M2R-010: diagnostics of the winning attempt's step pools (walk steps
    # only; start pools are not sampled per step and stay uninstrumented).
    mean_entropy_bits: float = 0.0
    mean_normalized_entropy: float = 0.0
    mean_branching: float = 0.0
    min_confidence: float = 1.0
    diagnostic_steps: int = 0
    # M2R-100: temperature actually applied, averaged over the measured steps.
    mean_applied_temperature: float = 0.0
    # M2R-210: what the temporal blend actually did. ``applied_alpha`` is the
    # configured intent; the other two are the effect — the share of steps where
    # the short layer had any weight at all, and how far the blend moved the
    # sampled distribution away from the long layer alone. A knob that is set
    # but inert reads as alpha > 0 with both of these at 0, which is exactly the
    # distinction Phase 2 had to discover the hard way.
    applied_alpha: float = 0.0
    blend_step_coverage: float = 0.0
    mean_blend_displacement: float = 0.0
    # M2R-901: доля шагов, где проекция order-2 добавила токен, и среднее
    # расхождение слитого распределения с чистым P3.
    interp_step_coverage: float = 0.0
    mean_interp_displacement: float = 0.0


class _GenerationAttempt(NamedTuple):
    """Одна попытка генерации: текст плюс её счётчики для трассировки.

    Собирается только по именам: девять из десяти полей — однотипные
    счётчики, и позиционный вызов держался на памяти о порядке.
    ``GenerationTrace`` остаётся отдельным типом: это суммарная трассировка
    по всем попыткам (у неё есть ``attempts_used`` и нет текста), и её
    форму читают инструменты оценки.
    """

    text: str
    markov_order_used: int
    jump_count: int
    rejection_reason: str | None
    token_count: int
    start_source: str
    leading_punctuation_stripped: int = 0
    context_exact_matches: int = 0
    context_casefold_matches: int = 0
    hidden_context_fallbacks: int = 0
    mean_entropy_bits: float = 0.0
    mean_normalized_entropy: float = 0.0
    mean_branching: float = 0.0
    min_confidence: float = 1.0
    diagnostic_steps: int = 0
    # M2R-100: temperature actually applied, averaged over the measured steps.
    mean_applied_temperature: float = 0.0
    # M2R-210: see GenerationTrace for what these three mean.
    applied_alpha: float = 0.0
    blend_step_coverage: float = 0.0
    mean_blend_displacement: float = 0.0
    # M2R-901: доля шагов, где проекция order-2 добавила токен, и среднее
    # расхождение слитого распределения с чистым P3.
    interp_step_coverage: float = 0.0
    mean_interp_displacement: float = 0.0


class _ContextualStateSelection(NamedTuple):
    state: tuple[str, str, str]
    order: int
    match_kind: str


def remember_bounded[T](
    values: OrderedDict[T, None],
    value: T,
    limit: int,
) -> None:
    if value not in values:
        values[value] = None
    if len(values) > limit:
        values.popitem(last=False)


def tokenize(text: str, normalize_lower: bool = False) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    return [t.lower() for t in tokens] if normalize_lower else tokens


def detokenize(tokens: list[str], max_chars: int) -> str:
    if not tokens:
        return ""

    text = ""
    for token in tokens:
        if not text:
            candidate = token
        elif token in PUNCT_SET:
            candidate = text + token
        else:
            candidate = text + " " + token

        if len(candidate) > max_chars:
            break
        text = candidate

    return text.strip()


def escalated_randomness_strength(
    base_strength: float,
    attempt_index: int,
    total_attempts: int,
    max_strength: float = 3.0,
) -> float:
    if total_attempts <= 1:
        return max(0.0, min(max_strength, base_strength))

    base = max(0.0, min(max_strength, base_strength))
    if base >= max_strength:
        return base

    progress = attempt_index / (total_attempts - 1)
    return min(max_strength, base + (max_strength - base) * progress)


def context_start_probability(context_start_bias: float) -> float:
    """Probability of entering the contextual-start path: (bias-1)/bias, clamped to [0, 1)."""
    if context_start_bias <= 1.0:
        return 0.0
    return max(0.0, (context_start_bias - 1.0) / context_start_bias)


def build_windows(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    if size <= 0 or len(tokens) < size:
        return []
    return [tuple(tokens[idx : idx + size]) for idx in range(len(tokens) - size + 1)]


def content_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in PUNCT_SET]


def content_token_indexes(tokens: list[str]) -> list[int]:
    return [index for index, token in enumerate(tokens) if token not in PUNCT_SET]


def is_short_generated_reply(tokens: list[str]) -> bool:
    content = content_tokens(tokens)
    return 0 < len(content) <= SHORT_REPLY_MAX_CONTENT_TOKENS


def longest_shared_run(tokens_a: list[str], tokens_b: list[str]) -> int:
    """Длина самой длинной общей непрерывной последовательности токенов.

    Динамика по одной строке: ``run[j]`` — длина общего хвоста, кончающегося
    на ``tokens_a[i]`` и ``tokens_b[j]``. Прежняя форма досчитывала совпадение
    заново от каждой пары позиций (кубическая в худшем случае) и вызывается
    дважды на каждую завершённую попытку генерации.
    """
    if not tokens_a or not tokens_b:
        return 0

    best = 0
    previous = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        current = [0] * (len(tokens_b) + 1)
        for index_b, token_b in enumerate(tokens_b):
            if token_a == token_b:
                run = previous[index_b] + 1
                current[index_b + 1] = run
                if run > best:
                    best = run
        previous = current
    return best


def max_consecutive_run(tokens: list[str]) -> int:
    if not tokens:
        return 0

    best = 1
    current = 1
    for index in range(1, len(tokens)):
        if tokens[index] == tokens[index - 1]:
            current += 1
            if current > best:
                best = current
        else:
            current = 1
    return best


def has_degraded_recent_window(
    tokens: list[str],
    window_size: int = 8,
    min_window_tokens: int = 6,
    dominance_threshold: float = 0.75,
) -> bool:
    recent_content = content_tokens(tokens)[-window_size:]
    if len(recent_content) < min_window_tokens:
        return False

    if max_consecutive_run(recent_content) >= 4:
        return True

    counts = Counter(recent_content)
    return (
        len(counts) <= 2
        and (max(counts.values()) / len(recent_content)) >= dominance_threshold
    )


def find_repetitive_tail_start(
    tokens: list[str],
    min_tail_tokens: int = 5,
    min_prefix_tokens: int = 4,
    tail_scan_limit: int = 12,
    dominance_threshold: float = 0.7,
) -> int | None:
    content_indexes = content_token_indexes(tokens)
    if len(content_indexes) < min_prefix_tokens + min_tail_tokens:
        return None

    content = [tokens[index] for index in content_indexes]
    first_candidate = max(min_prefix_tokens, len(content) - tail_scan_limit)
    last_candidate = len(content) - min_tail_tokens
    for start_content_idx in range(first_candidate, last_candidate + 1):
        tail = content[start_content_idx:]
        counts = Counter(tail)
        if max_consecutive_run(tail) >= 4:
            return content_indexes[start_content_idx]
        if (
            len(counts) <= 2
            and (max(counts.values()) / len(tail)) >= dominance_threshold
        ):
            return content_indexes[start_content_idx]
    return None


def trim_repetitive_tail(tokens: list[str]) -> list[str]:
    trim_start = find_repetitive_tail_start(tokens)
    if trim_start is None:
        return tokens

    trimmed = tokens[:trim_start]
    while trimmed and trimmed[-1] in PUNCT_SET:
        trimmed.pop()
    return trimmed if len(content_tokens(trimmed)) >= 4 else tokens


_SENTENCE_END_PUNCT = {".", "!", "?"}
_NON_TERMINAL_PUNCT = PUNCT_SET - _SENTENCE_END_PUNCT


def trim_to_sentence_boundary(tokens: list[str], min_content_tokens: int = 4) -> list[str]:
    """Trim to the last sentence-ending punctuation (.!?) for natural endings."""
    if len(content_tokens(tokens)) < min_content_tokens:
        return tokens
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in _SENTENCE_END_PUNCT:
            trimmed = tokens[:i + 1]
            if len(content_tokens(trimmed)) >= min_content_tokens:
                return trimmed
    return tokens


def finalize_reply_ending(
    tokens: list[str],
    min_content_tokens: int = 4,
) -> list[str]:
    if len(content_tokens(tokens)) < min_content_tokens:
        return tokens[:]

    finalized = tokens[:]
    content_count = len(content_tokens(finalized))
    while finalized:
        last = finalized[-1]
        is_bad_word = last not in PUNCT_SET and last.lower() in BAD_ENDING_WORDS
        is_non_terminal_punct = last in _NON_TERMINAL_PUNCT
        if not (is_bad_word or is_non_terminal_punct):
            break
        if is_bad_word and content_count <= min_content_tokens:
            break
        finalized.pop()
        if is_bad_word:
            content_count -= 1

    if finalized and finalized[-1] not in _SENTENCE_END_PUNCT:
        finalized.append(".")
    return finalized


def strip_leading_punctuation(tokens: list[str]) -> list[str]:
    first_content_index = 0
    while (
        first_content_index < len(tokens)
        and tokens[first_content_index] in PUNCT_SET
    ):
        first_content_index += 1
    return tokens[first_content_index:]


def is_low_diversity_reply(
    tokens: list[str],
    min_total_tokens: int = 8,
    dominance_threshold: float = 0.8,
) -> bool:
    content = content_tokens(tokens)
    if len(content) < min_total_tokens:
        return False

    counts = Counter(content)
    if max_consecutive_run(content) >= 5:
        return True
    return (
        len(counts) <= 2
        and (max(counts.values()) / len(content)) >= dominance_threshold
    )


def is_context_heavy_reply(
    generated_tokens: list[str], context_tokens: list[str]
) -> bool:
    if len(generated_tokens) < 4 or not context_tokens:
        return False

    generated_content = content_tokens(generated_tokens)
    context_content = content_tokens(context_tokens)
    if len(generated_content) < 4 or len(context_content) < 3:
        return False

    context_token_set = set(context_content)
    overlap_count = sum(1 for token in generated_content if token in context_token_set)
    overlap_ratio = overlap_count / len(generated_content)
    shared_run = longest_shared_run(generated_content, context_content)
    uses_only_context_tokens = all(
        token in context_token_set for token in generated_content
    )
    has_local_loops = len(set(generated_content)) <= max(2, len(generated_content) // 2)

    if uses_only_context_tokens and has_local_loops:
        return True
    if overlap_ratio >= 0.92 and shared_run >= 5:
        return True
    if shared_run >= max(4, len(generated_content) - 1):
        return True
    return False


def is_short_context_copy(generated_tokens: list[str], context_tokens: list[str]) -> bool:
    generated_content = content_tokens(generated_tokens)
    context_content = content_tokens(context_tokens)
    if not generated_content or not context_content:
        return False
    if len(generated_content) > SHORT_REPLY_MAX_CONTENT_TOKENS:
        return False

    shared_run = longest_shared_run(generated_content, context_content)
    return shared_run >= len(generated_content)


@dataclass(frozen=True, slots=True)
class FinalizedCandidate:
    """Outcome of the shared tail pipeline and form gates.

    ``rejection_reason`` is None exactly when the candidate is usable; ``text``
    and ``tokens`` are then its final form. On rejection both are empty and the
    reason is one of the ``REJECTION_*`` constants.
    """

    text: str
    tokens: list[str]
    rejection_reason: str | None
    leading_punctuation_stripped: int


def finalize_candidate_tokens(
    generated: list[str],
    *,
    max_chars: int,
    context_tokens: list[str],
) -> FinalizedCandidate:
    """Run the tail pipeline and the four form gates over raw tokens.

    Single source of truth for "a candidate is finished and shaped". Every
    branch that assembles a candidate goes through here — the main walk via
    ``_finalize_attempt``, seeded assembly via ``ResponseGenerator``. A branch
    that skips it does not merely look different: it forfeits
    ``CLEAN_END_BONUS`` and risks ``BAD_ENDING_PENALTY``, so the scorer marks it
    down for a defect in its construction rather than for its content. That is
    exactly what happened to the seeded branch (map §3.4), and a second copy of
    these steps would let it happen again silently.
    """
    generated = trim_repetitive_tail(generated)
    generated = trim_to_sentence_boundary(generated)
    generated = finalize_reply_ending(generated)
    finalized_token_count = len(generated)
    generated = strip_leading_punctuation(generated)
    stripped = finalized_token_count - len(generated)
    text = detokenize(generated, max_chars=max_chars)

    def reject(reason: str) -> FinalizedCandidate:
        return FinalizedCandidate(
            text="",
            tokens=[],
            rejection_reason=reason,
            leading_punctuation_stripped=stripped,
        )

    if len(text) < 5:
        return reject(REJECTION_RESULT_TOO_SHORT)
    # Без normalize_lower намеренно: текст ответа уже в том регистре, в
    # котором чат учил модель (флаг применяется на входе, при токенизации
    # сообщения), и контекст сюда приходит в том же соглашении. Приведение
    # к нижнему регистру здесь было бы не выравниванием, а сменой правила:
    # в профиле с сохранением регистра гейты формы стали бы
    # регистронезависимыми. Проверено: в дефолтном профиле это no-op —
    # хеш генерации не меняется (tools/generation_hash.py).
    tokens = tokenize(text)
    is_short_reply = is_short_generated_reply(tokens)
    if not is_short_reply and is_low_diversity_reply(tokens):
        return reject(REJECTION_LOW_DIVERSITY)
    if is_short_reply:
        if is_short_context_copy(tokens, context_tokens):
            return reject(REJECTION_SHORT_CONTEXT_COPY)
    elif is_context_heavy_reply(tokens, context_tokens):
        return reject(REJECTION_CONTEXT_HEAVY)
    return FinalizedCandidate(
        text=text,
        tokens=tokens,
        rejection_reason=None,
        leading_punctuation_stripped=stripped,
    )


def context_decay(step_index: int) -> float:
    return max(0.25, 0.92 ** step_index)


def exploration_adjusted_power(power: float, explore_probability: float) -> float:
    exploration = max(0.0, min(1.0, explore_probability))
    return max(
        EXPLORATION_POWER_FLOOR,
        max(0.0, power) * (1.0 - EXPLORATION_FLATTENING * exploration),
    )


def sampled_exploration(
    explore_probability: float,
    exploring: bool,
    rng: random.Random,
) -> float:
    if not exploring:
        return 0.0
    # A triangular temperature sample avoids both an abrupt uniform branch and
    # extreme single-draw temperatures while remaining deterministic per RNG.
    sample_mean = (rng.random() + rng.random()) / 2.0
    return max(0.0, min(1.0, explore_probability)) * sample_mean


def _roll_exploration(
    explore_probability: float,
    power: float,
    rng: random.Random,
) -> tuple[bool, float]:
    """Roll the exploring branch and derive the frequency power for this pick."""
    exploring = rng.random() < explore_probability
    frequency_power = exploration_adjusted_power(
        power,
        sampled_exploration(explore_probability, exploring, rng),
    )
    return exploring, frequency_power


def exploration_weighted_choice[T](
    population: list[T],
    weights: list[float],
    rng: random.Random,
) -> T:
    index = min(
        range(len(population)),
        key=lambda candidate_index: rng.expovariate(weights[candidate_index]),
    )
    return population[index]


def weighted_index_choice(
    weights: list[float],
    *,
    exploring: bool,
    rng: random.Random,
) -> int:
    """``weighted_population_choice`` над позициями вместо самих элементов.

    Тот же розыгрыш теми же обращениями к генератору случайных чисел: и
    ``rng.choices``, и экспоненциальный трюк работают по индексам, а элементы
    только сопровождают вес. Разница в том, что вызывающему не нужно строить
    список кандидатов: на живом чате это 4.5 тысячи кортежей, собираемых
    заново на каждой попытке генерации ради одного победителя.
    """
    if exploring:
        return min(
            range(len(weights)),
            key=lambda candidate_index: rng.expovariate(weights[candidate_index]),
        )
    return rng.choices(population=range(len(weights)), weights=weights, k=1)[0]


def weighted_population_choice[T](
    population: list[T],
    weights: list[float],
    *,
    exploring: bool,
    rng: random.Random,
) -> T:
    """Pick one item from a weighted population.

    Exploring mode spreads probability via ``exploration_weighted_choice``;
    otherwise samples proportionally with ``rng.choices``. Centralises the
    ``if exploring`` branch repeated across contextual-state selection.
    """
    if exploring:
        return exploration_weighted_choice(population, weights, rng)
    return rng.choices(population=population, weights=weights, k=1)[0]


def weighted_next_choice(
    items: Sequence[TransitionRow],
    explore_probability: float,
    power: float,
    rng: random.Random,
    context_token_set: set[str] | None = None,
    context_pairs: Container[tuple[str, ...]] | None = None,
    context_triplets: Container[tuple[str, ...]] | None = None,
    current_state: tuple[str, ...] | None = None,
    context_bias: float = 1.0,
    step_index: int = 0,
    recent_tokens: list[str] | None = None,
    seen_pairs: Container[tuple[str, ...]] | None = None,
    seen_triplets: Container[tuple[str, ...]] | None = None,
    repetition_penalty_strength: float = 1.0,
    base_weights: Sequence[float] | None = None,
) -> str:
    # ПРЕДУСЛОВИЕ: items упорядочены по токену. Его обеспечивает источник —
    # запросы переходов идут с ORDER BY, а фильтрация порядок сохраняет (см.
    # test_transition_rows_are_ordered). Пересортировка здесь стоила ~3 с на
    # 160 генераций на копии прода и не меняла ни одного результата: порядок
    # уже тот же самый.
    ordered_items = items
    exploring, frequency_power = _roll_exploration(explore_probability, power, rng)
    step_bias = 1.0 + (max(1.0, context_bias) - 1.0) * context_decay(step_index)
    penalty_strength = max(0.0, repetition_penalty_strength)
    recent_tokens = recent_tokens or []
    weights: list[float] = []
    for index, row in enumerate(ordered_items):
        token = row[0]
        # ``base_weights`` is the temporal blend's output (M2R-210). None means
        # the blend is off and the raw long count is the weight — the same
        # arithmetic as before the blend existed, which is what makes the
        # neutral configuration byte-identical rather than merely close.
        base = max(row[1], 1) if base_weights is None else base_weights[index]
        weight = base**frequency_power
        if context_token_set and token in context_token_set:
            weight *= step_bias
        if current_state and context_pairs and len(current_state) >= 1:
            if (current_state[-1], token) in context_pairs:
                weight *= 1.0 + (step_bias - 1.0) * 1.10
        if current_state and context_triplets and len(current_state) >= 2:
            if (current_state[-2], current_state[-1], token) in context_triplets:
                weight *= 1.0 + (step_bias - 1.0) * 1.25

        # Penalize local loops and repeated n-grams so reply-context bias does not
        # collapse into token spam.
        if recent_tokens:
            repeat_count = recent_tokens.count(token)
            if repeat_count > 0:
                weight /= 1.0 + repeat_count * 0.85 * penalty_strength
            if token == recent_tokens[-1]:
                weight *= max(0.01, 1.0 - 0.96 * penalty_strength)
            elif len(recent_tokens) >= 2 and token == recent_tokens[-2]:
                weight *= max(0.05, 1.0 - 0.70 * penalty_strength)

        if current_state and seen_pairs and len(current_state) >= 1:
            if (current_state[-1], token) in seen_pairs:
                weight *= max(0.05, 1.0 - 0.65 * penalty_strength)
        if current_state and seen_triplets and len(current_state) >= 2:
            if (current_state[-2], current_state[-1], token) in seen_triplets:
                weight *= max(0.01, 1.0 - 0.94 * penalty_strength)

        # The 0.01 floor keeps a heavily penalized candidate from reaching zero.
        # It is calibrated against raw counts, which are >= 1; blended weights
        # are probabilities, where a legitimate candidate in a 200-token pool
        # sits below 0.01 and the same floor would flatten the tail into a
        # plateau. The blended path therefore floors at an epsilon that only
        # guards against zero, which is all the floor was ever for.
        weight = max(weight, 0.01 if base_weights is None else 1e-12)
        weights.append(weight)
    index = weighted_index_choice(weights, exploring=exploring, rng=rng)
    return ordered_items[index][0]


def weighted_start2_choice(
    items: list[tuple[str, str, int]],
    explore_probability: float,
    power: float,
    rng: random.Random,
) -> tuple[str, str]:
    # Предусловие то же, что у weighted_next_choice: items упорядочены по
    # (w1, w2) — так их отдаёт запрос стартов.
    ordered_items = items
    exploring, frequency_power = _roll_exploration(explore_probability, power, rng)
    weights = [max(cnt, 1) ** frequency_power for _, _, cnt in ordered_items]
    index = weighted_index_choice(weights, exploring=exploring, rng=rng)
    w1, w2, _ = ordered_items[index]
    return w1, w2


def context_start_stems(context_tokens: list[str]) -> frozenset[str]:
    """Stems of the context's informative tokens, for start affinity."""
    return frozenset(
        stem_token(token)
        for token in (raw.casefold() for raw in content_tokens(context_tokens))
        if token not in STOPWORDS
    )


def weighted_start3_choice(
    items: list[tuple[str, str, str, int]],
    explore_probability: float,
    power: float,
    rng: random.Random,
    *,
    context_stems: frozenset[str] | None = None,
    context_start_affinity: float = 1.0,
) -> tuple[str, str, str]:
    # Предусловие то же: items упорядочены по (w1, w2, w3). На живом чате это
    # 4.5 тысячи стартов, и сортировка выполнялась на каждой из десяти попыток
    # генерации — самая дорогая строка горячего пути после самого отбора.
    ordered_items = items
    exploring, frequency_power = _roll_exploration(explore_probability, power, rng)
    weights = [max(cnt, 1) ** frequency_power for _, _, _, cnt in ordered_items]
    if context_stems and context_start_affinity > 1.0:
        # Answers live in starts, not in continuations: a contextual anchor can
        # only retrace what followed the question in the corpus, while a start
        # like «слава гнойный пидор» that *shares stems* with the question is
        # the reply we want — boost it exponentially per shared stem so it
        # surfaces among thousands of unrelated starts. A start whose stems are
        # ALL contained in the context gets no boost: it re-asks the question
        # («кто гнойный пидор» is itself a learned start and out-boosted the
        # actual answers 10:1 — same parrot problem as the scorer's echo guard).
        boosted: list[float] = []
        for weight, (w1, w2, w3, _) in zip(weights, ordered_items, strict=True):
            state_stems = {
                stem_folded(w1),
                stem_folded(w2),
                stem_folded(w3),
            }
            shared = len(state_stems & context_stems)
            if shared and not state_stems <= context_stems:
                weight *= context_start_affinity**shared
            boosted.append(weight)
        weights = boosted
    index = weighted_index_choice(weights, exploring=exploring, rng=rng)
    w1, w2, w3, _ = ordered_items[index]
    return w1, w2, w3


def _fold_transition(
    rows: list[TransitionRow],
    token: str,
    delta: int,
    now: int,
    half_life_days: float,
) -> list[TransitionRow]:
    """A copy of a cached transition list with ``(token, +delta)`` upserted.

    Copy-on-write on purpose: a walk holds its fetched pool across awaits, and
    the old wipe policy gave it snapshot semantics — mutating the shared list
    would let an interleaved learn change a pool mid-generation. The copy
    mirrors a SQL read with ``ORDER BY <token column>``; sampling relies on
    that order, so inserts go through bisection. Python's str comparison
    (code points) agrees with SQLite's default BINARY collation (UTF-8 bytes)
    — UTF-8 preserves code-point order.

    Both layers are folded, and the short one is folded whatever the blend is
    set to (M2R-210): learning maintains the temporal record so that turning
    the blend on later finds data rather than an empty layer. The arithmetic is
    the same helper the SQL writer uses, which is what keeps a folded cache
    equal to a fresh read.
    """
    updated = rows.copy()
    index = bisect.bisect_left(updated, token, key=lambda row: row[0])
    if index < len(updated) and updated[index][0] == token:
        _, count, s_value, s_updated_at = updated[index]
        updated[index] = (
            token,
            count + delta,
            short_observed(s_value, s_updated_at, now, half_life_days, delta),
            now,
        )
    else:
        updated.insert(index, (token, delta, float(delta), now))
    return updated


def _fold_start_row[T: tuple[Any, ...]](
    rows: list[T], words: tuple[str, ...]
) -> list[T]:
    """A copy of a cached starts list with the start row upserted.

    Ordered by the word prefix, same collation and copy-on-write arguments as
    :func:`_fold_transition`. A learned message contributes exactly one
    observation of its start state.
    """
    updated = rows.copy()
    index = bisect.bisect_left(updated, words, key=lambda row: tuple(row[:-1]))
    if index < len(updated) and tuple(updated[index][:-1]) == words:
        updated[index] = cast("T", (*words, updated[index][-1] + 1))
    else:
        updated.insert(index, cast("T", (*words, 1)))
    return updated


@dataclass(slots=True)
class MarkovGenerator:
    # Порт, а не конкретное хранилище: ядру достаточно чтения цепи.
    db: MarkovReadPort
    max_steps: int = 90
    cache_limit: int = 1024

    _cache3: OrderedDict[tuple[int, str, str, str], list[TransitionRow]] = field(
        default_factory=OrderedDict, init=False
    )
    _cache2: OrderedDict[tuple[int, str, str], list[TransitionRow]] = field(
        default_factory=OrderedDict, init=False
    )
    _cache_starts3: OrderedDict[int, list[tuple[str, str, str, int]]] = field(
        default_factory=OrderedDict, init=False
    )
    _cache_starts2: OrderedDict[int, list[tuple[str, str, int]]] = field(
        default_factory=OrderedDict, init=False
    )
    _context_state_matcher: ContextStateMatcher = field(init=False)
    telemetry: GenerationTelemetry = field(
        default_factory=GenerationTelemetry, init=False
    )

    def __post_init__(self) -> None:
        self._context_state_matcher = ContextStateMatcher(
            self.db,
            cache_limit=self.cache_limit,
        )

    def invalidate_all_caches(self) -> None:
        """Drop every chat's cached distributions.

        Used when a change invalidates the stored weights process-wide — so far
        only a global half-life change, which zeroes the short layer everywhere
        (TZ §7.2). Rare and coarse on purpose: correctness first, and the caches
        refill on demand.
        """
        self._cache3.clear()
        self._cache2.clear()
        self._cache_starts3.clear()
        self._cache_starts2.clear()
        self._context_state_matcher.invalidate_all_caches()

    def invalidate_chat_cache(self, chat_id: int) -> None:
        # Each cache is unrolled separately so its concrete key type is preserved
        # (a shared loop would widen the key to a union and break .pop typing).
        for key3 in [k for k in self._cache3 if k[0] == chat_id]:
            self._cache3.pop(key3, None)
        for key2 in [k for k in self._cache2 if k[0] == chat_id]:
            self._cache2.pop(key2, None)
        self._cache_starts3.pop(chat_id, None)
        self._cache_starts2.pop(chat_id, None)
        self._context_state_matcher.invalidate_chat_cache(chat_id)

    def apply_learning_deltas(
        self,
        chat_id: int,
        tokens: list[str],
        now: int = 0,
        half_life_days: float = 3.0,
    ) -> None:
        """Fold one learned message into the cached distributions (M2R-030).

        Mirrors the delta construction of ``save_message_and_update_model``
        (same n-gram counters, same start tuples, and since M2R-210 the same
        temporal arithmetic at the same ``now``) so a folded cache equals a
        fresh SQL read — the equivalence is enforced by tests, and the
        ``markov_cache_incremental=false`` knob falls back to
        :meth:`invalidate_chat_cache` entirely. Only cached keys are touched;
        cold keys stay cold and read fresh rows on demand.
        """
        if len(tokens) < 2:
            return
        trans3: Counter[tuple[str, str, str, str]] = Counter(
            (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])
            for i in range(len(tokens) - 3)
        )
        trans2: Counter[tuple[str, str, str]] = Counter(
            (tokens[i], tokens[i + 1], tokens[i + 2])
            for i in range(len(tokens) - 2)
        )
        for (w1, w2, w3, w4), delta in trans3.items():
            key3 = (chat_id, w1, w2, w3)
            cached3 = self._cache3.get(key3)
            if cached3 is not None:
                self._cache3[key3] = _fold_transition(
                    cached3, w4, delta, now, half_life_days
                )
        for (w1, w2, w3), delta in trans2.items():
            key2 = (chat_id, w1, w2)
            cached2 = self._cache2.get(key2)
            if cached2 is not None:
                self._cache2[key2] = _fold_transition(
                    cached2, w3, delta, now, half_life_days
                )
        starts2 = self._cache_starts2.get(chat_id)
        if starts2 is not None:
            self._cache_starts2[chat_id] = _fold_start_row(
                starts2, (tokens[0], tokens[1])
            )
        if len(tokens) >= 3:
            starts3 = self._cache_starts3.get(chat_id)
            if starts3 is not None:
                self._cache_starts3[chat_id] = _fold_start_row(
                    starts3, (tokens[0], tokens[1], tokens[2])
                )
        state3_deltas: dict[tuple[str, ...], int] = {}
        for (w1, w2, w3, _w4), delta in trans3.items():
            state = (w1, w2, w3)
            state3_deltas[state] = state3_deltas.get(state, 0) + delta
        state2_deltas: dict[tuple[str, ...], int] = {}
        for (w1, w2, _w3), delta in trans2.items():
            state2 = (w1, w2)
            state2_deltas[state2] = state2_deltas.get(state2, 0) + delta
        self._context_state_matcher.apply_state_deltas(chat_id, 3, state3_deltas)
        self._context_state_matcher.apply_state_deltas(chat_id, 2, state2_deltas)

    def _touch_cache[K, V](
        self,
        cache: OrderedDict[K, V],
        key: K,
        value: V,
    ) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.cache_limit:
            cache.popitem(last=False)

    async def _get_starts3(self, chat_id: int) -> list[tuple[str, str, str, int]]:
        if chat_id in self._cache_starts3:
            self.telemetry.note_cache(hit=True)
            return self._cache_starts3[chat_id]
        self.telemetry.note_cache(hit=False)
        rows = await self.db.get_starts3(chat_id)
        self._touch_cache(self._cache_starts3, chat_id, rows)
        return rows

    async def _get_starts2(self, chat_id: int) -> list[tuple[str, str, int]]:
        if chat_id in self._cache_starts2:
            self.telemetry.note_cache(hit=True)
            return self._cache_starts2[chat_id]
        self.telemetry.note_cache(hit=False)
        rows = await self.db.get_starts(chat_id)
        self._touch_cache(self._cache_starts2, chat_id, rows)
        return rows

    async def _get3(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> list[TransitionRow]:
        key = (chat_id, w1, w2, w3)
        if key in self._cache3:
            self.telemetry.note_cache(hit=True)
            self._cache3.move_to_end(key)
            return self._cache3[key]
        self.telemetry.note_cache(hit=False)
        rows = await self.db.get_transitions3(chat_id, w1, w2, w3)
        self._touch_cache(self._cache3, key, rows)
        return rows

    def transition_was_available(
        self, chat_id: int, state: tuple[str, ...], token: str
    ) -> bool:
        """Did the chain hold a transition from ``state`` to ``token``? (M2R-320)

        Answered strictly from the pools already cached by walks — never a
        query: collocation scoring runs per candidate and must not touch the
        database (design D4). An unknown pool answers False, which withholds
        the break penalty rather than firing it — missing evidence must not
        punish the candidate.
        """
        if len(state) >= 2:
            pool2 = self._cache2.get((chat_id, state[-2], state[-1]))
            if pool2 is not None:
                return any(row[0] == token for row in pool2)
        if len(state) >= 3:
            pool3 = self._cache3.get((chat_id, state[-3], state[-2], state[-1]))
            if pool3 is not None:
                return any(row[0] == token for row in pool3)
        return False

    async def _get2(self, chat_id: int, w1: str, w2: str) -> list[TransitionRow]:
        key = (chat_id, w1, w2)
        if key in self._cache2:
            self.telemetry.note_cache(hit=True)
            self._cache2.move_to_end(key)
            return self._cache2[key]
        self.telemetry.note_cache(hit=False)
        rows = await self.db.get_transitions(chat_id, w1, w2)
        self._touch_cache(self._cache2, key, rows)
        return rows

    async def _build_exact3_candidates(
        self,
        chat_id: int,
        windows3: list[tuple[str, ...]],
        total3: int,
        frequency_power: float,
    ) -> list[tuple[tuple[str, str, str], float, int]]:
        """Exact 3-gram context windows that have stored transitions, weighted
        by transition count and recency."""
        candidates3: list[tuple[tuple[str, str, str], float, int]] = []
        for index, window in enumerate(windows3):
            transitions = await self._get3(chat_id, window[0], window[1], window[2])
            if transitions:
                transition_count = sum(row[1] for row in transitions)
                recency_bonus = 1.0 + ((index + 1) / total3) * 0.35
                weight = max(transition_count, 1) ** frequency_power * recency_bonus
                candidates3.append(
                    ((window[0], window[1], window[2]), weight, transition_count)
                )
        return candidates3

    async def _build_exact2_candidates(
        self,
        chat_id: int,
        windows2: list[tuple[str, ...]],
        total2: int,
        frequency_power: float,
    ) -> list[tuple[tuple[str, str], list[TransitionRow], float]]:
        """Exact 2-gram context windows that have stored transitions, carrying
        their transitions for the follow-up step, weighted by count and
        recency."""
        candidates2: list[tuple[tuple[str, str], list[TransitionRow], float]] = []
        for index, window in enumerate(windows2):
            transitions = await self._get2(chat_id, window[0], window[1])
            if transitions:
                transition_count = sum(row[1] for row in transitions)
                recency_bonus = 1.0 + ((index + 1) / total2) * 0.30
                weight = max(transition_count, 1) ** frequency_power * recency_bonus
                candidates2.append(((window[0], window[1]), transitions, weight))
        return candidates2

    async def _build_casefold3_candidates(
        self,
        chat_id: int,
        windows3: list[tuple[str, ...]],
        total3: int,
        frequency_power: float,
    ) -> list[tuple[tuple[str, str, str], float, int]]:
        """Casefold 3-gram start candidates via the context-state matcher,
        weighted by count and recency. Each entry keeps its transition count.
        """
        candidates: list[tuple[tuple[str, str, str], float, int]] = []
        for index, window in enumerate(windows3):
            matches = await self._context_state_matcher.match(chat_id, window, 3)
            recency_bonus = 1.0 + ((index + 1) / total3) * 0.35
            for match in matches:
                if match.match_kind != "casefold":
                    continue
                state3 = (match.state[0], match.state[1], match.state[2])
                weight = (
                    max(match.transition_count, 1) ** frequency_power
                    * recency_bonus
                )
                candidates.append((state3, weight, match.transition_count))
        return candidates

    async def _build_casefold2_candidates(
        self,
        chat_id: int,
        windows2: list[tuple[str, ...]],
        total2: int,
        frequency_power: float,
    ) -> list[tuple[tuple[str, str], list[TransitionRow], float]]:
        """Casefold 2-gram start candidates that still have stored
        transitions."""
        additions: list[tuple[tuple[str, str], list[TransitionRow], float]] = []
        for index, window in enumerate(windows2):
            matches = await self._context_state_matcher.match(chat_id, window, 2)
            recency_bonus = 1.0 + ((index + 1) / total2) * 0.30
            for match in matches:
                if match.match_kind != "casefold":
                    continue
                state2 = (match.state[0], match.state[1])
                transitions = await self._get2(chat_id, state2[0], state2[1])
                if not transitions:
                    continue
                weight = (
                    max(match.transition_count, 1) ** frequency_power
                    * recency_bonus
                )
                additions.append((state2, transitions, weight))
        return additions

    @staticmethod
    def _select_state3(
        candidates: list[tuple[tuple[str, str, str], float, int]],
        match_kind: str,
        *,
        exploring: bool,
        rng: random.Random,
    ) -> _ContextualStateSelection:
        """Weighted pick of a 3-gram start state from ``(state, weight, count)``
        candidates."""
        population = [state for state, _, _ in candidates]
        weights = [weight for _, weight, _ in candidates]
        selected = weighted_population_choice(
            population, weights, exploring=exploring, rng=rng
        )
        return _ContextualStateSelection(selected, 3, match_kind)

    async def _select_contextual_state(
        self,
        chat_id: int,
        context_tokens: list[str],
        start_explore: float,
        start_power: float,
        next_explore: float,
        next_power: float,
        context_token_set: set[str],
        context_pairs: set[tuple[str, ...]],
        context_triplets: set[tuple[str, ...]],
        context_bias: float,
        repetition_penalty_strength: float,
        fuzzy_context_casefold: bool,
        rng: random.Random,
    ) -> _ContextualStateSelection | None:
        exploring, frequency_power = _roll_exploration(
            start_explore, start_power, rng
        )
        windows3 = build_windows(context_tokens, 3)
        total3 = len(windows3)
        candidates3 = await self._build_exact3_candidates(
            chat_id, windows3, total3, frequency_power
        )

        if candidates3:
            return self._select_state3(
                candidates3, "exact", exploring=exploring, rng=rng
            )

        windows2 = build_windows(context_tokens, 2)
        total2 = len(windows2)
        candidates2 = await self._build_exact2_candidates(
            chat_id, windows2, total2, frequency_power
        )

        match_kind = "exact"
        if not candidates2 and fuzzy_context_casefold:
            casefold_candidates3 = await self._build_casefold3_candidates(
                chat_id, windows3, total3, frequency_power
            )
            if casefold_candidates3:
                return self._select_state3(
                    casefold_candidates3, "casefold", exploring=exploring, rng=rng
                )

            additions = await self._build_casefold2_candidates(
                chat_id, windows2, total2, frequency_power
            )
            candidates2.extend(additions)
            match_kind = "casefold"

        if not candidates2:
            return None
        population2 = list(range(len(candidates2)))
        weights2 = [weight for _, _, weight in candidates2]
        selected_index = weighted_population_choice(
            population2, weights2, exploring=exploring, rng=rng
        )
        (w1, w2), variants, _ = candidates2[selected_index]
        w3 = weighted_next_choice(
            variants,
            next_explore,
            next_power,
            rng,
            context_token_set=context_token_set,
            context_pairs=context_pairs,
            context_triplets=context_triplets,
            current_state=(w1, w2),
            context_bias=context_bias,
            step_index=0,
            repetition_penalty_strength=repetition_penalty_strength,
        )
        return _ContextualStateSelection((w1, w2, w3), 2, match_kind)

    async def generate_text(
        self,
        chat_id: int,
        max_chars: int,
        *,
        max_tokens: int = 45,
        seed_tokens: list[str] | None = None,
        context_tokens: list[str] | None = None,
        context_bias: float = 1.0,
        context_start_bias: float = 1.0,
        context_start_affinity: float = 1.0,
        randomness_strength: float = 1.0,
        repetition_penalty_strength: float = 1.0,
        markov_order: int = 3,
        enable_backoff: bool = True,
        fuzzy_context_casefold: bool = False,
        jump_probability: float = 0.0,
        context_jump_boost: float = 1.0,
        order_mix_probability: float = 0.0,
        context_anchor_splice_probability: float = 0.0,
        entropy_sampling: EntropySampling = EntropySampling(),
        temporal_blend: TemporalBlend = TemporalBlend(),
        interpolation: OrderInterpolation = OrderInterpolation(),
        now: int = 0,
        rng: random.Random | None = None,
        attempt_budget: int = DEFAULT_GENERATION_ATTEMPT_BUDGET,
    ) -> str:
        """Текст без трассировки — форма для тестов и инструментов.

        Рабочий путь ходит через ``generate_text_with_trace``: ему нужна
        трассировка. Обёртка оставлена намеренно (её единственные
        вызывающие — тесты и харнессы), чтобы им не разбирать кортеж на
        каждом вызове; это не мёртвый код, а сокращённая форма API.
        """
        text, _ = await self.generate_text_with_trace(
            chat_id=chat_id,
            max_chars=max_chars,
            max_tokens=max_tokens,
            seed_tokens=seed_tokens,
            context_tokens=context_tokens,
            context_bias=context_bias,
            context_start_bias=context_start_bias,
            context_start_affinity=context_start_affinity,
            randomness_strength=randomness_strength,
            repetition_penalty_strength=repetition_penalty_strength,
            markov_order=markov_order,
            enable_backoff=enable_backoff,
            fuzzy_context_casefold=fuzzy_context_casefold,
            jump_probability=jump_probability,
            context_jump_boost=context_jump_boost,
            order_mix_probability=order_mix_probability,
            context_anchor_splice_probability=context_anchor_splice_probability,
            entropy_sampling=entropy_sampling,
            temporal_blend=temporal_blend,
            interpolation=interpolation,
            now=now,
            rng=rng,
            attempt_budget=attempt_budget,
        )
        return text

    async def generate_text_with_trace(
        self,
        chat_id: int,
        max_chars: int,
        # Keyword-only past this point: what follows is a run of same-typed
        # knobs — three of them named context_*bias*/context_*affinity* — that
        # a positional call could silently transpose. Every caller already
        # passes them by name, so this pins the existing convention rather
        # than changing any behaviour.
        *,
        max_tokens: int = 45,
        seed_tokens: list[str] | None = None,
        context_tokens: list[str] | None = None,
        context_bias: float = 1.0,
        context_start_bias: float = 1.0,
        context_start_affinity: float = 1.0,
        randomness_strength: float = 1.0,
        repetition_penalty_strength: float = 1.0,
        markov_order: int = 3,
        enable_backoff: bool = True,
        fuzzy_context_casefold: bool = False,
        jump_probability: float = 0.0,
        context_jump_boost: float = 1.0,
        order_mix_probability: float = 0.0,
        context_anchor_splice_probability: float = 0.0,
        entropy_sampling: EntropySampling = EntropySampling(),
        temporal_blend: TemporalBlend = TemporalBlend(),
        interpolation: OrderInterpolation = OrderInterpolation(),
        now: int = 0,
        rng: random.Random | None = None,
        attempt_budget: int = DEFAULT_GENERATION_ATTEMPT_BUDGET,
    ) -> tuple[str, GenerationTrace]:
        generation_rng = rng or random.Random()
        total_attempts = max(1, attempt_budget)
        attempts: list[_GenerationAttempt] = []
        for attempt_index in range(total_attempts):
            attempt_randomness_strength = escalated_randomness_strength(
                randomness_strength,
                attempt_index,
                total_attempts,
            )
            attempt = await self._generate_text_once(
                chat_id=chat_id,
                max_chars=max_chars,
                max_tokens=max_tokens,
                seed_tokens=seed_tokens,
                context_tokens=context_tokens,
                context_bias=context_bias,
                context_start_bias=context_start_bias,
                context_start_affinity=context_start_affinity,
                randomness_strength=attempt_randomness_strength,
                repetition_penalty_strength=repetition_penalty_strength,
                markov_order=markov_order,
                enable_backoff=enable_backoff,
                fuzzy_context_casefold=fuzzy_context_casefold,
                jump_probability=jump_probability,
                context_jump_boost=context_jump_boost,
                order_mix_probability=order_mix_probability,
                context_anchor_splice_probability=context_anchor_splice_probability,
                entropy_sampling=entropy_sampling,
                temporal_blend=temporal_blend,
                interpolation=interpolation,
                now=now,
                rng=generation_rng,
                emit_start=True,
            )
            attempts.append(attempt)
            if attempt.text:
                break

        # Per-attempt counters are summed over all attempts; the final
        # attempt supplies the outcome fields (rejected attempts always carry
        # an empty text, token_count 0 and a rejection reason).
        last = attempts[-1]
        trace = GenerationTrace(
            attempts_used=len(attempts),
            markov_order_used=last.markov_order_used,
            jump_count=sum(a.jump_count for a in attempts),
            rejection_reason=last.rejection_reason,
            token_count=last.token_count,
            start_source=last.start_source,
            leading_punctuation_stripped=sum(
                a.leading_punctuation_stripped for a in attempts
            ),
            context_exact_matches=sum(a.context_exact_matches for a in attempts),
            context_casefold_matches=sum(
                a.context_casefold_matches for a in attempts
            ),
            hidden_context_fallbacks=sum(
                a.hidden_context_fallbacks for a in attempts
            ),
            # M2R-010: the winning (last) attempt's step-pool diagnostics; a
            # rejected final attempt still carries its own measured steps.
            mean_entropy_bits=last.mean_entropy_bits,
            mean_normalized_entropy=last.mean_normalized_entropy,
            mean_branching=last.mean_branching,
            min_confidence=last.min_confidence,
            diagnostic_steps=last.diagnostic_steps,
            mean_applied_temperature=last.mean_applied_temperature,
            applied_alpha=last.applied_alpha,
            blend_step_coverage=last.blend_step_coverage,
            mean_blend_displacement=last.mean_blend_displacement,
            interp_step_coverage=last.interp_step_coverage,
            mean_interp_displacement=last.mean_interp_displacement,
        )
        self.telemetry.note_generation(
            entropy_bits_sum=last.mean_entropy_bits * last.diagnostic_steps,
            normalized_entropy_sum=(
                last.mean_normalized_entropy * last.diagnostic_steps
            ),
            branching_sum=last.mean_branching * last.diagnostic_steps,
            applied_temperature_sum=(
                last.mean_applied_temperature * last.diagnostic_steps
            ),
            blend_covered_steps=round(
                last.blend_step_coverage * last.diagnostic_steps
            ),
            blend_displacement_sum=(
                last.mean_blend_displacement * last.diagnostic_steps
            ),
            interp_covered_steps=round(
                last.interp_step_coverage * last.diagnostic_steps
            ),
            interp_displacement_sum=(
                last.mean_interp_displacement * last.diagnostic_steps
            ),
            steps=last.diagnostic_steps,
        )
        self._log_trace(trace)
        return last.text, trace

    def _log_trace(self, trace: GenerationTrace) -> None:
        logger.debug(
            "Generation trace: attempts=%s order=%s jumps=%s rejection=%s tokens=%s "
            "start_source=%s leading_punctuation_stripped=%s context_exact=%s "
            "context_casefold=%s hidden_context_fallbacks=%s "
            "entropy=%.3f norm_entropy=%.3f branching=%.1f confidence_min=%.3f "
            "temperature=%.2f diag_steps=%s",
            trace.attempts_used,
            trace.markov_order_used,
            trace.jump_count,
            trace.rejection_reason,
            trace.token_count,
            trace.start_source,
            trace.leading_punctuation_stripped,
            trace.context_exact_matches,
            trace.context_casefold_matches,
            trace.hidden_context_fallbacks,
            trace.mean_entropy_bits,
            trace.mean_normalized_entropy,
            trace.mean_branching,
            trace.min_confidence,
            trace.mean_applied_temperature,
            trace.diagnostic_steps,
        )

    async def _pick_seed_start(
        self,
        chat_id: int,
        seed_tokens: list[str] | None,
        *,
        default_order: int,
        next_explore: float,
        next_power: float,
        context_token_set: set[str],
        context_pairs: set[tuple[str, ...]],
        context_triplets: set[tuple[str, ...]],
        context_bias: float,
        repetition_penalty_strength: float,
        rng: random.Random,
    ) -> tuple[tuple[str, str, str], int] | None:
        """Resolve a start triplet from explicit ``seed_tokens``.

        Returns ``(start3, order_used)`` when the seed anchors a start (a stored
        3-gram start, or a stored 2-gram start extended by one weighted step),
        or ``None`` when the seed yields nothing. Callers treat a non-``None``
        result as ``start_source = "seed"``.
        """
        if seed_tokens and len(seed_tokens) >= 3:
            seeded3 = await self.db.get_start3_if_exists(
                chat_id, seed_tokens[0], seed_tokens[1], seed_tokens[2]
            )
            if seeded3:
                return (seeded3[0], seeded3[1], seeded3[2]), default_order
        if seed_tokens and len(seed_tokens) >= 2:
            seeded2 = await self.db.get_start_if_exists(
                chat_id, seed_tokens[0], seed_tokens[1]
            )
            if seeded2:
                w1, w2 = seeded2[0], seeded2[1]
                variants = await self._get2(chat_id, w1, w2)
                if variants:
                    w3 = weighted_next_choice(
                        variants,
                        next_explore,
                        next_power,
                        rng,
                        context_token_set=context_token_set,
                        context_pairs=context_pairs,
                        context_triplets=context_triplets,
                        current_state=(w1, w2),
                        context_bias=context_bias,
                        step_index=0,
                        repetition_penalty_strength=repetition_penalty_strength,
                    )
                    return (w1, w2, w3), 2
        return None

    async def rank_seeds(
        self,
        chat_id: int,
        tokens: list[str],
        *,
        min_support: float,
        branch_min: float,
        branch_ideal: float,
        branch_max: float,
        min_token_len: int,
    ) -> list[SeedScore]:
        """Score a message's tokens as seed anchors, best first (M2R-410).

        Gathers the seed-score inputs (df, support, forward/reverse branching)
        from the M2R-400 read API — once per reply, only when the caller has a
        non-zero seeded ratio — and hands them to the pure scorer. Support and
        forward branching both come from the one ``get_seed_forward`` read.
        """
        eligible = [
            token
            for token in dict.fromkeys(tokens)
            if is_scorable(token, min_token_len=min_token_len)
        ]
        if not eligible:
            self.telemetry.note_seed_ranking(no_corpus=False)
            return []
        n_docs = await self.db.get_n_docs(chat_id)
        # M3R-141: an empty df aggregate silently disables the whole seeded
        # channel — from the outside it looks exactly like a channel whose knob
        # is off. Counted here rather than at the caller because this is the
        # only place that knows *which* emptiness happened.
        self.telemetry.note_seed_ranking(no_corpus=n_docs <= 0)
        if n_docs <= 0:
            return []
        df_of: dict[str, int] = {}
        support_of: dict[str, int] = {}
        forward_branch_of: dict[str, int] = {}
        reverse_branch_of: dict[str, int] = {}
        for token in eligible:
            forward = await self.db.get_seed_forward(chat_id, token)
            forward_branch_of[token] = len(forward)
            support_of[token] = sum(cnt for _, cnt in forward)
            reverse_branch_of[token] = await self.db.get_reverse_branch(
                chat_id, token
            )
            df_of[token] = await self.db.get_token_df(chat_id, token)
        return score_seeds(
            eligible,
            df_of=df_of,
            support_of=support_of,
            forward_branch_of=forward_branch_of,
            reverse_branch_of=reverse_branch_of,
            n_docs=n_docs,
            min_support=min_support,
            branch_min=branch_min,
            branch_ideal=branch_ideal,
            branch_max=branch_max,
            min_token_len=min_token_len,
        )

    def _seeded_step(
        self,
        pool: list[TransitionRow],
        *,
        next_explore: float,
        next_power: float,
        repetition_penalty_strength: float,
        entropy_sampling: EntropySampling,
        temporal_blend: TemporalBlend,
        now: int,
        recent: list[str],
        rng: random.Random,
    ) -> str:
        """One sampling step of the seeded assembler (M2R-410).

        Head and tail of a seeded candidate share this one stepper so their
        sampling cannot drift: the same blend → step-power → weighted-choice
        the forward walk uses, minus the walk's jump/anchor/order-3 machinery
        (a seeded candidate is order-2 both ways, and the reverse index is
        order-2 only). No context bias — the candidate is anchored by its seed,
        not by the message.
        """
        blended = temporal_blend.blend(pool, now)
        step_power = _step_power(None, pool, next_power, entropy_sampling, blended)
        return weighted_next_choice(
            pool,
            next_explore,
            step_power,
            rng,
            recent_tokens=recent,
            repetition_penalty_strength=repetition_penalty_strength,
            base_weights=None if blended is None else blended.weights,
        )

    async def generate_seeded_candidate(
        self,
        chat_id: int,
        seed: str,
        *,
        max_tokens: int,
        head_share: float,
        next_explore: float,
        next_power: float,
        repetition_penalty_strength: float,
        entropy_sampling: EntropySampling = EntropySampling(),
        temporal_blend: TemporalBlend = TemporalBlend(),
        now: int = 0,
        rng: random.Random,
    ) -> list[str] | None:
        """Assemble one candidate anchored on ``seed`` (M2R-410, TZ §9.5).

        The tail grows forward on the chain and the head grows backward on the
        reverse order-2 index, so the anchor sits mid-reply rather than at the
        start. Returns the token list or ``None`` when the seed has no forward
        continuation to bootstrap from (the transparent fallback — the caller
        simply gets one fewer seeded candidate).

        The reverse pool is read fresh, not cached: seeded candidates are a
        minority of the pool, the read is index-served, and a reverse cache
        would have to mirror the incremental-learning fold for no measured
        gain.
        """
        budget = max(2, max_tokens)
        forward = await self.db.get_seed_forward(chat_id, seed)
        if not forward:
            return None
        # Bootstrap the tail's order-2 pair: pick the seed's second token by
        # long count (one draw; the blend governs the per-step walk, not this).
        weights = [max(cnt, 1) ** next_power for _, cnt in forward]
        second = forward[weighted_index_choice(weights, exploring=False, rng=rng)][0]

        head_budget = max(0, min(budget - 2, int(budget * head_share)))
        tail_budget = budget - head_budget

        tail = [seed, second]
        a, b = seed, second
        while len(tail) < tail_budget:
            pool = await self._get2(chat_id, a, b)
            if not pool:
                break
            nxt = self._seeded_step(
                pool,
                next_explore=next_explore,
                next_power=next_power,
                repetition_penalty_strength=repetition_penalty_strength,
                entropy_sampling=entropy_sampling,
                temporal_blend=temporal_blend,
                now=now,
                recent=tail[-10:],
                rng=rng,
            )
            tail.append(nxt)
            if has_degraded_recent_window(tail):
                tail.pop()
                break
            a, b = b, nxt

        head: list[str] = []
        left0, left1 = seed, second
        while len(head) < head_budget:
            pool = await self.db.get_reverse_transitions(chat_id, left0, left1)
            if not pool:
                break
            prev = self._seeded_step(
                pool,
                next_explore=next_explore,
                next_power=next_power,
                repetition_penalty_strength=repetition_penalty_strength,
                entropy_sampling=entropy_sampling,
                temporal_blend=temporal_blend,
                now=now,
                recent=head[-10:],
                rng=rng,
            )
            head.append(prev)
            if has_degraded_recent_window([*reversed(head), seed]):
                head.pop()
                break
            left0, left1 = prev, left0
        head.reverse()
        return head + tail

    async def _pick_global_start(
        self,
        chat_id: int,
        starts3: list[tuple[str, str, str, int]],
        starts2: list[tuple[str, str, int]],
        *,
        default_order: int,
        start_explore: float,
        start_power: float,
        next_explore: float,
        next_power: float,
        context_token_set: set[str],
        context_pairs: set[tuple[str, ...]],
        context_triplets: set[tuple[str, ...]],
        context_bias: float,
        repetition_penalty_strength: float,
        context_stems: frozenset[str] = frozenset(),
        context_start_affinity: float = 1.0,
        rng: random.Random,
    ) -> tuple[tuple[str, str, str], int] | None:
        """Resolve a start triplet from the global start tables.

        Prefers a stored 3-gram start; otherwise picks a 2-gram start and
        extends it by one weighted step. Returns ``(start3, order_used)``, or
        ``None`` when a chosen 2-gram start has no continuation — the caller
        rejects that with ``no_start_transition``. Callers invoke this only
        when at least one of ``starts3``/``starts2`` is non-empty.
        """
        if starts3:
            return (
                weighted_start3_choice(
                    starts3,
                    start_explore,
                    start_power,
                    rng,
                    context_stems=context_stems,
                    context_start_affinity=context_start_affinity,
                ),
                default_order,
            )
        if starts2:
            w1, w2 = weighted_start2_choice(starts2, start_explore, start_power, rng)
            variants = await self._get2(chat_id, w1, w2)
            if not variants:
                return None
            w3 = weighted_next_choice(
                variants,
                next_explore,
                next_power,
                rng,
                context_token_set=context_token_set,
                context_pairs=context_pairs,
                context_triplets=context_triplets,
                current_state=(w1, w2),
                context_bias=context_bias,
                step_index=0,
                repetition_penalty_strength=repetition_penalty_strength,
            )
            return (w1, w2, w3), 2
        return None

    async def _pick_contextual_start(
        self,
        chat_id: int,
        context_tokens: list[str],
        use_contextual_start: bool,
        *,
        start_explore: float,
        start_power: float,
        next_explore: float,
        next_power: float,
        context_token_set: set[str],
        context_pairs: set[tuple[str, ...]],
        context_triplets: set[tuple[str, ...]],
        context_bias: float,
        repetition_penalty_strength: float,
        fuzzy_context_casefold: bool,
        rng: random.Random,
    ) -> _ContextualStateSelection | None:
        """Select a hidden start state anchored on the reply context.

        Returns ``None`` when no contextual start is attempted (no context or
        the probabilistic roll declined) or when no context state matched —
        the caller distinguishes the two to count ``hidden_context_fallbacks``.
        """
        if not (context_tokens and use_contextual_start):
            return None
        return await self._select_contextual_state(
            chat_id=chat_id,
            context_tokens=context_tokens,
            start_explore=start_explore,
            start_power=start_power,
            next_explore=next_explore,
            next_power=next_power,
            context_token_set=context_token_set,
            context_pairs=context_pairs,
            context_triplets=context_triplets,
            context_bias=context_bias,
            repetition_penalty_strength=repetition_penalty_strength,
            fuzzy_context_casefold=fuzzy_context_casefold,
            rng=rng,
        )

    @staticmethod
    def _contextual_match_counts(
        selection: _ContextualStateSelection,
    ) -> tuple[int, int]:
        """Map a contextual match kind to its ``(exact, casefold)`` trace
        counters."""
        if selection.match_kind == "exact":
            return 1, 0
        return 0, 1

    def _finalize_attempt(
        self,
        generated: list[str],
        *,
        max_chars: int,
        context_tokens: list[str],
        order_used: int,
        jump_count: int,
        start_source: str,
        context_exact_matches: int,
        context_casefold_matches: int,
        hidden_context_fallbacks: int,
        diagnostics: _DiagnosticsAccumulator | None = None,
        applied_alpha: float = 0.0,
    ) -> _GenerationAttempt:
        """Attach trace metadata to a candidate run through the shared pipeline.

        The tail pipeline and the four form gates live in
        ``finalize_candidate_tokens`` — this wrapper only turns their outcome
        into a ``_GenerationAttempt``, so the main walk and every other branch
        that assembles a candidate share one definition of "finished".
        """
        final = finalize_candidate_tokens(
            generated, max_chars=max_chars, context_tokens=context_tokens
        )
        diag = diagnostics or _DiagnosticsAccumulator()
        steps = diag.steps
        rejected = final.rejection_reason is not None
        return _GenerationAttempt(
            text=final.text,
            markov_order_used=order_used,
            jump_count=jump_count,
            rejection_reason=final.rejection_reason,
            token_count=0 if rejected else len(final.tokens),
            start_source=start_source,
            leading_punctuation_stripped=final.leading_punctuation_stripped,
            context_exact_matches=context_exact_matches,
            context_casefold_matches=context_casefold_matches,
            hidden_context_fallbacks=hidden_context_fallbacks,
            mean_entropy_bits=diag.entropy_bits_sum / steps if steps else 0.0,
            mean_normalized_entropy=(
                diag.normalized_entropy_sum / steps if steps else 0.0
            ),
            mean_branching=diag.branching_sum / steps if steps else 0.0,
            min_confidence=diag.min_confidence,
            diagnostic_steps=steps,
            mean_applied_temperature=(
                diag.applied_temperature_sum / steps if steps else 0.0
            ),
            applied_alpha=applied_alpha,
            blend_step_coverage=(diag.blend_covered_steps / steps if steps else 0.0),
            mean_blend_displacement=(
                diag.blend_displacement_sum / steps if steps else 0.0
            ),
            interp_step_coverage=(
                diag.interp_covered_steps / steps if steps else 0.0
            ),
            mean_interp_displacement=(
                diag.interp_displacement_sum / steps if steps else 0.0
            ),
        )

    async def _run_generation_loop(
        self,
        chat_id: int,
        start3: tuple[str, str, str],
        *,
        emit_tokens: list[str],
        order: int,
        order_used: int,
        max_tokens: int,
        max_chars: int,
        enable_backoff: bool,
        starts3: list[tuple[str, str, str, int]],
        context_token_set: set[str],
        context_pairs: set[tuple[str, ...]],
        context_triplets: set[tuple[str, ...]],
        start_explore: float,
        start_power: float,
        next_explore: float,
        next_power: float,
        context_bias: float,
        repetition_penalty_strength: float,
        jump_probability: float,
        order_mix_probability: float,
        anchor_state: tuple[str, str, str] | None = None,
        anchor_emit_tokens: list[str] | None = None,
        anchor_target_tokens: int = 0,
        rng: random.Random,
        diagnostics: _DiagnosticsAccumulator | None = None,
        entropy_sampling: EntropySampling = EntropySampling(),
        temporal_blend: TemporalBlend = TemporalBlend(),
        interpolation: OrderInterpolation = OrderInterpolation(),
        now: int = 0,
    ) -> tuple[list[str], int, int, bool]:
        """Walk the Markov chain from ``start3`` until a stop condition.

        Returns ``(generated, order_used, jump_count, anchor_spliced)``.
        ``emit_tokens`` is the prefix actually written into the output (the
        full start for a global or seed start, a trimmed tail or nothing for a
        contextual start) while the chain always walks from the full ``start3``
        state. Per-step transitions fall back from order 3 to 2 when
        ``enable_backoff`` allows; dedup state suppresses immediate triplet
        repeats. Only local state is mutated.

        A non-``None`` ``anchor_state`` is a deferred contextual anchor: once
        the reply reaches ``anchor_target_tokens`` tokens (or the walk
        dead-ends), ``anchor_emit_tokens`` are spliced in behind a connective
        and the chain continues from the anchor state. A pending anchor
        suppresses global jumps and its splice counts as the reply's jump, so
        both channels share the one-aside-per-reply budget.
        """
        token_limit = max(1, max_tokens)
        w1, w2, w3 = start3
        start_tokens = [w1, w2, w3]
        generated: list[str] = emit_tokens[:token_limit]
        dedup_seed = start_tokens
        visited_triplets: OrderedDict[tuple[str, str, str], None] = (
            OrderedDict.fromkeys([(w1, w2, w3)])
        )
        seen_pairs: OrderedDict[tuple[str, ...], None] = OrderedDict.fromkeys(
            build_windows(dedup_seed, 2)
        )
        seen_triplets: OrderedDict[tuple[str, ...], None] = OrderedDict.fromkeys(
            build_windows(dedup_seed, 3)
        )
        jump_count = 0
        last_jump_end = 0
        used_connectives: list[tuple[str, ...]] = []
        anchor_pending = anchor_state is not None

        def splice_anchor() -> None:
            """Voice the deferred anchor: connective + emission tokens, then
            continue the chain from the anchor state. Consumes the jump slot.

            Always a wordy connective, never the silent «.» splice: the
            finalize pass trims the reply to the last sentence end, so a
            silent marker whose tail earns no own terminal punctuation would
            cut the anchor right back out while the trace still claims it."""
            nonlocal w1, w2, w3, jump_count, last_jump_end, anchor_pending
            assert anchor_state is not None
            emit = anchor_emit_tokens or []
            trim_splice_tail(generated)
            connective = pick_jump_connective(
                rng,
                exclude=used_connectives
                + [
                    phrase
                    for phrase in JUMP_CONNECTIVE_TOKENS
                    if emit and emit[0].casefold() in phrase
                ],
            )
            used_connectives.append(connective)
            generated.extend(splice_marker_tokens(generated, connective))
            generated.extend(emit)
            last_jump_end = len(generated)
            w1, w2, w3 = anchor_state
            remember_bounded(visited_triplets, (w1, w2, w3), 40)
            if len(generated) >= 2:
                remember_bounded(seen_pairs, (generated[-2], generated[-1]), 80)
            if len(generated) >= 3:
                remember_bounded(
                    seen_triplets,
                    (generated[-3], generated[-2], generated[-1]),
                    80,
                )
            jump_count += 1
            anchor_pending = False

        for step_index in range(self.max_steps):
            if len(generated) >= token_limit:
                break
            if anchor_pending and len(generated) >= anchor_target_tokens:
                splice_anchor()
                continue
            if (
                not anchor_pending
                and len(generated) >= JUMP_MIN_GENERATED_TOKENS
                and jump_count < JUMP_MAX_PER_REPLY
                and (
                    jump_count == 0
                    or len(generated) - last_jump_end >= JUMP_MIN_TOKENS_BETWEEN
                )
                and rng.random() < jump_probability
                and starts3
                and order >= 3
            ):
                # Jumps drift to a *global* learned start on purpose: the reply
                # context is already anchored by the (emitted) contextual start
                # and the step bias, and a contextual jump target used to splice
                # the same context n-gram back in several times per reply,
                # which read as parroting the question.
                nw1, nw2, nw3 = weighted_start3_choice(
                    starts3,
                    start_explore,
                    start_power,
                    rng,
                )
                # Splice a connective and the new sentence-start triplet into the
                # output so the topic drift is voiced. The new triplet is a real
                # learned start, so "<text>, кстати <new sentence>" reads as an
                # aside instead of silently dropping into mid-chain (which is why
                # the jump was disabled before M4). Overshooting token_limit is
                # fine — the top-of-loop guard and the finalize trims handle it.
                trim_splice_tail(generated)
                # Exclude connectives containing the jump target's first word
                # anywhere, not just as the last token: ", ну и" + "ну ..."
                # stuttered into "ну и ну" when only phrase[-1] was checked.
                connective = pick_splice_connective(
                    rng,
                    exclude=used_connectives
                    + [
                        phrase
                        for phrase in JUMP_CONNECTIVE_TOKENS
                        if nw1 in phrase
                    ],
                )
                used_connectives.append(connective)
                generated.extend(splice_marker_tokens(generated, connective))
                generated.extend((nw1, nw2, nw3))
                last_jump_end = len(generated)
                w1, w2, w3 = nw1, nw2, nw3
                remember_bounded(visited_triplets, (w1, w2, w3), 40)
                remember_bounded(seen_pairs, (generated[-2], generated[-1]), 80)
                remember_bounded(
                    seen_triplets, (generated[-3], generated[-2], generated[-1]), 80
                )
                jump_count += 1
                continue

            recent_window = generated[-10:]
            pool3 = await self._get3(chat_id, w1, w2, w3) if order >= 3 else []
            # Order-mix branching valve: with probability order_mix_probability
            # take this step from the order-2 pool even though order-3 has a
            # continuation — ~98% of order-3 states carry exactly one option,
            # so the order-3 walk replays its source message. Divert only when
            # order-2 genuinely widens the choice; emptying pool3 reuses the
            # regular backoff path below (hence the enable_backoff guard).
            if (
                pool3
                and enable_backoff
                and order_mix_probability > 0.0
                and rng.random() < order_mix_probability
            ):
                mix_pool2 = await self._get2(chat_id, w2, w3)
                if len(mix_pool2) > len(pool3):
                    pool3 = []
            if pool3 and order >= 3:
                candidates = [
                    row for row in pool3 if (w2, w3, row[0]) not in visited_triplets
                ]
                pool = candidates or pool3
                # M2R-010: diagnostics of the pool actually sampled from, after
                # dedup filtering and the order-mix decision. M2R-100 samples
                # from the same number: the step's power is the entropy-adjusted
                # one, and the adjustment happens here — before
                # ``weighted_next_choice`` rolls exploration on top of it.
                # M2R-210: the blend runs first, so entropy and temperature both
                # describe the distribution that is really sampled.
                blended = temporal_blend.blend(pool, now)
                if diagnostics is not None:
                    diagnostics.note_blend(blended)
                # M2R-900: soft interpolation with the order-2 projection of this
                # state. Guarded by ``enabled`` so a neutral beta never reads the
                # projection at all — the byte-identity invariant is about the
                # read as much as about the arithmetic.
                merged: InterpolatedPool | None = None
                if interpolation.enabled:
                    proj2 = await self._get2(chat_id, w2, w3)
                    blended_proj = temporal_blend.blend(proj2, now)
                    merged = interpolation.merge(
                        pool,
                        proj2,
                        base3=None if blended is None else blended.weights,
                        base2=None if blended_proj is None else blended_proj.weights,
                    )
                    if diagnostics is not None:
                        diagnostics.note_interp(merged)
                if merged is not None:
                    # The merged distribution rides the same channel the blend
                    # already uses: pool plus normalized weights, so the sampler
                    # keeps its 1e-12 floor instead of the count-calibrated 0.01.
                    pool = merged.rows
                    blended = BlendedPool(merged.weights, 0.0)
                step_power = _step_power(
                    diagnostics, pool, next_power, entropy_sampling, blended
                )
                w4 = weighted_next_choice(
                    pool,
                    next_explore,
                    step_power,
                    rng,
                    context_token_set=context_token_set,
                    context_pairs=context_pairs,
                    context_triplets=context_triplets,
                    current_state=(w2, w3),
                    context_bias=context_bias,
                    step_index=step_index,
                    recent_tokens=recent_window,
                    seen_pairs=seen_pairs,
                    seen_triplets=seen_triplets,
                    repetition_penalty_strength=repetition_penalty_strength,
                    base_weights=None if blended is None else blended.weights,
                )
            else:
                if not enable_backoff and order >= 3:
                    if anchor_pending:
                        splice_anchor()
                        continue
                    break

                pool2 = await self._get2(chat_id, w2, w3)
                if pool2:
                    blended2 = temporal_blend.blend(pool2, now)
                    if diagnostics is not None:
                        diagnostics.note_blend(blended2)
                    step_power = _step_power(
                        diagnostics, pool2, next_power, entropy_sampling, blended2
                    )
                    w4 = weighted_next_choice(
                        pool2,
                        next_explore,
                        step_power,
                        rng,
                        base_weights=None if blended2 is None else blended2.weights,
                        context_token_set=context_token_set,
                        context_pairs=context_pairs,
                        context_triplets=context_triplets,
                        current_state=(w2, w3),
                        context_bias=context_bias,
                        step_index=step_index,
                        recent_tokens=recent_window,
                        seen_pairs=seen_pairs,
                        seen_triplets=seen_triplets,
                        repetition_penalty_strength=repetition_penalty_strength,
                    )
                    order_used = min(order_used, 2)
                else:
                    if anchor_pending:
                        splice_anchor()
                        continue
                    break

            generated.append(w4)
            if has_degraded_recent_window(generated):
                break
            maybe_text = detokenize(generated, max_chars=max_chars)
            if len(generated) >= token_limit or len(maybe_text) >= max_chars:
                break

            w1, w2, w3 = w2, w3, w4
            remember_bounded(visited_triplets, (w1, w2, w3), 40)
            if len(generated) >= 2:
                remember_bounded(
                    seen_pairs,
                    (generated[-2], generated[-1]),
                    80,
                )
            if len(generated) >= 3:
                remember_bounded(
                    seen_triplets,
                    (generated[-3], generated[-2], generated[-1]),
                    80,
                )

        anchor_spliced = anchor_state is not None and not anchor_pending
        return generated, order_used, jump_count, anchor_spliced

    async def _generate_text_once(
        self,
        chat_id: int,
        max_chars: int,
        max_tokens: int = 45,
        seed_tokens: list[str] | None = None,
        context_tokens: list[str] | None = None,
        context_bias: float = 1.0,
        context_start_bias: float = 1.0,
        context_start_affinity: float = 1.0,
        randomness_strength: float = 1.0,
        repetition_penalty_strength: float = 1.0,
        markov_order: int = 3,
        enable_backoff: bool = True,
        fuzzy_context_casefold: bool = False,
        jump_probability: float = 0.0,
        context_jump_boost: float = 1.0,
        order_mix_probability: float = 0.0,
        context_anchor_splice_probability: float = 0.0,
        entropy_sampling: EntropySampling = EntropySampling(),
        temporal_blend: TemporalBlend = TemporalBlend(),
        interpolation: OrderInterpolation = OrderInterpolation(),
        now: int = 0,
        rng: random.Random | None = None,
        emit_start: bool = True,
    ) -> _GenerationAttempt:
        generation_rng = rng or random.Random()
        order = 3 if markov_order >= 3 else 2
        order_used = order
        strength = max(0.0, min(3.0, randomness_strength))
        next_explore = min(0.98, 0.12 + 0.18 * strength)
        next_power = max(0.15, 0.72 - 0.16 * strength)
        start_explore = min(0.98, 0.20 + 0.20 * strength)
        start_power = max(0.15, 0.75 - 0.18 * strength)

        context_tokens = context_tokens or []
        context_token_set = set(context_tokens)
        context_pairs = set(build_windows(context_tokens, 2))
        context_triplets = set(build_windows(context_tokens, 3))
        context_stems = (
            context_start_stems(context_tokens)
            if context_tokens and context_start_affinity > 1.0
            else frozenset()
        )

        starts3 = await self._get_starts3(chat_id) if order >= 3 else []
        starts2 = await self._get_starts2(chat_id)
        if not starts3 and not starts2:
            return _GenerationAttempt(
                text="",
                markov_order_used=0,
                jump_count=0,
                rejection_reason=REJECTION_NO_STARTS,
                token_count=0,
                start_source="global",
            )

        start3: tuple[str, str, str] | None = None
        start_source = "global"
        contextual_emit_tokens: list[str] = []
        deferred_anchor: _ContextualStateSelection | None = None
        deferred_anchor_emit: list[str] = []
        context_exact_matches = 0
        context_casefold_matches = 0
        hidden_context_fallbacks = 0
        seed_start = await self._pick_seed_start(
            chat_id,
            seed_tokens,
            default_order=order_used,
            next_explore=next_explore,
            next_power=next_power,
            context_token_set=context_token_set,
            context_pairs=context_pairs,
            context_triplets=context_triplets,
            context_bias=context_bias,
            repetition_penalty_strength=repetition_penalty_strength,
            rng=generation_rng,
        )
        if seed_start is not None:
            start3, order_used = seed_start
            start_source = "seed"

        use_contextual_start = generation_rng.random() < context_start_probability(
            context_start_bias
        )
        if start3 is None:
            contextual_state = await self._pick_contextual_start(
                chat_id,
                context_tokens,
                use_contextual_start,
                start_explore=start_explore,
                start_power=start_power,
                next_explore=next_explore,
                next_power=next_power,
                context_token_set=context_token_set,
                context_pairs=context_pairs,
                context_triplets=context_triplets,
                context_bias=context_bias,
                repetition_penalty_strength=repetition_penalty_strength,
                fuzzy_context_casefold=fuzzy_context_casefold,
                rng=generation_rng,
            )
            if contextual_state is not None:
                emission = context_emission_tokens(contextual_state.state)
                # The probability is checked before the roll so a disabled
                # knob consumes no RNG draw and the sequence (and the eval
                # baselines) stays byte-identical to the pre-knob pipeline.
                if (
                    emission
                    and context_anchor_splice_probability > 0.0
                    and generation_rng.random()
                    < context_anchor_splice_probability
                ):
                    # Anchor segmentation: defer the (visible) anchor — the
                    # walk starts globally and the anchor is spliced in later
                    # by the generation loop, so the context surfaces mid- or
                    # end-reply. Hidden anchors (empty emission) are never
                    # deferred: a silent splice would drop into mid-chain,
                    # which is exactly what voiced jumps exist to avoid.
                    deferred_anchor = contextual_state
                    deferred_anchor_emit = emission
                else:
                    start3 = contextual_state.state
                    order_used = contextual_state.order
                    # Visible contextual start: emit the trimmed tail of the
                    # matched window so the reply picks the context up out loud
                    # instead of only continuing after it. An empty tail (all
                    # stopwords) keeps the start hidden.
                    contextual_emit_tokens = emission
                    emit_start = False
                    start_source = (
                        "context" if contextual_emit_tokens else "hidden_context"
                    )
                (
                    context_exact_matches,
                    context_casefold_matches,
                ) = self._contextual_match_counts(contextual_state)
            elif context_tokens and use_contextual_start:
                hidden_context_fallbacks = 1

        if start3 is None:
            global_start = await self._pick_global_start(
                chat_id,
                starts3,
                starts2,
                default_order=order_used,
                start_explore=start_explore,
                start_power=start_power,
                next_explore=next_explore,
                next_power=next_power,
                context_token_set=context_token_set,
                context_pairs=context_pairs,
                context_triplets=context_triplets,
                context_bias=context_bias,
                repetition_penalty_strength=repetition_penalty_strength,
                context_stems=context_stems,
                context_start_affinity=context_start_affinity,
                rng=generation_rng,
            )
            if global_start is None:
                return _GenerationAttempt(
                    text="",
                    markov_order_used=2,
                    jump_count=0,
                    rejection_reason=REJECTION_NO_START_TRANSITION,
                    token_count=0,
                    start_source=start_source,
                    leading_punctuation_stripped=0,
                    context_exact_matches=context_exact_matches,
                    context_casefold_matches=context_casefold_matches,
                    hidden_context_fallbacks=hidden_context_fallbacks,
                )
            start3, order_used = global_start

        emit_tokens = list(start3) if emit_start else contextual_emit_tokens
        # Context+chaos: a contextual anchor's continuation is a corpus
        # retrace, so only those walks get the boosted drift probability.
        # Deferred-anchor walks keep the base value: their jump budget is
        # reserved for the anchor splice anyway.
        effective_jump_probability = jump_probability
        if start_source in ("context", "hidden_context"):
            effective_jump_probability = min(
                1.0, jump_probability * max(1.0, context_jump_boost)
            )
        # The anchor lands at a uniformly rolled token position, so its place
        # in the reply varies from just-past-the-start to the tail; the walk
        # dead-ending earlier moves the splice to the dead end instead.
        anchor_target_tokens = 0
        if deferred_anchor is not None:
            latest = max(
                JUMP_MIN_GENERATED_TOKENS,
                max(1, max_tokens) - len(deferred_anchor_emit) - 3,
            )
            anchor_target_tokens = generation_rng.randint(
                JUMP_MIN_GENERATED_TOKENS, latest
            )
        diagnostics = _DiagnosticsAccumulator()
        generated, order_used, jump_count, anchor_spliced = (
            await self._run_generation_loop(
                chat_id,
                start3,
                emit_tokens=emit_tokens,
                order=order,
                order_used=order_used,
                max_tokens=max_tokens,
                max_chars=max_chars,
                enable_backoff=enable_backoff,
                starts3=starts3,
                context_token_set=context_token_set,
                context_pairs=context_pairs,
                context_triplets=context_triplets,
                start_explore=start_explore,
                start_power=start_power,
                next_explore=next_explore,
                next_power=next_power,
                context_bias=context_bias,
                repetition_penalty_strength=repetition_penalty_strength,
                jump_probability=effective_jump_probability,
                order_mix_probability=order_mix_probability,
                anchor_state=(
                    deferred_anchor.state if deferred_anchor is not None else None
                ),
                anchor_emit_tokens=deferred_anchor_emit,
                anchor_target_tokens=anchor_target_tokens,
                rng=generation_rng,
                diagnostics=diagnostics,
                entropy_sampling=entropy_sampling,
                temporal_blend=temporal_blend,
                interpolation=interpolation,
                now=now,
            )
        )
        if deferred_anchor is not None:
            # An unspliced anchor (walk hit a limit first) leaves a plain
            # global reply — the trace must not claim context anchoring.
            start_source = "context_spliced" if anchor_spliced else "global"
            if not anchor_spliced:
                context_exact_matches = 0
                context_casefold_matches = 0

        return self._finalize_attempt(
            generated,
            max_chars=max_chars,
            context_tokens=context_tokens,
            order_used=order_used,
            jump_count=jump_count,
            start_source=start_source,
            context_exact_matches=context_exact_matches,
            context_casefold_matches=context_casefold_matches,
            hidden_context_fallbacks=hidden_context_fallbacks,
            diagnostics=diagnostics,
            applied_alpha=temporal_blend.alpha,
        )
