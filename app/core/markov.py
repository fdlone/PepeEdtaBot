from __future__ import annotations

import logging
import random
import re
from collections import Counter, OrderedDict
from collections.abc import Container, Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

from app.core.context_state_matcher import ContextStateMatcher
from app.core.lexicon import BAD_ENDING_WORDS, STOPWORDS
from app.core.morphology import stem_token
from app.infrastructure.database import Database

logger = logging.getLogger("chat_markov")

TOKEN_RE = re.compile(r"\w+|[.,!?;:]", re.UNICODE)
PUNCT_SET = {".", ",", "!", "?", ";", ":"}
DEFAULT_GENERATION_ATTEMPT_BUDGET = 1
EXPLORATION_FLATTENING = 1.5
EXPLORATION_POWER_FLOOR = 0.02
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
)

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
    context_prefix_matches: int = 0
    context_prefix_singleton_matches: int = 0
    hidden_context_fallbacks: int = 0


class _GenerationAttempt(NamedTuple):
    text: str
    markov_order_used: int
    jump_count: int
    rejection_reason: str | None
    token_count: int
    start_source: str
    leading_punctuation_stripped: int = 0
    context_exact_matches: int = 0
    context_casefold_matches: int = 0
    context_prefix_matches: int = 0
    context_prefix_singleton_matches: int = 0
    hidden_context_fallbacks: int = 0


class _ContextualStateSelection(NamedTuple):
    state: tuple[str, str, str]
    order: int
    match_kind: str
    transition_count: int = 0


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


def extract_best_seed(tokens: list[str], seed_length: int) -> list[str]:
    """Return up to seed_length tokens from the end, skipping trailing punctuation."""
    trimmed = tokens[:]
    while trimmed and trimmed[-1] in PUNCT_SET:
        trimmed.pop()
    if len(trimmed) >= seed_length:
        return trimmed[-seed_length:]
    return tokens[-seed_length:] if len(tokens) >= seed_length else tokens[:]


def longest_shared_run(tokens_a: list[str], tokens_b: list[str]) -> int:
    best = 0
    for idx_a in range(len(tokens_a)):
        for idx_b in range(len(tokens_b)):
            run = 0
            while (
                idx_a + run < len(tokens_a)
                and idx_b + run < len(tokens_b)
                and tokens_a[idx_a + run] == tokens_b[idx_b + run]
            ):
                run += 1
            if run > best:
                best = run
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
    items: list[tuple[str, int]],
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
) -> str:
    ordered_items = sorted(items, key=lambda item: item[0])
    population = [token for token, _ in ordered_items]
    exploring, frequency_power = _roll_exploration(explore_probability, power, rng)
    step_bias = 1.0 + (max(1.0, context_bias) - 1.0) * context_decay(step_index)
    penalty_strength = max(0.0, repetition_penalty_strength)
    recent_tokens = recent_tokens or []
    weights: list[float] = []
    for token, cnt in ordered_items:
        weight = max(cnt, 1) ** frequency_power
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

        weight = max(weight, 0.01)
        weights.append(weight)
    return weighted_population_choice(
        population, weights, exploring=exploring, rng=rng
    )


def weighted_start2_choice(
    items: list[tuple[str, str, int]],
    explore_probability: float,
    power: float,
    rng: random.Random,
) -> tuple[str, str]:
    ordered_items = sorted(items, key=lambda item: (item[0], item[1]))
    population = [(w1, w2) for w1, w2, _ in ordered_items]
    exploring, frequency_power = _roll_exploration(explore_probability, power, rng)
    weights = [max(cnt, 1) ** frequency_power for _, _, cnt in ordered_items]
    return weighted_population_choice(
        population, weights, exploring=exploring, rng=rng
    )


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
    ordered_items = sorted(items, key=lambda item: (item[0], item[1], item[2]))
    population = [(w1, w2, w3) for w1, w2, w3, _ in ordered_items]
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
        for weight, (w1, w2, w3) in zip(weights, population, strict=True):
            state_stems = {
                stem_token(w1.casefold()),
                stem_token(w2.casefold()),
                stem_token(w3.casefold()),
            }
            shared = len(state_stems & context_stems)
            if shared and not state_stems <= context_stems:
                weight *= context_start_affinity**shared
            boosted.append(weight)
        weights = boosted
    return weighted_population_choice(
        population, weights, exploring=exploring, rng=rng
    )


@dataclass(slots=True)
class MarkovGenerator:
    db: Database
    max_steps: int = 90
    cache_limit: int = 1024

    _cache3: OrderedDict[tuple[int, str, str, str], list[tuple[str, int]]] = field(
        default_factory=OrderedDict, init=False
    )
    _cache2: OrderedDict[tuple[int, str, str], list[tuple[str, int]]] = field(
        default_factory=OrderedDict, init=False
    )
    _cache_starts3: OrderedDict[int, list[tuple[str, str, str, int]]] = field(
        default_factory=OrderedDict, init=False
    )
    _cache_starts2: OrderedDict[int, list[tuple[str, str, int]]] = field(
        default_factory=OrderedDict, init=False
    )
    _context_state_matcher: ContextStateMatcher = field(init=False)

    def __post_init__(self) -> None:
        self._context_state_matcher = ContextStateMatcher(
            self.db,
            cache_limit=self.cache_limit,
        )

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
            return self._cache_starts3[chat_id]
        rows = await self.db.get_starts3(chat_id)
        self._touch_cache(self._cache_starts3, chat_id, rows)
        return rows

    async def _get_starts2(self, chat_id: int) -> list[tuple[str, str, int]]:
        if chat_id in self._cache_starts2:
            return self._cache_starts2[chat_id]
        rows = await self.db.get_starts(chat_id)
        self._touch_cache(self._cache_starts2, chat_id, rows)
        return rows

    async def _get3(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> list[tuple[str, int]]:
        key = (chat_id, w1, w2, w3)
        if key in self._cache3:
            self._cache3.move_to_end(key)
            return self._cache3[key]
        rows = await self.db.get_transitions3(chat_id, w1, w2, w3)
        self._touch_cache(self._cache3, key, rows)
        return rows

    async def _get2(self, chat_id: int, w1: str, w2: str) -> list[tuple[str, int]]:
        key = (chat_id, w1, w2)
        if key in self._cache2:
            self._cache2.move_to_end(key)
            return self._cache2[key]
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
                transition_count = sum(count for _, count in transitions)
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
    ) -> list[tuple[tuple[str, str], list[tuple[str, int]], float]]:
        """Exact 2-gram context windows that have stored transitions, carrying
        their transitions for the follow-up step, weighted by count and
        recency."""
        candidates2: list[tuple[tuple[str, str], list[tuple[str, int]], float]] = []
        for index, window in enumerate(windows2):
            transitions = await self._get2(chat_id, window[0], window[1])
            if transitions:
                transition_count = sum(count for _, count in transitions)
                recency_bonus = 1.0 + ((index + 1) / total2) * 0.30
                weight = max(transition_count, 1) ** frequency_power * recency_bonus
                candidates2.append(((window[0], window[1]), transitions, weight))
        return candidates2

    async def _build_fuzzy3_candidates(
        self,
        chat_id: int,
        windows3: list[tuple[str, ...]],
        total3: int,
        frequency_power: float,
        *,
        match_kind: str,
        include_prefix: bool,
    ) -> list[tuple[tuple[str, str, str], float, int]]:
        """Fuzzy 3-gram start candidates of one ``match_kind`` (casefold or
        prefix) via the context-state matcher, weighted by count, recency and —
        for prefix matches — similarity. Each entry keeps its transition count.
        """
        candidates: list[tuple[tuple[str, str, str], float, int]] = []
        for index, window in enumerate(windows3):
            matches = await self._context_state_matcher.match(
                chat_id,
                window,
                3,
                include_prefix=include_prefix,
            )
            recency_bonus = 1.0 + ((index + 1) / total3) * 0.35
            for match in matches:
                if match.match_kind != match_kind:
                    continue
                state3 = (match.state[0], match.state[1], match.state[2])
                similarity = match.similarity if include_prefix else 1.0
                weight = (
                    max(match.transition_count, 1) ** frequency_power
                    * similarity
                    * recency_bonus
                )
                candidates.append((state3, weight, match.transition_count))
        return candidates

    async def _build_fuzzy2_candidates(
        self,
        chat_id: int,
        windows2: list[tuple[str, ...]],
        total2: int,
        frequency_power: float,
        *,
        match_kind: str,
        include_prefix: bool,
    ) -> tuple[
        list[tuple[tuple[str, str], list[tuple[str, int]], float]],
        dict[tuple[str, str], int],
    ]:
        """Fuzzy 2-gram start candidates of one ``match_kind`` that still have
        stored transitions. Returns the weighted candidates plus, per state,
        the best transition count (used to flag prefix singletons)."""
        additions: list[tuple[tuple[str, str], list[tuple[str, int]], float]] = []
        transition_counts: dict[tuple[str, str], int] = {}
        for index, window in enumerate(windows2):
            matches = await self._context_state_matcher.match(
                chat_id,
                window,
                2,
                include_prefix=include_prefix,
            )
            recency_bonus = 1.0 + ((index + 1) / total2) * 0.30
            for match in matches:
                if match.match_kind != match_kind:
                    continue
                state2 = (match.state[0], match.state[1])
                transitions = await self._get2(chat_id, state2[0], state2[1])
                if not transitions:
                    continue
                similarity = match.similarity if include_prefix else 1.0
                weight = (
                    max(match.transition_count, 1) ** frequency_power
                    * similarity
                    * recency_bonus
                )
                additions.append((state2, transitions, weight))
                transition_counts[state2] = max(
                    transition_counts.get(state2, 0),
                    match.transition_count,
                )
        return additions, transition_counts

    @staticmethod
    def _select_state3(
        candidates: list[tuple[tuple[str, str, str], float, int]],
        match_kind: str,
        *,
        exploring: bool,
        rng: random.Random,
    ) -> _ContextualStateSelection:
        """Weighted pick of a 3-gram start state from ``(state, weight, count)``
        candidates. Only prefix matches carry a transition count (the best one
        among duplicate states), consumed by the singleton stats."""
        population = [state for state, _, _ in candidates]
        weights = [weight for _, weight, _ in candidates]
        selected = weighted_population_choice(
            population, weights, exploring=exploring, rng=rng
        )
        selected_count = 0
        if match_kind == "prefix":
            selected_count = max(
                transition_count
                for state, _, transition_count in candidates
                if state == selected
            )
        return _ContextualStateSelection(selected, 3, match_kind, selected_count)

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
        fuzzy_context_prefix: bool,
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
        prefix_transition_counts: dict[tuple[str, str], int] = {}
        if not candidates2 and fuzzy_context_casefold:
            casefold_candidates3 = await self._build_fuzzy3_candidates(
                chat_id, windows3, total3, frequency_power,
                match_kind="casefold", include_prefix=False,
            )
            if casefold_candidates3:
                return self._select_state3(
                    casefold_candidates3, "casefold", exploring=exploring, rng=rng
                )

            additions, _ = await self._build_fuzzy2_candidates(
                chat_id, windows2, total2, frequency_power,
                match_kind="casefold", include_prefix=False,
            )
            candidates2.extend(additions)
            match_kind = "casefold"

        if not candidates2 and fuzzy_context_prefix:
            prefix_candidates3 = await self._build_fuzzy3_candidates(
                chat_id, windows3, total3, frequency_power,
                match_kind="prefix", include_prefix=True,
            )
            if prefix_candidates3:
                return self._select_state3(
                    prefix_candidates3, "prefix", exploring=exploring, rng=rng
                )

            additions, prefix_transition_counts = await self._build_fuzzy2_candidates(
                chat_id, windows2, total2, frequency_power,
                match_kind="prefix", include_prefix=True,
            )
            candidates2.extend(additions)
            match_kind = "prefix"

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
        return _ContextualStateSelection(
            (w1, w2, w3),
            2,
            match_kind,
            prefix_transition_counts.get((w1, w2), 0),
        )

    async def generate_text(
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
        fuzzy_context_prefix: bool = False,
        context_emit_start: bool = False,
        jump_probability: float = 0.0,
        rng: random.Random | None = None,
        attempt_budget: int = DEFAULT_GENERATION_ATTEMPT_BUDGET,
    ) -> str:
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
            fuzzy_context_prefix=fuzzy_context_prefix,
            context_emit_start=context_emit_start,
            jump_probability=jump_probability,
            rng=rng,
            attempt_budget=attempt_budget,
        )
        return text

    async def generate_text_with_trace(
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
        fuzzy_context_prefix: bool = False,
        context_emit_start: bool = False,
        jump_probability: float = 0.0,
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
                fuzzy_context_prefix=fuzzy_context_prefix,
                context_emit_start=context_emit_start,
                jump_probability=jump_probability,
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
            context_prefix_matches=sum(a.context_prefix_matches for a in attempts),
            context_prefix_singleton_matches=sum(
                a.context_prefix_singleton_matches for a in attempts
            ),
            hidden_context_fallbacks=sum(
                a.hidden_context_fallbacks for a in attempts
            ),
        )
        self._log_trace(trace)
        return last.text, trace

    def _log_trace(self, trace: GenerationTrace) -> None:
        logger.debug(
            "Generation trace: attempts=%s order=%s jumps=%s rejection=%s tokens=%s "
            "start_source=%s leading_punctuation_stripped=%s context_exact=%s "
            "context_casefold=%s context_prefix=%s prefix_singletons=%s "
            "hidden_context_fallbacks=%s",
            trace.attempts_used,
            trace.markov_order_used,
            trace.jump_count,
            trace.rejection_reason,
            trace.token_count,
            trace.start_source,
            trace.leading_punctuation_stripped,
            trace.context_exact_matches,
            trace.context_casefold_matches,
            trace.context_prefix_matches,
            trace.context_prefix_singleton_matches,
            trace.hidden_context_fallbacks,
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
        fuzzy_context_prefix: bool,
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
            fuzzy_context_prefix=fuzzy_context_prefix,
            rng=rng,
        )

    @staticmethod
    def _contextual_match_counts(
        selection: _ContextualStateSelection,
    ) -> tuple[int, int, int, int]:
        """Map a contextual match kind to its trace counters.

        Returns ``(exact, casefold, prefix, prefix_singleton)``; a prefix match
        is also a "singleton" when it rests on a single observed transition.
        """
        if selection.match_kind == "exact":
            return 1, 0, 0, 0
        if selection.match_kind == "casefold":
            return 0, 1, 0, 0
        return 0, 0, 1, (1 if selection.transition_count == 1 else 0)

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
        context_prefix_matches: int,
        context_prefix_singleton_matches: int,
        hidden_context_fallbacks: int,
    ) -> _GenerationAttempt:
        """Trim, strip and quality-gate generated tokens into an attempt.

        Applies the tail pipeline (repetitive-tail/sentence-boundary/ending
        trims, leading-punctuation strip) then rejects results that are too
        short, low-diversity, a short context copy, or context-heavy. All
        outcomes carry the same trace metadata accumulated by the caller.
        """
        generated = trim_repetitive_tail(generated)
        generated = trim_to_sentence_boundary(generated)
        generated = finalize_reply_ending(generated)
        finalized_token_count = len(generated)
        generated = strip_leading_punctuation(generated)
        leading_punctuation_stripped = finalized_token_count - len(generated)
        result = detokenize(generated, max_chars=max_chars)

        def reject(reason: str) -> _GenerationAttempt:
            return _GenerationAttempt(
                "",
                order_used,
                jump_count,
                reason,
                0,
                start_source,
                leading_punctuation_stripped,
                context_exact_matches,
                context_casefold_matches,
                context_prefix_matches,
                context_prefix_singleton_matches,
                hidden_context_fallbacks,
            )

        if len(result) < 5:
            return reject(REJECTION_RESULT_TOO_SHORT)
        result_tokens = tokenize(result)
        is_short_reply = is_short_generated_reply(result_tokens)
        if not is_short_reply and is_low_diversity_reply(result_tokens):
            return reject(REJECTION_LOW_DIVERSITY)
        if is_short_reply:
            if is_short_context_copy(result_tokens, context_tokens):
                return reject(REJECTION_SHORT_CONTEXT_COPY)
        elif is_context_heavy_reply(result_tokens, context_tokens):
            return reject(REJECTION_CONTEXT_HEAVY)
        return _GenerationAttempt(
            result,
            order_used,
            jump_count,
            None,
            len(result_tokens),
            start_source,
            leading_punctuation_stripped,
            context_exact_matches,
            context_casefold_matches,
            context_prefix_matches,
            context_prefix_singleton_matches,
            hidden_context_fallbacks,
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
        rng: random.Random,
    ) -> tuple[list[str], int, int]:
        """Walk the Markov chain from ``start3`` until a stop condition.

        Returns ``(generated, order_used, jump_count)``. ``emit_tokens`` is the
        prefix actually written into the output (the full start for a global or
        seed start, a trimmed tail or nothing for a contextual start) while the
        chain always walks from the full ``start3`` state. Per-step transitions
        fall back from order 3 to 2 when ``enable_backoff`` allows; dedup state
        suppresses immediate triplet repeats. Only local state is mutated.
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

        for step_index in range(self.max_steps):
            if len(generated) >= token_limit:
                break
            if (
                len(generated) >= JUMP_MIN_GENERATED_TOKENS
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
                connective = pick_jump_connective(
                    rng,
                    exclude=used_connectives
                    + [
                        phrase
                        for phrase in JUMP_CONNECTIVE_TOKENS
                        if nw1 in phrase
                    ],
                )
                used_connectives.append(connective)
                generated.extend(connective)
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
            if pool3 and order >= 3:
                candidates = [
                    (cand, cnt)
                    for cand, cnt in pool3
                    if (w2, w3, cand) not in visited_triplets
                ]
                pool = candidates or pool3
                w4 = weighted_next_choice(
                    pool,
                    next_explore,
                    next_power,
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
                )
            else:
                if not enable_backoff and order >= 3:
                    break

                pool2 = await self._get2(chat_id, w2, w3)
                if pool2:
                    w4 = weighted_next_choice(
                        pool2,
                        next_explore,
                        next_power,
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
                    )
                    order_used = min(order_used, 2)
                else:
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

        return generated, order_used, jump_count

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
        fuzzy_context_prefix: bool = False,
        context_emit_start: bool = False,
        jump_probability: float = 0.0,
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
            return _GenerationAttempt("", 0, 0, REJECTION_NO_STARTS, 0, "global")

        start3: tuple[str, str, str] | None = None
        start_source = "global"
        contextual_emit_tokens: list[str] = []
        context_exact_matches = 0
        context_casefold_matches = 0
        context_prefix_matches = 0
        context_prefix_singleton_matches = 0
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
                fuzzy_context_prefix=fuzzy_context_prefix,
                rng=generation_rng,
            )
            if contextual_state is not None:
                start3 = contextual_state.state
                order_used = contextual_state.order
                # Visible contextual start: emit the trimmed tail of the matched
                # window so the reply picks the context up out loud instead of
                # only continuing after it (design change over Phase 4.1d).
                contextual_emit_tokens = (
                    context_emission_tokens(start3) if context_emit_start else []
                )
                emit_start = False
                start_source = (
                    "context" if contextual_emit_tokens else "hidden_context"
                )
                (
                    context_exact_matches,
                    context_casefold_matches,
                    context_prefix_matches,
                    context_prefix_singleton_matches,
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
                    "",
                    2,
                    0,
                    REJECTION_NO_START_TRANSITION,
                    0,
                    start_source,
                    0,
                    context_exact_matches,
                    context_casefold_matches,
                    context_prefix_matches,
                    context_prefix_singleton_matches,
                    hidden_context_fallbacks,
                )
            start3, order_used = global_start

        emit_tokens = list(start3) if emit_start else contextual_emit_tokens
        generated, order_used, jump_count = await self._run_generation_loop(
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
            jump_probability=jump_probability,
            rng=generation_rng,
        )

        return self._finalize_attempt(
            generated,
            max_chars=max_chars,
            context_tokens=context_tokens,
            order_used=order_used,
            jump_count=jump_count,
            start_source=start_source,
            context_exact_matches=context_exact_matches,
            context_casefold_matches=context_casefold_matches,
            context_prefix_matches=context_prefix_matches,
            context_prefix_singleton_matches=context_prefix_singleton_matches,
            hidden_context_fallbacks=hidden_context_fallbacks,
        )
