from __future__ import annotations

import logging
import random
import re
from collections import Counter, OrderedDict
from collections.abc import Container
from dataclasses import dataclass, field
from typing import NamedTuple

from app.core.context_state_matcher import ContextStateMatcher
from app.core.lexicon import BAD_ENDING_WORDS
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
    hidden_context_fallbacks: int = 0


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
    exploring = rng.random() < explore_probability
    frequency_power = exploration_adjusted_power(
        power,
        sampled_exploration(explore_probability, exploring, rng),
    )
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
    if exploring:
        return exploration_weighted_choice(population, weights, rng)
    return rng.choices(population=population, weights=weights, k=1)[0]


def weighted_start2_choice(
    items: list[tuple[str, str, int]],
    explore_probability: float,
    power: float,
    rng: random.Random,
) -> tuple[str, str]:
    ordered_items = sorted(items, key=lambda item: (item[0], item[1]))
    population = [(w1, w2) for w1, w2, _ in ordered_items]
    exploring = rng.random() < explore_probability
    frequency_power = exploration_adjusted_power(
        power,
        sampled_exploration(explore_probability, exploring, rng),
    )
    weights = [max(cnt, 1) ** frequency_power for _, _, cnt in ordered_items]
    if exploring:
        return exploration_weighted_choice(population, weights, rng)
    return rng.choices(population=population, weights=weights, k=1)[0]


def weighted_start3_choice(
    items: list[tuple[str, str, str, int]],
    explore_probability: float,
    power: float,
    rng: random.Random,
) -> tuple[str, str, str]:
    ordered_items = sorted(items, key=lambda item: (item[0], item[1], item[2]))
    population = [(w1, w2, w3) for w1, w2, w3, _ in ordered_items]
    exploring = rng.random() < explore_probability
    frequency_power = exploration_adjusted_power(
        power,
        sampled_exploration(explore_probability, exploring, rng),
    )
    weights = [max(cnt, 1) ** frequency_power for _, _, _, cnt in ordered_items]
    if exploring:
        return exploration_weighted_choice(population, weights, rng)
    return rng.choices(population=population, weights=weights, k=1)[0]


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
    _cache1: OrderedDict[tuple[int, str], list[tuple[str, int]]] = field(
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
        for key1 in [k for k in self._cache1 if k[0] == chat_id]:
            self._cache1.pop(key1, None)
        self._cache_starts3.pop(chat_id, None)
        self._cache_starts2.pop(chat_id, None)
        self._context_state_matcher.invalidate_chat_cache(chat_id)

    def _touch_cache[K](
        self,
        cache: OrderedDict[K, list[tuple[str, int]]],
        key: K,
        value: list[tuple[str, int]],
    ) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.cache_limit:
            cache.popitem(last=False)

    async def _get_starts3(self, chat_id: int) -> list[tuple[str, str, str, int]]:
        if chat_id in self._cache_starts3:
            return self._cache_starts3[chat_id]
        rows = await self.db.get_starts3(chat_id)
        self._cache_starts3[chat_id] = rows
        if len(self._cache_starts3) > self.cache_limit:
            self._cache_starts3.popitem(last=False)
        return rows

    async def _get_starts2(self, chat_id: int) -> list[tuple[str, str, int]]:
        if chat_id in self._cache_starts2:
            return self._cache_starts2[chat_id]
        rows = await self.db.get_starts(chat_id)
        self._cache_starts2[chat_id] = rows
        if len(self._cache_starts2) > self.cache_limit:
            self._cache_starts2.popitem(last=False)
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

    async def _get1(self, chat_id: int, w1: str) -> list[tuple[str, int]]:
        key = (chat_id, w1)
        if key in self._cache1:
            self._cache1.move_to_end(key)
            return self._cache1[key]
        rows = await self.db.get_transitions1(chat_id, w1)
        self._touch_cache(self._cache1, key, rows)
        return rows

    async def _select_contextual_start3(
        self,
        chat_id: int,
        context_tokens: list[str],
        explore_probability: float,
        power: float,
        context_start_bias: float,
        rng: random.Random,
    ) -> tuple[str, str, str] | None:
        windows = build_windows(context_tokens, 3)
        if not windows:
            return None

        exploring = rng.random() < explore_probability
        frequency_power = exploration_adjusted_power(
            power,
            sampled_exploration(explore_probability, exploring, rng),
        )
        candidates: list[tuple[tuple[str, str, str], float]] = []
        total = len(windows)
        for index, window in enumerate(windows):
            seeded3 = await self.db.get_start3_if_exists(
                chat_id, window[0], window[1], window[2]
            )
            if not seeded3:
                continue
            recency_bonus = 1.0 + ((index + 1) / total) * 0.35
            weight = (
                (max(seeded3[3], 1) ** frequency_power)
                * max(1.0, context_start_bias)
                * recency_bonus
            )
            candidates.append(((seeded3[0], seeded3[1], seeded3[2]), weight))

        if not candidates:
            return None
        population = [start for start, _ in candidates]
        weights = [weight for _, weight in candidates]
        if exploring:
            return exploration_weighted_choice(population, weights, rng)
        return rng.choices(population=population, weights=weights, k=1)[0]

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
        exploring = rng.random() < start_explore
        frequency_power = exploration_adjusted_power(
            start_power,
            sampled_exploration(start_explore, exploring, rng),
        )
        windows3 = build_windows(context_tokens, 3)
        candidates3: list[tuple[tuple[str, str, str], float]] = []
        total3 = len(windows3)
        for index, window in enumerate(windows3):
            transitions = await self._get3(chat_id, window[0], window[1], window[2])
            if transitions:
                transition_count = sum(count for _, count in transitions)
                recency_bonus = 1.0 + ((index + 1) / total3) * 0.35
                weight = max(transition_count, 1) ** frequency_power * recency_bonus
                candidates3.append(((window[0], window[1], window[2]), weight))

        if candidates3:
            population3 = [state for state, _ in candidates3]
            weights3 = [weight for _, weight in candidates3]
            if exploring:
                state3 = exploration_weighted_choice(population3, weights3, rng)
            else:
                state3 = rng.choices(population=population3, weights=weights3, k=1)[0]
            return _ContextualStateSelection(state3, 3, "exact")

        windows2 = build_windows(context_tokens, 2)
        candidates2: list[tuple[tuple[str, str], list[tuple[str, int]], float]] = []
        total2 = len(windows2)
        for index, window in enumerate(windows2):
            transitions = await self._get2(chat_id, window[0], window[1])
            if transitions:
                transition_count = sum(count for _, count in transitions)
                recency_bonus = 1.0 + ((index + 1) / total2) * 0.30
                weight = max(transition_count, 1) ** frequency_power * recency_bonus
                candidates2.append(((window[0], window[1]), transitions, weight))

        match_kind = "exact"
        if not candidates2 and fuzzy_context_casefold:
            casefold_candidates3: list[
                tuple[tuple[str, str, str], float]
            ] = []
            for index, window in enumerate(windows3):
                matches = await self._context_state_matcher.match(
                    chat_id,
                    window,
                    3,
                )
                recency_bonus = 1.0 + ((index + 1) / total3) * 0.35
                for match in matches:
                    if match.match_kind != "casefold":
                        continue
                    state3 = (match.state[0], match.state[1], match.state[2])
                    weight = (
                        max(match.transition_count, 1) ** frequency_power
                        * recency_bonus
                    )
                    casefold_candidates3.append((state3, weight))

            if casefold_candidates3:
                population3 = [state for state, _ in casefold_candidates3]
                weights3 = [weight for _, weight in casefold_candidates3]
                if exploring:
                    selected_state3 = exploration_weighted_choice(
                        population3,
                        weights3,
                        rng,
                    )
                else:
                    selected_state3 = rng.choices(
                        population=population3,
                        weights=weights3,
                        k=1,
                    )[0]
                return _ContextualStateSelection(
                    selected_state3,
                    3,
                    "casefold",
                )

            for index, window in enumerate(windows2):
                matches = await self._context_state_matcher.match(
                    chat_id,
                    window,
                    2,
                )
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
                    candidates2.append((state2, transitions, weight))
            match_kind = "casefold"

        if not candidates2:
            return None
        population2 = [index for index in range(len(candidates2))]
        weights2 = [weight for _, _, weight in candidates2]
        if exploring:
            selected_index = exploration_weighted_choice(population2, weights2, rng)
        else:
            selected_index = rng.choices(
                population=population2,
                weights=weights2,
                k=1,
            )[0]
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
        max_tokens: int = 45,
        seed_tokens: list[str] | None = None,
        context_tokens: list[str] | None = None,
        context_bias: float = 1.0,
        context_start_bias: float = 1.0,
        randomness_strength: float = 1.0,
        repetition_penalty_strength: float = 1.0,
        markov_order: int = 3,
        enable_backoff: bool = True,
        backoff_min_order: int = 1,
        fuzzy_context_casefold: bool = False,
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
            randomness_strength=randomness_strength,
            repetition_penalty_strength=repetition_penalty_strength,
            markov_order=markov_order,
            enable_backoff=enable_backoff,
            backoff_min_order=backoff_min_order,
            fuzzy_context_casefold=fuzzy_context_casefold,
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
        randomness_strength: float = 1.0,
        repetition_penalty_strength: float = 1.0,
        markov_order: int = 3,
        enable_backoff: bool = True,
        backoff_min_order: int = 1,
        fuzzy_context_casefold: bool = False,
        rng: random.Random | None = None,
        attempt_budget: int = DEFAULT_GENERATION_ATTEMPT_BUDGET,
    ) -> tuple[str, GenerationTrace]:
        generation_rng = rng or random.Random()
        total_attempts = max(1, attempt_budget)
        last_attempt = _GenerationAttempt(
            text="",
            markov_order_used=0,
            jump_count=0,
            rejection_reason=REJECTION_NO_STARTS,
            token_count=0,
            start_source="global",
        )
        total_jump_count = 0
        total_leading_punctuation_stripped = 0
        total_context_exact_matches = 0
        total_context_casefold_matches = 0
        total_hidden_context_fallbacks = 0
        for attempt_index in range(total_attempts):
            attempt_randomness_strength = escalated_randomness_strength(
                randomness_strength,
                attempt_index,
                total_attempts,
            )
            last_attempt = await self._generate_text_once(
                chat_id=chat_id,
                max_chars=max_chars,
                max_tokens=max_tokens,
                seed_tokens=seed_tokens,
                context_tokens=context_tokens,
                context_bias=context_bias,
                context_start_bias=context_start_bias,
                randomness_strength=attempt_randomness_strength,
                repetition_penalty_strength=repetition_penalty_strength,
                markov_order=markov_order,
                enable_backoff=enable_backoff,
                backoff_min_order=backoff_min_order,
                fuzzy_context_casefold=fuzzy_context_casefold,
                rng=generation_rng,
                emit_start=True,
            )
            total_jump_count += last_attempt.jump_count
            total_leading_punctuation_stripped += (
                last_attempt.leading_punctuation_stripped
            )
            total_context_exact_matches += last_attempt.context_exact_matches
            total_context_casefold_matches += last_attempt.context_casefold_matches
            total_hidden_context_fallbacks += last_attempt.hidden_context_fallbacks
            if last_attempt.text:
                trace = GenerationTrace(
                    attempts_used=attempt_index + 1,
                    markov_order_used=last_attempt.markov_order_used,
                    jump_count=total_jump_count,
                    rejection_reason=None,
                    token_count=last_attempt.token_count,
                    start_source=last_attempt.start_source,
                    leading_punctuation_stripped=(
                        total_leading_punctuation_stripped
                    ),
                    context_exact_matches=total_context_exact_matches,
                    context_casefold_matches=total_context_casefold_matches,
                    hidden_context_fallbacks=total_hidden_context_fallbacks,
                )
                self._log_trace(trace)
                return last_attempt.text, trace

        trace = GenerationTrace(
            attempts_used=total_attempts,
            markov_order_used=last_attempt.markov_order_used,
            jump_count=total_jump_count,
            rejection_reason=last_attempt.rejection_reason,
            token_count=0,
            start_source=last_attempt.start_source,
            leading_punctuation_stripped=total_leading_punctuation_stripped,
            context_exact_matches=total_context_exact_matches,
            context_casefold_matches=total_context_casefold_matches,
            hidden_context_fallbacks=total_hidden_context_fallbacks,
        )
        self._log_trace(trace)
        return "", trace

    def _log_trace(self, trace: GenerationTrace) -> None:
        logger.debug(
            "Generation trace: attempts=%s order=%s jumps=%s rejection=%s tokens=%s "
            "start_source=%s leading_punctuation_stripped=%s context_exact=%s "
            "context_casefold=%s hidden_context_fallbacks=%s",
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
        )

    async def _generate_text_once(
        self,
        chat_id: int,
        max_chars: int,
        max_tokens: int = 45,
        seed_tokens: list[str] | None = None,
        context_tokens: list[str] | None = None,
        context_bias: float = 1.0,
        context_start_bias: float = 1.0,
        randomness_strength: float = 1.0,
        repetition_penalty_strength: float = 1.0,
        markov_order: int = 3,
        enable_backoff: bool = True,
        backoff_min_order: int = 1,
        fuzzy_context_casefold: bool = False,
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
        # Disabled until a jump can splice tokens into the returned text safely.
        jump_probability = 0.0

        context_tokens = context_tokens or []
        context_token_set = set(context_tokens)
        context_pairs = set(build_windows(context_tokens, 2))
        context_triplets = set(build_windows(context_tokens, 3))

        starts3 = await self._get_starts3(chat_id) if order >= 3 else []
        starts2 = await self._get_starts2(chat_id)
        if not starts3 and not starts2:
            return _GenerationAttempt("", 0, 0, REJECTION_NO_STARTS, 0, "global")

        start3: tuple[str, str, str] | None = None
        start_source = "global"
        context_exact_matches = 0
        context_casefold_matches = 0
        hidden_context_fallbacks = 0
        if seed_tokens and len(seed_tokens) >= 3:
            seeded3 = await self.db.get_start3_if_exists(
                chat_id, seed_tokens[0], seed_tokens[1], seed_tokens[2]
            )
            if seeded3:
                start3 = (seeded3[0], seeded3[1], seeded3[2])
                start_source = "seed"
        if start3 is None and seed_tokens and len(seed_tokens) >= 2:
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
                        generation_rng,
                        context_token_set=context_token_set,
                        context_pairs=context_pairs,
                        context_triplets=context_triplets,
                        current_state=(w1, w2),
                        context_bias=context_bias,
                        step_index=0,
                        repetition_penalty_strength=repetition_penalty_strength,
                    )
                    start3 = (w1, w2, w3)
                    order_used = 2
                    start_source = "seed"

        use_contextual_start = generation_rng.random() < context_start_probability(
            context_start_bias
        )
        if start3 is None and context_tokens and use_contextual_start:
            contextual_state = await self._select_contextual_state(
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
                rng=generation_rng,
            )
            if contextual_state is not None:
                start3 = contextual_state.state
                order_used = contextual_state.order
                emit_start = False
                start_source = "hidden_context"
                if contextual_state.match_kind == "exact":
                    context_exact_matches = 1
                else:
                    context_casefold_matches = 1
            else:
                hidden_context_fallbacks = 1

        if start3 is None:
            if starts3:
                start3 = weighted_start3_choice(
                    starts3,
                    start_explore,
                    start_power,
                    generation_rng,
                )
            elif starts2:
                w1, w2 = weighted_start2_choice(
                    starts2,
                    start_explore,
                    start_power,
                    generation_rng,
                )
                variants = await self._get2(chat_id, w1, w2)
                if not variants:
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
                        hidden_context_fallbacks,
                    )
                w3 = weighted_next_choice(
                    variants,
                    next_explore,
                    next_power,
                    generation_rng,
                    context_token_set=context_token_set,
                    context_pairs=context_pairs,
                    context_triplets=context_triplets,
                    current_state=(w1, w2),
                    context_bias=context_bias,
                    step_index=0,
                    repetition_penalty_strength=repetition_penalty_strength,
                )
                start3 = (w1, w2, w3)
                order_used = 2

        if start3 is None:
            return _GenerationAttempt(
                "",
                order_used,
                0,
                REJECTION_NO_START_TRANSITION,
                0,
                start_source,
                0,
                context_exact_matches,
                context_casefold_matches,
                hidden_context_fallbacks,
            )

        token_limit = max(1, max_tokens)
        w1, w2, w3 = start3
        start_tokens = [w1, w2, w3]
        generated: list[str] = start_tokens[:token_limit] if emit_start else []
        dedup_seed = generated if emit_start else start_tokens
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

        for step_index in range(self.max_steps):
            if len(generated) >= token_limit:
                break
            if (
                len(generated) > 8
                and generation_rng.random() < jump_probability
                and starts3
                and order >= 3
            ):
                contextual_jump = None
                if context_tokens:
                    contextual_jump = await self._select_contextual_start3(
                        chat_id,
                        context_tokens,
                        start_explore,
                        start_power,
                        context_start_bias,
                        generation_rng,
                    )
                if contextual_jump is None:
                    contextual_jump = weighted_start3_choice(
                        starts3,
                        start_explore,
                        start_power,
                        generation_rng,
                    )
                w1, w2, w3 = contextual_jump
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
                    generation_rng,
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
                        generation_rng,
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
                    if not enable_backoff or backoff_min_order > 1:
                        break
                    pool1 = await self._get1(chat_id, w3)
                    if not pool1:
                        break
                    w4 = weighted_next_choice(
                        pool1,
                        next_explore,
                        next_power,
                        generation_rng,
                        context_token_set=context_token_set,
                        context_pairs=context_pairs,
                        current_state=(w3,),
                        context_bias=context_bias,
                        step_index=step_index,
                        recent_tokens=recent_window,
                        seen_pairs=seen_pairs,
                        repetition_penalty_strength=repetition_penalty_strength,
                    )
                    order_used = 1

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

        generated = trim_repetitive_tail(generated)
        generated = trim_to_sentence_boundary(generated)
        generated = finalize_reply_ending(generated)
        finalized_token_count = len(generated)
        generated = strip_leading_punctuation(generated)
        leading_punctuation_stripped = finalized_token_count - len(generated)
        result = detokenize(generated, max_chars=max_chars)
        if len(result) < 5:
            return _GenerationAttempt(
                "",
                order_used,
                jump_count,
                REJECTION_RESULT_TOO_SHORT,
                0,
                start_source,
                leading_punctuation_stripped,
                context_exact_matches,
                context_casefold_matches,
                hidden_context_fallbacks,
            )
        result_tokens = tokenize(result)
        is_short_reply = is_short_generated_reply(result_tokens)
        if not is_short_reply and is_low_diversity_reply(result_tokens):
            return _GenerationAttempt(
                "",
                order_used,
                jump_count,
                REJECTION_LOW_DIVERSITY,
                0,
                start_source,
                leading_punctuation_stripped,
                context_exact_matches,
                context_casefold_matches,
                hidden_context_fallbacks,
            )
        if is_short_reply:
            if is_short_context_copy(result_tokens, context_tokens):
                return _GenerationAttempt(
                    "",
                    order_used,
                    jump_count,
                    REJECTION_SHORT_CONTEXT_COPY,
                    0,
                    start_source,
                    leading_punctuation_stripped,
                    context_exact_matches,
                    context_casefold_matches,
                    hidden_context_fallbacks,
                )
        elif is_context_heavy_reply(result_tokens, context_tokens):
            return _GenerationAttempt(
                "",
                order_used,
                jump_count,
                REJECTION_CONTEXT_HEAVY,
                0,
                start_source,
                leading_punctuation_stripped,
                context_exact_matches,
                context_casefold_matches,
                hidden_context_fallbacks,
            )
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
            hidden_context_fallbacks,
        )
