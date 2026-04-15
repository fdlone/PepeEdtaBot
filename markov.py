from __future__ import annotations

import random
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from db import Database

TOKEN_RE = re.compile(r"\w+|[.,!?;:]", re.UNICODE)
PUNCT_SET = {".", ",", "!", "?", ";", ":"}


def tokenize(text: str, normalize_lower: bool = False) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    return [t.lower() for t in tokens] if normalize_lower else tokens


def detokenize(tokens: list[str], max_chars: int) -> str:
    if not tokens:
        return ""

    parts: list[str] = []
    for token in tokens:
        if not parts:
            parts.append(token)
            continue
        if token in PUNCT_SET:
            parts[-1] = parts[-1] + token
        else:
            parts.append(" " + token)

    text = "".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def build_windows(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    if size <= 0 or len(tokens) < size:
        return []
    return [tuple(tokens[idx : idx + size]) for idx in range(len(tokens) - size + 1)]


def content_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in PUNCT_SET]


def content_token_indexes(tokens: list[str]) -> list[int]:
    return [index for index, token in enumerate(tokens) if token not in PUNCT_SET]


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
    return len(counts) <= 2 and (max(counts.values()) / len(recent_content)) >= dominance_threshold


def find_repetitive_tail_start(
    tokens: list[str],
    min_tail_tokens: int = 5,
    min_prefix_tokens: int = 4,
    tail_scan_limit: int = 12,
    dominance_threshold: float = 0.7,
) -> Optional[int]:
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
        if len(counts) <= 2 and (max(counts.values()) / len(tail)) >= dominance_threshold:
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
    return len(counts) <= 2 and (max(counts.values()) / len(content)) >= dominance_threshold


def is_context_heavy_reply(generated_tokens: list[str], context_tokens: list[str]) -> bool:
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
    uses_only_context_tokens = all(token in context_token_set for token in generated_content)
    has_local_loops = len(set(generated_content)) <= max(2, len(generated_content) // 2)

    if uses_only_context_tokens and has_local_loops:
        return True
    if overlap_ratio >= 0.85 and shared_run >= 4:
        return True
    if shared_run >= max(4, len(generated_content) - 1):
        return True
    return False


def context_decay(step_index: int) -> float:
    if step_index <= 4:
        return 1.0
    if step_index <= 9:
        return 0.7
    return 0.4


def weighted_next_choice(
    items: list[tuple[str, int]],
    explore_probability: float,
    power: float,
    context_token_set: Optional[set[str]] = None,
    context_pairs: Optional[set[tuple[str, str]]] = None,
    context_triplets: Optional[set[tuple[str, str, str]]] = None,
    current_state: Optional[tuple[str, ...]] = None,
    context_bias: float = 1.0,
    step_index: int = 0,
    recent_tokens: Optional[list[str]] = None,
    seen_pairs: Optional[set[tuple[str, str]]] = None,
    seen_triplets: Optional[set[tuple[str, str, str]]] = None,
    repetition_penalty_strength: float = 1.0,
) -> str:
    population = [token for token, _ in items]
    if random.random() < explore_probability:
        return random.choice(population)

    step_bias = 1.0 + (max(1.0, context_bias) - 1.0) * context_decay(step_index)
    penalty_strength = max(0.0, repetition_penalty_strength)
    recent_tokens = recent_tokens or []
    weights: list[float] = []
    for token, cnt in items:
        weight = max(cnt, 1) ** power
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
    return random.choices(population=population, weights=weights, k=1)[0]


def weighted_start2_choice(items: list[tuple[str, str, int]], explore_probability: float, power: float) -> tuple[str, str]:
    population = [(w1, w2) for w1, w2, _ in items]
    if random.random() < explore_probability:
        return random.choice(population)
    weights = [max(cnt, 1) ** power for _, _, cnt in items]
    return random.choices(population=population, weights=weights, k=1)[0]


def weighted_start3_choice(
    items: list[tuple[str, str, str, int]], explore_probability: float, power: float
) -> tuple[str, str, str]:
    population = [(w1, w2, w3) for w1, w2, w3, _ in items]
    if random.random() < explore_probability:
        return random.choice(population)
    weights = [max(cnt, 1) ** power for _, _, _, cnt in items]
    return random.choices(population=population, weights=weights, k=1)[0]


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

    def invalidate_chat_cache(self, chat_id: int) -> None:
        for cache in (self._cache3, self._cache2, self._cache1):
            keys = [k for k in cache if k[0] == chat_id]
            for key in keys:
                cache.pop(key, None)

    def _touch_cache(self, cache: OrderedDict, key: tuple, value: list[tuple[str, int]]) -> None:
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.cache_limit:
            cache.popitem(last=False)

    async def _get3(self, chat_id: int, w1: str, w2: str, w3: str) -> list[tuple[str, int]]:
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
    ) -> Optional[tuple[str, str, str]]:
        windows = build_windows(context_tokens, 3)
        if not windows:
            return None

        candidates: list[tuple[tuple[str, str, str], float]] = []
        total = len(windows)
        for index, window in enumerate(windows):
            seeded3 = await self.db.get_start3_if_exists(chat_id, window[0], window[1], window[2])
            if not seeded3:
                continue
            recency_bonus = 1.0 + ((index + 1) / total) * 0.35
            weight = (max(seeded3[3], 1) ** power) * max(1.0, context_start_bias) * recency_bonus
            candidates.append(((seeded3[0], seeded3[1], seeded3[2]), weight))

        if not candidates:
            return None
        if random.random() < explore_probability:
            return random.choice([start for start, _ in candidates])
        return random.choices(
            population=[start for start, _ in candidates],
            weights=[weight for _, weight in candidates],
            k=1,
        )[0]

    async def _select_contextual_start2(
        self,
        chat_id: int,
        context_tokens: list[str],
        explore_probability: float,
        power: float,
        context_start_bias: float,
        next_explore: float,
        next_power: float,
        context_token_set: set[str],
        context_pairs: set[tuple[str, str]],
        context_triplets: set[tuple[str, str, str]],
        repetition_penalty_strength: float,
    ) -> Optional[tuple[str, str, str]]:
        windows = build_windows(context_tokens, 2)
        if not windows:
            return None

        candidates: list[tuple[tuple[str, str], float]] = []
        total = len(windows)
        for index, window in enumerate(windows):
            seeded2 = await self.db.get_start_if_exists(chat_id, window[0], window[1])
            if not seeded2:
                continue
            recency_bonus = 1.0 + ((index + 1) / total) * 0.30
            weight = (max(seeded2[2], 1) ** power) * max(1.0, context_start_bias) * recency_bonus
            candidates.append(((seeded2[0], seeded2[1]), weight))

        if not candidates:
            return None
        if random.random() < explore_probability:
            w1, w2 = random.choice([start for start, _ in candidates])
        else:
            w1, w2 = random.choices(
                population=[start for start, _ in candidates],
                weights=[weight for _, weight in candidates],
                k=1,
            )[0]

        variants = await self._get2(chat_id, w1, w2)
        if not variants:
            return None
        w3 = weighted_next_choice(
            variants,
            next_explore,
            next_power,
            context_token_set=context_token_set,
            context_pairs=context_pairs,
            context_triplets=context_triplets,
            current_state=(w1, w2),
            context_bias=context_start_bias,
            step_index=0,
            repetition_penalty_strength=repetition_penalty_strength,
        )
        return (w1, w2, w3)

    async def generate_text(
        self,
        chat_id: int,
        max_chars: int,
        seed_tokens: Optional[list[str]] = None,
        context_tokens: Optional[list[str]] = None,
        context_bias: float = 1.0,
        context_start_bias: float = 1.0,
        randomness_strength: float = 1.0,
        repetition_penalty_strength: float = 1.0,
        markov_order: int = 3,
        enable_backoff: bool = True,
        backoff_min_order: int = 1,
    ) -> str:
        order = 3 if markov_order >= 3 else 2
        strength = max(0.0, min(3.0, randomness_strength))
        next_explore = min(0.98, 0.12 + 0.18 * strength)
        next_power = max(0.15, 0.72 - 0.16 * strength)
        start_explore = min(0.98, 0.20 + 0.20 * strength)
        start_power = max(0.15, 0.75 - 0.18 * strength)
        jump_probability = min(0.32, 0.03 + 0.08 * strength)

        context_tokens = context_tokens or []
        context_token_set = set(context_tokens)
        context_pairs = set(build_windows(context_tokens, 2))
        context_triplets = set(build_windows(context_tokens, 3))

        starts3 = await self.db.get_starts3(chat_id) if order >= 3 else []
        starts2 = await self.db.get_starts(chat_id)
        if not starts3 and not starts2:
            return ""

        start3: Optional[tuple[str, str, str]] = None
        if seed_tokens and len(seed_tokens) >= 3:
            seeded3 = await self.db.get_start3_if_exists(
                chat_id, seed_tokens[0], seed_tokens[1], seed_tokens[2]
            )
            if seeded3:
                start3 = (seeded3[0], seeded3[1], seeded3[2])
        if start3 is None and seed_tokens and len(seed_tokens) >= 2:
            seeded2 = await self.db.get_start_if_exists(chat_id, seed_tokens[0], seed_tokens[1])
            if seeded2:
                w1, w2 = seeded2[0], seeded2[1]
                variants = await self._get2(chat_id, w1, w2)
                if variants:
                    w3 = weighted_next_choice(
                        variants,
                        next_explore,
                        next_power,
                        context_token_set=context_token_set,
                        context_pairs=context_pairs,
                        context_triplets=context_triplets,
                        current_state=(w1, w2),
                        context_bias=context_bias,
                        step_index=0,
                        repetition_penalty_strength=repetition_penalty_strength,
                    )
                    start3 = (w1, w2, w3)

        if start3 is None and context_tokens:
            if order >= 3:
                start3 = await self._select_contextual_start3(
                    chat_id, context_tokens, start_explore, start_power, context_start_bias
                )
            if start3 is None:
                start3 = await self._select_contextual_start2(
                    chat_id=chat_id,
                    context_tokens=context_tokens,
                    explore_probability=start_explore,
                    power=start_power,
                    context_start_bias=context_start_bias,
                    next_explore=next_explore,
                    next_power=next_power,
                    context_token_set=context_token_set,
                    context_pairs=context_pairs,
                    context_triplets=context_triplets,
                    repetition_penalty_strength=repetition_penalty_strength,
                )

        if start3 is None:
            if starts3:
                start3 = weighted_start3_choice(starts3, start_explore, start_power)
            elif starts2:
                w1, w2 = weighted_start2_choice(starts2, start_explore, start_power)
                variants = await self._get2(chat_id, w1, w2)
                if not variants:
                    return ""
                w3 = weighted_next_choice(
                    variants,
                    next_explore,
                    next_power,
                    context_token_set=context_token_set,
                    context_pairs=context_pairs,
                    context_triplets=context_triplets,
                    current_state=(w1, w2),
                    context_bias=context_bias,
                    step_index=0,
                    repetition_penalty_strength=repetition_penalty_strength,
                )
                start3 = (w1, w2, w3)

        if start3 is None:
            return ""

        w1, w2, w3 = start3
        generated: list[str] = [w1, w2, w3]
        visited_triplets: set[tuple[str, str, str]] = {(w1, w2, w3)}
        seen_pairs = set(build_windows(generated, 2))
        seen_triplets = set(build_windows(generated, 3))
        jump_count = 0

        for step_index in range(self.max_steps):
            if len(generated) > 8 and random.random() < jump_probability and starts3 and order >= 3:
                contextual_jump = None
                if context_tokens:
                    contextual_jump = await self._select_contextual_start3(
                        chat_id, context_tokens, start_explore, start_power, context_start_bias
                    )
                if contextual_jump is None:
                    contextual_jump = weighted_start3_choice(starts3, start_explore, start_power)
                w1, w2, w3 = contextual_jump
                jump_count += 1
                continue

            pool3 = await self._get3(chat_id, w1, w2, w3) if order >= 3 else []
            if pool3 and order >= 3:
                candidates = [
                    (cand, cnt) for cand, cnt in pool3 if (w2, w3, cand) not in visited_triplets
                ]
                pool = candidates or pool3
                w4 = weighted_next_choice(
                    pool,
                    next_explore,
                    next_power,
                    context_token_set=context_token_set,
                    context_pairs=context_pairs,
                    context_triplets=context_triplets,
                    current_state=(w2, w3),
                    context_bias=context_bias,
                    step_index=step_index,
                    recent_tokens=generated,
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
                        context_token_set=context_token_set,
                        context_pairs=context_pairs,
                        context_triplets=context_triplets,
                        current_state=(w2, w3),
                        context_bias=context_bias,
                        step_index=step_index,
                        recent_tokens=generated,
                        seen_pairs=seen_pairs,
                        seen_triplets=seen_triplets,
                        repetition_penalty_strength=repetition_penalty_strength,
                    )
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
                        context_token_set=context_token_set,
                        context_pairs=context_pairs,
                        current_state=(w3,),
                        context_bias=context_bias,
                        step_index=step_index,
                        recent_tokens=generated,
                        seen_pairs=seen_pairs,
                        repetition_penalty_strength=repetition_penalty_strength,
                    )

            generated.append(w4)
            if has_degraded_recent_window(generated):
                break
            maybe_text = detokenize(generated, max_chars=max_chars)
            if len(maybe_text) >= max_chars:
                break

            w1, w2, w3 = w2, w3, w4
            visited_triplets.add((w1, w2, w3))
            if len(generated) >= 2:
                seen_pairs.add((generated[-2], generated[-1]))
            if len(generated) >= 3:
                seen_triplets.add((generated[-3], generated[-2], generated[-1]))
            if len(visited_triplets) > 40:
                visited_triplets.pop()
            if len(seen_pairs) > 80:
                seen_pairs.pop()
            if len(seen_triplets) > 80:
                seen_triplets.pop()

        generated = trim_repetitive_tail(generated)
        result = detokenize(generated, max_chars=max_chars)
        if len(result) < 5:
            return ""
        if is_low_diversity_reply(generated):
            return ""
        if is_context_heavy_reply(generated, context_tokens):
            return ""
        if await self.db.message_exists(chat_id, result):
            return ""
        if strength >= 1.5 and len(generated) > 10 and jump_count == 0:
            return ""
        return result
