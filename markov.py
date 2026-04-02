from __future__ import annotations

import random
import re
from collections import OrderedDict
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
) -> str:
    population = [token for token, _ in items]
    if random.random() < explore_probability:
        return random.choice(population)

    step_bias = 1.0 + (max(1.0, context_bias) - 1.0) * context_decay(step_index)
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
                )
                start3 = (w1, w2, w3)

        if start3 is None:
            return ""

        w1, w2, w3 = start3
        generated: list[str] = [w1, w2, w3]
        visited_triplets: set[tuple[str, str, str]] = {(w1, w2, w3)}
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
                    )

            generated.append(w4)
            maybe_text = detokenize(generated, max_chars=max_chars)
            if len(maybe_text) >= max_chars:
                break

            w1, w2, w3 = w2, w3, w4
            visited_triplets.add((w1, w2, w3))
            if len(visited_triplets) > 40:
                visited_triplets.pop()

        result = detokenize(generated, max_chars=max_chars)
        if len(result) < 5:
            return ""
        if await self.db.message_exists(chat_id, result):
            return ""
        if strength >= 1.5 and len(generated) > 10 and jump_count == 0:
            return ""
        return result
