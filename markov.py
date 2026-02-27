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


def weighted_next_choice(items: list[tuple[str, int]], explore_probability: float, power: float) -> str:
    population = [token for token, _ in items]
    if random.random() < explore_probability:
        return random.choice(population)
    weights = [max(cnt, 1) ** power for _, cnt in items]
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

    async def generate_text(
        self,
        chat_id: int,
        max_chars: int,
        seed_tokens: Optional[list[str]] = None,
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
                    w3 = weighted_next_choice(variants, next_explore, next_power)
                    start3 = (w1, w2, w3)

        if start3 is None:
            if starts3:
                start3 = weighted_start3_choice(starts3, start_explore, start_power)
            elif starts2:
                w1, w2 = weighted_start2_choice(starts2, start_explore, start_power)
                variants = await self._get2(chat_id, w1, w2)
                if not variants:
                    return ""
                w3 = weighted_next_choice(variants, next_explore, next_power)
                start3 = (w1, w2, w3)

        if start3 is None:
            return ""

        w1, w2, w3 = start3
        generated: list[str] = [w1, w2, w3]
        visited_triplets: set[tuple[str, str, str]] = {(w1, w2, w3)}
        jump_count = 0

        for _ in range(self.max_steps):
            if len(generated) > 8 and random.random() < jump_probability and starts3 and order >= 3:
                nw1, nw2, nw3 = weighted_start3_choice(starts3, start_explore, start_power)
                w1, w2, w3 = nw1, nw2, nw3
                jump_count += 1
                continue

            pool3 = await self._get3(chat_id, w1, w2, w3) if order >= 3 else []
            if pool3 and order >= 3:
                candidates = [
                    (cand, cnt) for cand, cnt in pool3 if (w2, w3, cand) not in visited_triplets
                ]
                pool = candidates or pool3
                w4 = weighted_next_choice(pool, next_explore, next_power)
            else:
                if not enable_backoff and order >= 3:
                    break

                pool2 = await self._get2(chat_id, w2, w3)
                if pool2:
                    w4 = weighted_next_choice(pool2, next_explore, next_power)
                else:
                    if not enable_backoff or backoff_min_order > 1:
                        break
                    pool1 = await self._get1(chat_id, w3)
                    if not pool1:
                        break
                    w4 = weighted_next_choice(pool1, next_explore, next_power)

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
