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
    if normalize_lower:
        return [t.lower() for t in tokens]
    return tokens


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


def weighted_choice(
    items: list[tuple[str, int]],
    explore_probability: float,
    weight_power: float,
) -> Optional[str]:
    if not items:
        return None
    population = [token for token, _ in items]
    if random.random() < explore_probability:
        return random.choice(population)

    # Flatten high frequencies so rare continuations are sampled more often.
    weights = [max(cnt, 1) ** weight_power for _, cnt in items]
    return random.choices(population=population, weights=weights, k=1)[0]


def weighted_start_choice(
    items: list[tuple[str, str, int]],
    explore_probability: float,
    weight_power: float,
) -> Optional[tuple[str, str]]:
    if not items:
        return None
    population = [(w1, w2) for w1, w2, _ in items]
    if random.random() < explore_probability:
        return random.choice(population)

    weights = [max(cnt, 1) ** weight_power for _, _, cnt in items]
    return random.choices(population=population, weights=weights, k=1)[0]


@dataclass(slots=True)
class MarkovGenerator:
    db: Database
    max_steps: int = 80
    cache_limit: int = 512
    _transition_cache: OrderedDict[tuple[int, str, str], list[tuple[str, int]]] = field(
        default_factory=OrderedDict, init=False
    )

    def invalidate_chat_cache(self, chat_id: int) -> None:
        keys_to_delete = [k for k in self._transition_cache if k[0] == chat_id]
        for key in keys_to_delete:
            self._transition_cache.pop(key, None)

    async def _get_transitions_cached(
        self, chat_id: int, w1: str, w2: str
    ) -> list[tuple[str, int]]:
        key = (chat_id, w1, w2)
        if key in self._transition_cache:
            self._transition_cache.move_to_end(key)
            return self._transition_cache[key]

        rows = await self.db.get_transitions(chat_id, w1, w2)
        self._transition_cache[key] = rows
        self._transition_cache.move_to_end(key)
        if len(self._transition_cache) > self.cache_limit:
            self._transition_cache.popitem(last=False)
        return rows

    async def generate_text(
        self,
        chat_id: int,
        max_chars: int,
        seed_tokens: Optional[list[str]] = None,
        randomness_strength: float = 1.0,
    ) -> str:
        strength = max(0.0, min(3.0, randomness_strength))
        next_explore = min(0.95, 0.20 + 0.25 * strength)
        next_power = max(0.12, 0.70 - 0.18 * strength)
        start_explore = min(0.98, 0.30 + 0.24 * strength)
        start_power = max(0.10, 0.65 - 0.18 * strength)
        jump_probability = min(0.40, 0.06 + 0.10 * strength)

        start_pair: Optional[tuple[str, str]] = None
        if seed_tokens and len(seed_tokens) >= 2:
            seeded = await self.db.get_start_if_exists(chat_id, seed_tokens[0], seed_tokens[1])
            if seeded:
                start_pair = (seeded[0], seeded[1])

        if not start_pair:
            starts = await self.db.get_starts(chat_id)
            start_pair = weighted_start_choice(
                starts, explore_probability=start_explore, weight_power=start_power
            )

        if not start_pair:
            return ""

        w1, w2 = start_pair
        generated = [w1, w2]
        recent_pairs: set[tuple[str, str]] = {(w1, w2)}
        starts_cache: Optional[list[tuple[str, str, int]]] = None
        jump_count = 0

        for _ in range(self.max_steps):
            # Occasional jump to a random frequent start pair to avoid near-copy sequences.
            if len(generated) > 6 and random.random() < jump_probability:
                if starts_cache is None:
                    starts_cache = await self.db.get_starts(chat_id)
                jumped = weighted_start_choice(
                    starts_cache,
                    explore_probability=start_explore,
                    weight_power=start_power,
                )
                if jumped:
                    w1, w2 = jumped
                    recent_pairs.add((w1, w2))
                    jump_count += 1
                    if len(recent_pairs) > 30:
                        recent_pairs.pop()
                    continue

            variants = await self._get_transitions_cached(chat_id, w1, w2)
            if not variants:
                break

            # Avoid trivial short loops if alternatives exist.
            filtered_variants = [
                (candidate, cnt)
                for candidate, cnt in variants
                if (w2, candidate) not in recent_pairs
            ]
            pool = filtered_variants or variants

            w3 = weighted_choice(
                pool, explore_probability=next_explore, weight_power=next_power
            )
            if not w3:
                break
            generated.append(w3)

            maybe_text = detokenize(generated, max_chars=max_chars)
            if len(maybe_text) >= max_chars:
                break

            w1, w2 = w2, w3
            recent_pairs.add((w1, w2))
            if len(recent_pairs) > 30:
                recent_pairs.pop()

        result = detokenize(generated, max_chars=max_chars)
        # If the generated text is exactly equal to one stored message, retry path in caller.
        if result and await self.db.message_exists(chat_id, result):
            return ""
        # Encourage at least one context jump on stronger randomness to reduce copy-like outputs.
        if strength >= 1.5 and jump_count == 0 and len(generated) > 8:
            return ""
        if len(result) < 5:
            return ""
        return result
