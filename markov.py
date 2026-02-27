from __future__ import annotations

import random
import re
from collections import OrderedDict
from dataclasses import dataclass
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


def weighted_choice(items: list[tuple[str, int]]) -> Optional[str]:
    if not items:
        return None
    population = [i[0] for i in items]
    weights = [i[1] for i in items]
    return random.choices(population=population, weights=weights, k=1)[0]


def weighted_start_choice(items: list[tuple[str, str, int]]) -> Optional[tuple[str, str]]:
    if not items:
        return None
    population = [(w1, w2) for w1, w2, _ in items]
    weights = [cnt for _, _, cnt in items]
    return random.choices(population=population, weights=weights, k=1)[0]


@dataclass(slots=True)
class MarkovGenerator:
    db: Database
    max_steps: int = 80
    cache_limit: int = 512

    def __post_init__(self) -> None:
        self._transition_cache: OrderedDict[tuple[int, str, str], list[tuple[str, int]]] = (
            OrderedDict()
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
    ) -> str:
        start_pair: Optional[tuple[str, str]] = None
        if seed_tokens and len(seed_tokens) >= 2:
            seeded = await self.db.get_start_if_exists(chat_id, seed_tokens[0], seed_tokens[1])
            if seeded:
                start_pair = (seeded[0], seeded[1])

        if not start_pair:
            starts = await self.db.get_starts(chat_id)
            start_pair = weighted_start_choice(starts)

        if not start_pair:
            return ""

        w1, w2 = start_pair
        generated = [w1, w2]

        for _ in range(self.max_steps):
            variants = await self._get_transitions_cached(chat_id, w1, w2)
            w3 = weighted_choice(variants)
            if not w3:
                break
            generated.append(w3)

            maybe_text = detokenize(generated, max_chars=max_chars)
            if len(maybe_text) >= max_chars:
                break

            w1, w2 = w2, w3

        result = detokenize(generated, max_chars=max_chars)
        if len(result) < 5:
            return ""
        return result
