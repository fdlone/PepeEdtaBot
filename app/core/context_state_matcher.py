from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from itertools import product

from app.infrastructure.database import Database

_MATCH_KIND_PRIORITY = {"exact": 0, "casefold": 1, "prefix": 2}
_CYRILLIC_WORD_RE = re.compile(r"[а-яё]+", re.IGNORECASE)
_MIN_CONTEXT_TOKEN_LENGTH = 5
_MIN_COMMON_PREFIX = 6
_MIN_SHORTER_COVERAGE = 0.75
_TOKEN_CANDIDATE_LIMIT = 6
_MIN_PREFIX_TRANSITION_COUNT = 2
_MIN_PREFIX_CONFIDENCE = 0.68
_STRING_WEIGHT = 0.65
_FREQUENCY_WEIGHT = 0.35
_FREQUENCY_REFERENCE_COUNT = 10


@dataclass(frozen=True, slots=True)
class ContextStateMatch:
    state: tuple[str, ...]
    match_kind: str
    similarity: float
    transition_count: int


@dataclass(slots=True)
class _StateIndex:
    exact: dict[tuple[str, ...], int]
    casefolded: dict[tuple[str, ...], tuple[tuple[tuple[str, ...], int], ...]]
    exact_tokens: frozenset[str]
    prefix_tokens: dict[str, tuple[tuple[str, int], ...]] | None = None


class ContextStateMatcher:
    def __init__(self, db: Database, cache_limit: int = 128) -> None:
        self._db = db
        self._cache_limit = max(1, cache_limit)
        self._cache: OrderedDict[tuple[int, int], _StateIndex] = OrderedDict()

    def invalidate_chat_cache(self, chat_id: int) -> None:
        for key in [key for key in self._cache if key[0] == chat_id]:
            self._cache.pop(key, None)

    async def match(
        self,
        chat_id: int,
        context_window: tuple[str, ...],
        order: int,
        include_prefix: bool = False,
    ) -> list[ContextStateMatch]:
        if order not in {2, 3}:
            raise ValueError("order must be 2 or 3")
        if len(context_window) != order:
            raise ValueError("context_window length must match order")

        index = await self._get_index(chat_id, order)
        matches: list[ContextStateMatch] = []
        exact_count = index.exact.get(context_window)
        if exact_count is not None:
            matches.append(
                ContextStateMatch(
                    state=context_window,
                    match_kind="exact",
                    similarity=1.0,
                    transition_count=exact_count,
                )
            )

        folded_window = tuple(token.casefold() for token in context_window)
        for state, transition_count in index.casefolded.get(folded_window, ()):
            if state == context_window:
                continue
            matches.append(
                ContextStateMatch(
                    state=state,
                    match_kind="casefold",
                    similarity=1.0,
                    transition_count=transition_count,
                )
            )

        if include_prefix:
            matches.extend(self._prefix_matches(context_window, index))

        return sorted(
            matches,
            key=lambda match: (
                _MATCH_KIND_PRIORITY[match.match_kind],
                -match.similarity,
                -match.transition_count,
                match.state,
            ),
        )

    def _prefix_matches(
        self,
        context_window: tuple[str, ...],
        index: _StateIndex,
    ) -> list[ContextStateMatch]:
        if index.prefix_tokens is None:
            index.prefix_tokens = self._build_prefix_index(index.exact)

        token_options: list[tuple[tuple[str, float, bool], ...]] = []
        for token in context_window:
            options = self._token_options(token, index)
            if not options:
                return []
            token_options.append(options)

        matches: list[ContextStateMatch] = []
        for combination in product(*token_options):
            if not any(is_prefix for _, _, is_prefix in combination):
                continue
            state = tuple(token for token, _, _ in combination)
            transition_count = index.exact.get(state)
            if (
                transition_count is None
                or transition_count < _MIN_PREFIX_TRANSITION_COUNT
            ):
                continue
            string_similarity = sum(
                similarity for _, similarity, _ in combination
            ) / len(combination)
            frequency_prior = min(
                1.0,
                math.log1p(transition_count)
                / math.log1p(_FREQUENCY_REFERENCE_COUNT),
            )
            confidence = (
                _STRING_WEIGHT * string_similarity
                + _FREQUENCY_WEIGHT * frequency_prior
            )
            if confidence < _MIN_PREFIX_CONFIDENCE:
                continue
            matches.append(
                ContextStateMatch(
                    state=state,
                    match_kind="prefix",
                    similarity=confidence,
                    transition_count=transition_count,
                )
            )
        return matches

    def _token_options(
        self,
        context_token: str,
        index: _StateIndex,
    ) -> tuple[tuple[str, float, bool], ...]:
        options: list[tuple[str, float, bool, int]] = (
            [
                (
                    context_token,
                    1.0,
                    False,
                    _FREQUENCY_REFERENCE_COUNT,
                )
            ]
            if context_token in index.exact_tokens
            else []
        )

        folded_context = context_token.casefold()
        if (
            len(folded_context) < _MIN_CONTEXT_TOKEN_LENGTH
            or _CYRILLIC_WORD_RE.fullmatch(folded_context) is None
            or index.prefix_tokens is None
        ):
            return tuple(
                (token, similarity, is_prefix)
                for token, similarity, is_prefix, _ in options
            )

        prefix_key = folded_context[:_MIN_COMMON_PREFIX]
        for candidate, max_count in index.prefix_tokens.get(prefix_key, ()):
            if candidate.casefold() == folded_context:
                continue
            similarity = _prefix_similarity(folded_context, candidate.casefold())
            if similarity is None:
                continue
            options.append((candidate, similarity, True, max_count))

        ordered = sorted(
            options,
            key=lambda item: (
                not (item[0] == context_token),
                -item[3],
                -item[1],
                item[0],
            ),
        )
        return tuple(
            (token, similarity, is_prefix)
            for token, similarity, is_prefix, _ in ordered[
                :_TOKEN_CANDIDATE_LIMIT
            ]
        )

    def _build_prefix_index(
        self,
        exact: dict[tuple[str, ...], int],
    ) -> dict[str, tuple[tuple[str, int], ...]]:
        token_counts: dict[str, int] = {}
        for state, transition_count in exact.items():
            for token in state:
                folded = token.casefold()
                if (
                    len(folded) < _MIN_COMMON_PREFIX
                    or _CYRILLIC_WORD_RE.fullmatch(folded) is None
                ):
                    continue
                token_counts[token] = max(
                    token_counts.get(token, 0),
                    transition_count,
                )

        grouped: dict[str, list[tuple[str, int]]] = {}
        for token, max_count in sorted(token_counts.items()):
            grouped.setdefault(
                token.casefold()[:_MIN_COMMON_PREFIX],
                [],
            ).append((token, max_count))
        return {
            prefix: tuple(
                sorted(
                    candidates,
                    key=lambda item: (-item[1], item[0]),
                )[:_TOKEN_CANDIDATE_LIMIT]
            )
            for prefix, candidates in sorted(grouped.items())
        }

    async def _get_index(self, chat_id: int, order: int) -> _StateIndex:
        key = (chat_id, order)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        rows = await self._db.get_markov_states(chat_id, order)
        exact = {state: transition_count for state, transition_count in rows}
        grouped: dict[
            tuple[str, ...],
            list[tuple[tuple[str, ...], int]],
        ] = {}
        for state, transition_count in rows:
            folded = tuple(token.casefold() for token in state)
            grouped.setdefault(folded, []).append((state, transition_count))
        casefolded = {
            folded: tuple(
                sorted(
                    states,
                    key=lambda item: (-item[1], item[0]),
                )
            )
            for folded, states in grouped.items()
        }
        index = _StateIndex(
            exact=exact,
            casefolded=casefolded,
            exact_tokens=frozenset(token for state in exact for token in state),
        )
        self._cache[key] = index
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return index


def _prefix_similarity(left: str, right: str) -> float | None:
    common_length = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        common_length += 1
    shorter_length = min(len(left), len(right))
    if (
        common_length < _MIN_COMMON_PREFIX
        or common_length / shorter_length < _MIN_SHORTER_COVERAGE
    ):
        return None
    return common_length / max(len(left), len(right))
