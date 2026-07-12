from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from app.core.morphology import stem_token
from app.infrastructure.database import Database

_MATCH_KIND_PRIORITY = {"exact": 0, "casefold": 1, "stem": 2}


@dataclass(frozen=True, slots=True)
class ContextStateMatch:
    state: tuple[str, ...]
    match_kind: str
    similarity: float
    transition_count: int


def _stem_key(tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(stem_token(token.casefold()) for token in tokens)


@dataclass(slots=True)
class _StateIndex:
    exact: dict[tuple[str, ...], int]
    casefolded: dict[tuple[str, ...], tuple[tuple[tuple[str, ...], int], ...]]
    # Lazy: keyed by the stemmed casefolded window, built on first stem lookup
    # so the exact/casefold path pays nothing for it.
    stemmed: dict[tuple[str, ...], tuple[tuple[tuple[str, ...], int], ...]] | None = (
        None
    )


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
        include_stem: bool = False,
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

        if include_stem:
            matches.extend(self._stem_matches(context_window, folded_window, index))

        return sorted(
            matches,
            key=lambda match: (
                _MATCH_KIND_PRIORITY[match.match_kind],
                -match.similarity,
                -match.transition_count,
                match.state,
            ),
        )

    def _stem_matches(
        self,
        context_window: tuple[str, ...],
        folded_window: tuple[str, ...],
        index: _StateIndex,
    ) -> list[ContextStateMatch]:
        """Model states whose stemmed form equals the stemmed context window.

        Folds Russian inflection ("билеты стоят" matches the learned "билет
        стоит") through the same approximate stemmer that context-affine start
        sampling and IDF relevance already use, so morphology handling stays a
        single mechanism. States already reported as exact/casefold matches are
        skipped.
        """
        if index.stemmed is None:
            index.stemmed = self._build_stem_index(index.exact)

        matches: list[ContextStateMatch] = []
        for state, transition_count in index.stemmed.get(
            _stem_key(context_window), ()
        ):
            if state == context_window:
                continue
            if tuple(token.casefold() for token in state) == folded_window:
                continue
            matches.append(
                ContextStateMatch(
                    state=state,
                    match_kind="stem",
                    similarity=1.0,
                    transition_count=transition_count,
                )
            )
        return matches

    @staticmethod
    def _build_stem_index(
        exact: dict[tuple[str, ...], int],
    ) -> dict[tuple[str, ...], tuple[tuple[tuple[str, ...], int], ...]]:
        grouped: dict[tuple[str, ...], list[tuple[tuple[str, ...], int]]] = {}
        for state, transition_count in exact.items():
            grouped.setdefault(_stem_key(state), []).append(
                (state, transition_count)
            )
        return {
            stem: tuple(
                sorted(states, key=lambda item: (-item[1], item[0]))
            )
            for stem, states in sorted(grouped.items())
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
        )
        self._cache[key] = index
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return index
