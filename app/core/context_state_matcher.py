from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from app.core.markov_port import MarkovReadPort


@dataclass(frozen=True, slots=True)
class ContextStateMatch:
    state: tuple[str, ...]
    transition_count: int


@dataclass(slots=True)
class _StateIndex:
    exact: dict[tuple[str, ...], int]


class ContextStateMatcher:
    """Exact lookup of context windows among the chain's states.

    The casefold tier (a second index keyed by the folded window) was removed
    2026-09-11 (change remove-dead-knobs): the corpus learns in lower case, so
    it never produced a start the exact tier had not, and the knob census
    measured it inert in both modes. Matching folds nothing: case is the
    tokenizer's business (``normalize_lower``), morphology lives in
    ``context_start_affinity`` and IDF relevance.
    """

    def __init__(self, db: MarkovReadPort, cache_limit: int = 128) -> None:
        self._db = db
        self._cache_limit = max(1, cache_limit)
        self._cache: OrderedDict[tuple[int, int], _StateIndex] = OrderedDict()

    def invalidate_chat_cache(self, chat_id: int) -> None:
        for key in [key for key in self._cache if key[0] == chat_id]:
            self._cache.pop(key, None)

    def invalidate_all_caches(self) -> None:
        self._cache.clear()

    def apply_state_deltas(
        self,
        chat_id: int,
        order: int,
        deltas: dict[tuple[str, ...], int],
    ) -> None:
        """Fold learned-message state deltas into a cached index (M2R-030).

        Only an already-cached ``(chat, order)`` index is updated — a cold one
        rebuilds from SQL on demand; a folded index equals a freshly built one
        (enforced by tests).
        """
        index = self._cache.get((chat_id, order))
        if index is None or not deltas:
            return
        for state, delta in deltas.items():
            index.exact[state] = index.exact.get(state, 0) + delta

    async def match(
        self,
        chat_id: int,
        context_window: tuple[str, ...],
        order: int,
    ) -> list[ContextStateMatch]:
        if order not in {2, 3}:
            raise ValueError("order must be 2 or 3")
        if len(context_window) != order:
            raise ValueError("context_window length must match order")

        index = await self._get_index(chat_id, order)
        exact_count = index.exact.get(context_window)
        if exact_count is None:
            return []
        return [ContextStateMatch(state=context_window, transition_count=exact_count)]

    async def _get_index(self, chat_id: int, order: int) -> _StateIndex:
        key = (chat_id, order)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        rows = await self._db.get_states(chat_id, order)
        index = _StateIndex(
            exact={state: transition_count for state, transition_count in rows}
        )
        self._cache[key] = index
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return index
