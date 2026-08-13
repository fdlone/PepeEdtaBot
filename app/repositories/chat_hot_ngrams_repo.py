from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from app.repositories.base_repo import DecayableCountsRepo


class ChatHotNgramsRepo(DecayableCountsRepo):
    """Sliding-window content n-gram counts for the L1 running-jokes channel.

    Keyed by raw ``chat_id`` to match the Markov model tables; per-chat
    aggregate only (no author). Bigrams are stored with ``w3 = ''``. Hotness
    is the window count's share of the all-time count in ``transitions``
    (bigrams aggregate over its ``w3``): a spike means the chat picked the
    phrase up recently.
    """

    async def bump(self, chat_id: int, ngrams: Iterable[tuple[str, ...]]) -> None:
        """Add one occurrence per listed n-gram (no-op if empty)."""
        counts = Counter(ngram for ngram in ngrams if len(ngram) in (2, 3))
        if not counts:
            return
        rows = [
            (chat_id, ngram[0], ngram[1], ngram[2] if len(ngram) == 3 else "", n)
            for ngram, n in counts.items()
        ]
        await self._execute_many(
            """
            INSERT INTO chat_hot_ngrams(chat_id, w1, w2, w3, cnt)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, w1, w2, w3) DO UPDATE SET
                cnt = cnt + excluded.cnt,
                updated_at = datetime('now')
            """,
            rows,
        )

    # M2R-310 / design D5: the meme variant keeps the eligibility filters and
    # changes only the ordering — the registry's association score decides,
    # window count breaks ties. Trigrams take their best adjacent pair; an
    # n-gram absent from the registry scores 0 and keeps its count order. The
    # bigram rows store w3 = '', so the second OR arm cannot match for them.
    _MEME_SCORE_SQL = """, COALESCE((
                    SELECT MAX(c.meme_score) FROM markov_collocations c
                    WHERE c.chat_id = h.chat_id AND c.status = 'active'
                      AND ((c.left_token = h.w1 AND c.right_token = h.w2)
                           OR (c.left_token = h.w2 AND c.right_token = h.w3))
                  ), 0.0)"""

    async def get_hot(
        self,
        chat_id: int,
        *,
        min_count: int,
        recency_share: float,
        limit: int = 8,
        meme_ordering: bool = False,
    ) -> list[tuple[str, ...]]:
        """Top n-grams whose window count is a big share of their all-time count.

        ``recency_share`` in (0..1]: 0.5 means at least half of all recorded
        occurrences happened inside the current window. Missing long-term rows
        fall back to the window count itself (share 1.0) so a brand-new meme
        still qualifies.

        ``meme_ordering`` (M2R-310) reorders the same selection by the
        collocation registry's association score. Off by default.

        Both orderings break ties on the n-gram itself. The result feeds
        ``random.choice`` on the reply path, so an undefined tie order would be
        an undefined draw (``generation-sampling-determinism``). Adding the
        tie-break is deliberate and does not move ``generation_hash`` today:
        at the default thresholds this read returns nothing on the frozen
        snapshot, so there are no ties to order. It stops being free the moment
        the hotness thresholds are lowered (M3R-145), which is why it lands
        before that sweep rather than with it.
        """
        # Всевременной счётчик берётся коррелированным подзапросом на строку
        # окна, а не агрегатом по всей таблице переходов чата. Окно — десятки
        # строк, таблица переходов — десятки тысяч: прежняя форма
        # (GROUP BY по всему чату для биграмм) стоила ~41 мс на вызов на копии
        # прода, а вызывается это почти на каждом ответе. Коррелированная форма
        # идёт по префиксу первичного ключа и укладывается в сотые доли
        # миллисекунды. Результат тот же: отсутствующая строка по-прежнему
        # означает «считаем всевременным счётчиком оконный».
        score = self._MEME_SCORE_SQL if meme_ordering else ""
        # Хвост ``, 1, 2, 3`` — тай-брейк по самой n-грамме: без него порядок
        # строк с равным счётчиком не определён, а результат идёт в
        # ``random.choice`` на пути ответа.
        order = (
            "ORDER BY 5 DESC, 4 DESC, 1, 2, 3"
            if meme_ordering
            else "ORDER BY 4 DESC, 1, 2, 3"
        )
        query = f"""
            SELECT h.w1, h.w2, h.w3, h.cnt{score}
            FROM chat_hot_ngrams h
            WHERE h.chat_id = ? AND h.w3 <> '' AND h.cnt >= ?
              AND h.cnt * 1.0 / MAX(COALESCE((
                    SELECT t.cnt FROM transitions t
                    WHERE t.chat_id = h.chat_id
                      AND t.w1 = h.w1 AND t.w2 = h.w2 AND t.w3 = h.w3
                  ), h.cnt), h.cnt) >= ?
            UNION ALL
            SELECT h.w1, h.w2, h.w3, h.cnt{score}
            FROM chat_hot_ngrams h
            WHERE h.chat_id = ? AND h.w3 = '' AND h.cnt >= ?
              AND h.cnt * 1.0 / MAX(COALESCE((
                    SELECT SUM(t.cnt) FROM transitions t
                    WHERE t.chat_id = h.chat_id
                      AND t.w1 = h.w1 AND t.w2 = h.w2
                  ), h.cnt), h.cnt) >= ?
            {order}
            LIMIT ?
        """
        params = (
            chat_id,
            min_count,
            recency_share,
            chat_id,
            min_count,
            recency_share,
            limit,
        )
        rows = await self._fetch_all(query, params)
        result: list[tuple[str, ...]] = []
        for w1, w2, w3, *_counts in rows:
            if w3:
                result.append((str(w1), str(w2), str(w3)))
            else:
                result.append((str(w1), str(w2)))
        return result

    async def decay_stale(self, cutoff_iso: str) -> int:
        """Halve counts of rows not bumped since ``cutoff_iso``; purge zeros."""
        return await self._decay_stale("chat_hot_ngrams", cutoff_iso)
