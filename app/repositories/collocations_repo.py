"""Per-chat collocation registry (M2R-300/320, TZ §10.2, §13).

Reads and writes the association-scored word pairs. The analyzer's arithmetic
lives in ``app/core/collocations.py``; the aggregates it needs are read here,
with the support threshold applied in SQL rather than after loading — measured
on the prod copy, that is the difference between 41 ms and 88 ms per pass,
because 88% of the chat's bigrams occur exactly once and can never be memes.
"""

from __future__ import annotations

from app.repositories.base_repo import BaseRepo

STATUS_CANDIDATE = "candidate"
STATUS_ACTIVE = "active"
STATUS_RETIRED = "retired"


class CollocationsRepo(BaseRepo):
    """Registry access plus the aggregates the analyzer scores over."""

    async def read_pair_counts(
        self, chat_id: int, *, min_joint_count: int
    ) -> list[tuple[str, str, int]]:
        """``(left, right, joint_count)`` for pairs at or above the threshold.

        The threshold is in the ``HAVING`` clause on purpose: the excluded tail
        is the overwhelming majority of rows and none of it can qualify. Nothing
        is deleted — an excluded pair keeps being learned and re-enters this
        result the moment its count crosses the bar.
        """
        rows = await self._fetch_all(
            """
            SELECT w1, w2, SUM(cnt)
            FROM transitions
            WHERE chat_id = ?
            GROUP BY w1, w2
            HAVING SUM(cnt) >= ?
            """,
            (chat_id, min_joint_count),
        )
        return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]

    async def read_token_marginals(
        self, chat_id: int
    ) -> tuple[dict[str, int], dict[str, int], int]:
        """``(left_totals, right_totals, corpus_total)`` over ALL pairs.

        Deliberately not derived from the thresholded pairs above: the filter
        decides which pairs get scored, never what the probabilities mean. Using
        filtered marginals would inflate every association by pretending the
        long tail does not exist.
        """
        left_rows = await self._fetch_all(
            "SELECT w1, SUM(cnt) FROM transitions WHERE chat_id = ? GROUP BY w1",
            (chat_id,),
        )
        right_rows = await self._fetch_all(
            "SELECT w2, SUM(cnt) FROM transitions WHERE chat_id = ? GROUP BY w2",
            (chat_id,),
        )
        left = {str(r[0]): int(r[1]) for r in left_rows}
        right = {str(r[0]): int(r[1]) for r in right_rows}
        return left, right, sum(left.values())

    async def replace_scored(
        self,
        chat_id: int,
        scored: list[tuple[str, str, int, float, float]],
    ) -> None:
        """Rewrite the chat's candidate/active entries from a scored ranking.

        ``scored`` is ``(left, right, joint_count, pmi, meme_score)``, already
        capped by the caller. Retired entries are left alone: retirement is a
        decision, and a pass that silently resurrected it would make the status
        column meaningless.
        """
        async with self._transaction() as db:
            cursor = await db.execute(
                "SELECT left_token, right_token FROM markov_collocations "
                "WHERE chat_id = ? AND status = ?",
                (chat_id, STATUS_RETIRED),
            )
            retired = {(str(r[0]), str(r[1])) for r in await cursor.fetchall()}
            await db.execute(
                "DELETE FROM markov_collocations WHERE chat_id = ? AND status != ?",
                (chat_id, STATUS_RETIRED),
            )
            if not scored:
                return
            await db.executemany(
                """
                INSERT INTO markov_collocations
                    (chat_id, left_token, right_token, joint_count, pmi,
                     meme_score, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(chat_id, left_token, right_token) DO UPDATE SET
                    joint_count = excluded.joint_count,
                    pmi = excluded.pmi,
                    meme_score = excluded.meme_score,
                    updated_at = excluded.updated_at
                """,
                [
                    (chat_id, left, right, joint, pmi, score, STATUS_ACTIVE)
                    for left, right, joint, pmi, score in scored
                    if (left, right) not in retired
                ],
            )

    async def get_active(self, chat_id: int) -> list[tuple[str, str]]:
        """Active pairs, highest scoring first — read once per reply."""
        rows = await self._fetch_all(
            """
            SELECT left_token, right_token
            FROM markov_collocations
            WHERE chat_id = ? AND status = ?
            ORDER BY meme_score DESC
            """,
            (chat_id, STATUS_ACTIVE),
        )
        return [(str(r[0]), str(r[1])) for r in rows]

    async def get_ranked(
        self, chat_id: int, limit: int
    ) -> list[tuple[str, str, int, float]]:
        """Top entries with their numbers — the manual rating round's input."""
        rows = await self._fetch_all(
            """
            SELECT left_token, right_token, joint_count, meme_score
            FROM markov_collocations
            WHERE chat_id = ? AND status != ?
            ORDER BY meme_score DESC
            LIMIT ?
            """,
            (chat_id, STATUS_RETIRED, limit),
        )
        return [(str(r[0]), str(r[1]), int(r[2]), float(r[3])) for r in rows]

    async def count_by_status(self, chat_id: int) -> dict[str, int]:
        """Registry size per status — the ``/stats`` view."""
        rows = await self._fetch_all(
            "SELECT status, COUNT(*) FROM markov_collocations "
            "WHERE chat_id = ? GROUP BY status",
            (chat_id,),
        )
        return {str(r[0]): int(r[1]) for r in rows}

    async def set_status(
        self, chat_id: int, left: str, right: str, status: str
    ) -> None:
        async with self._transaction() as db:
            await db.execute(
                "UPDATE markov_collocations SET status = ?, updated_at = datetime('now') "
                "WHERE chat_id = ? AND left_token = ? AND right_token = ?",
                (status, chat_id, left, right),
            )
