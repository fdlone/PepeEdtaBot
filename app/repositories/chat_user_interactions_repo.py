from __future__ import annotations

from app.repositories.base_repo import DecayableCountsRepo


def bucket_edges(threshold: int) -> tuple[int, ...]:
    """Lower bounds of the distribution buckets for a live threshold.

    Quarters of the threshold plus the threshold itself, so summing the
    buckets from the right answers "how many people would a threshold of
    75%/50%/25% add" without a second query. Degenerate thresholds collapse
    the ladder instead of producing empty or overlapping ranges.
    """
    if threshold <= 1:
        return (1,)
    quarters = (max(1, threshold * share // 4) for share in (1, 2, 3))
    return tuple(sorted({1, *quarters, threshold}))


class ChatUserInteractionsRepo(DecayableCountsRepo):
    """Per-chat count of answered bot mentions per anonymized user (L2 quirks).

    ``user_hash`` is HMAC-SHA256 under ``PIVO_HMAC_SECRET`` — the same
    anonymization as ``/pivo`` subscriptions: no reversible identity, no
    payload, only a counter. Keyed by raw ``chat_id`` to match the model
    tables so the rows are wiped together with them in ``clear_chat``.
    """

    async def bump(self, chat_id: int, user_hash: str) -> None:
        """Count one answered mention for the user (upsert ``cnt + 1``)."""
        await self._execute(
            """
            INSERT INTO chat_user_interactions(chat_id, user_hash, cnt)
            VALUES (?, ?, 1)
            ON CONFLICT(chat_id, user_hash) DO UPDATE SET
                cnt = cnt + 1,
                updated_at = datetime('now')
            """,
            (chat_id, user_hash),
        )

    async def get_count(self, chat_id: int, user_hash: str) -> int:
        """Interaction count for the user in the chat (0 when absent)."""
        row = await self._fetch_one(
            "SELECT cnt FROM chat_user_interactions"
            " WHERE chat_id = ? AND user_hash = ?",
            (chat_id, user_hash),
        )
        return int(row[0] or 0) if row else 0

    async def get_stats(
        self, chat_id: int, threshold: int
    ) -> tuple[int, int, int, tuple[tuple[int, int, int], ...]]:
        """Aggregate for the chat: ``(people, max_count, at_or_above, buckets)``.

        Numbers only — the caller must be able to answer "is the regulars
        threshold reachable here?" without anything that points at a person.
        The counters are anonymous by construction (HMAC), and a diagnostic
        must not undo that.

        ``buckets`` is ``(lower, upper_exclusive_or_zero, people)`` ascending,
        with the edges placed *relative to the live threshold*
        (:func:`bucket_edges`): the question the distribution exists to answer
        is "how many people would a lower threshold add", and absolute ranges
        would answer a different one. Grouping rather than per-person values is
        the privacy line — a list of counters is a profile of the chat.
        """
        edges = bucket_edges(threshold)
        # One cumulative SUM per edge: "people at or above X" is literally the
        # applied question, and the per-bucket counts are its differences.
        # ``at_or_above`` keeps its own SUM instead of being read off an edge —
        # the threshold may sit outside the bucket ladder (0, say), and a
        # clever lookup would answer the wrong question there.
        sums = ", ".join(
            "COALESCE(SUM(CASE WHEN cnt >= ? THEN 1 ELSE 0 END), 0)"
            for _ in (threshold, *edges)
        )
        row = await self._fetch_one(
            f"""
            SELECT COUNT(*), COALESCE(MAX(cnt), 0), {sums}
            FROM chat_user_interactions
            WHERE chat_id = ?
            """,
            (threshold, *edges, chat_id),
        )
        if row is None:
            return 0, 0, 0, ()
        cumulative = [int(row[3 + index] or 0) for index in range(len(edges))]
        buckets = tuple(
            (
                lower,
                edges[index + 1] if index + 1 < len(edges) else 0,
                cumulative[index]
                - (cumulative[index + 1] if index + 1 < len(cumulative) else 0),
            )
            for index, lower in enumerate(edges)
        )
        return int(row[0] or 0), int(row[1] or 0), int(row[2] or 0), buckets

    async def decay_stale(self, cutoff_iso: str) -> int:
        """Halve counts of users quiet since ``cutoff_iso`` so regulars fade."""
        return await self._decay_stale("chat_user_interactions", cutoff_iso)
