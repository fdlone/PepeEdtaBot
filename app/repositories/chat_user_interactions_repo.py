from __future__ import annotations

from app.repositories.base_repo import DecayableCountsRepo


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

    async def get_stats(self, chat_id: int, threshold: int) -> tuple[int, int, int]:
        """Aggregate for the chat: ``(people, max_count, at_or_above)``.

        Numbers only — the caller must be able to answer "is the regulars
        threshold reachable here?" without anything that points at a person.
        The counters are anonymous by construction (HMAC), and a diagnostic
        must not undo that.
        """
        row = await self._fetch_one(
            """
            SELECT COUNT(*), COALESCE(MAX(cnt), 0),
                   COALESCE(SUM(CASE WHEN cnt >= ? THEN 1 ELSE 0 END), 0)
            FROM chat_user_interactions
            WHERE chat_id = ?
            """,
            (threshold, chat_id),
        )
        if row is None:
            return 0, 0, 0
        return int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)

    async def decay_stale(self, cutoff_iso: str) -> int:
        """Halve counts of users quiet since ``cutoff_iso`` so regulars fade."""
        return await self._decay_stale("chat_user_interactions", cutoff_iso)
