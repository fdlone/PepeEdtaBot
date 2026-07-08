from __future__ import annotations

from collections.abc import Mapping

from app.repositories.base_repo import DecayableCountsRepo


class ChatEmojiStatsRepo(DecayableCountsRepo):
    """Per-chat emoji frequency for the M3 emoji channel.

    Keyed by raw ``chat_id`` to match the Markov model tables. Emojis are a
    per-chat aggregate (no author), so this stays within the existing privacy
    contour and is wiped together with the model in ``clear_chat``.
    """

    async def bump(self, chat_id: int, counts: Mapping[str, int]) -> None:
        """Add ``counts`` to the chat's emoji frequencies (no-op if empty)."""
        positive = {emoji: n for emoji, n in counts.items() if n > 0}
        if not positive:
            return
        await self._execute_many(
            """
            INSERT INTO chat_emoji_stats(chat_id, emoji, cnt)
            VALUES (?, ?, ?)
            ON CONFLICT(chat_id, emoji) DO UPDATE SET
                cnt = cnt + excluded.cnt,
                updated_at = datetime('now')
            """,
            [(chat_id, emoji, n) for emoji, n in positive.items()],
        )

    async def get_stats(self, chat_id: int) -> dict[str, int]:
        """Return {emoji: count} for a chat (empty if none learned)."""
        rows = await self._fetch_all(
            "SELECT emoji, cnt FROM chat_emoji_stats WHERE chat_id = ?",
            (chat_id,),
        )
        return {str(row[0]): int(row[1] or 0) for row in rows}

    async def decay_stale(self, cutoff_iso: str) -> int:
        """Halve counts of rows not bumped since ``cutoff_iso`` so dead memes fade."""
        return await self._decay_stale("chat_emoji_stats", cutoff_iso)
