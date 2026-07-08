from __future__ import annotations

from collections.abc import Mapping

from app.repositories.base_repo import BaseRepo


def _parse_indices(raw: str) -> tuple[int, ...]:
    """Parse the CSV ``recent_indices`` column, skipping malformed entries."""
    indices: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            indices.append(int(chunk))
        except ValueError:
            continue
    return tuple(indices)


class PivoPoolUsageRepo(BaseRepo):
    """Per-chat memory of recently used /pivo template indices (anti-repeat)."""

    async def get_recent(self, chat_hash: str) -> dict[str, tuple[int, ...]]:
        """Return {pool_name: recent indices (most-recent last)} for a chat."""
        rows = await self._fetch_all(
            """
            SELECT pool_name, recent_indices
            FROM pivo_pool_usage
            WHERE chat_hash = ?
            """,
            (chat_hash,),
        )
        return {str(row[0]): _parse_indices(str(row[1])) for row in rows}

    async def delete_chat(self, chat_hash: str) -> None:
        """Удаляет анти-повтор историю чата (используется /clear)."""
        await self._execute(
            "DELETE FROM pivo_pool_usage WHERE chat_hash = ?",
            (chat_hash,),
        )

    async def record(
        self,
        chat_hash: str,
        picks: Mapping[str, int],
        *,
        keep: int,
    ) -> None:
        """Append each pick to its pool history, trimming to the last ``keep``.

        A ``keep`` of 0 or fewer disables anti-repeat; nothing is stored.
        """
        if keep <= 0 or not picks:
            return
        async with self._transaction() as db:
            for pool_name, index in picks.items():
                cursor = await db.execute(
                    """
                    SELECT recent_indices
                    FROM pivo_pool_usage
                    WHERE chat_hash = ? AND pool_name = ?
                    """,
                    (chat_hash, pool_name),
                )
                row = await cursor.fetchone()
                history = list(_parse_indices(str(row[0]))) if row is not None else []
                history.append(index)
                trimmed = history[-keep:]
                serialized = ",".join(str(i) for i in trimmed)
                await db.execute(
                    """
                    INSERT INTO pivo_pool_usage(chat_hash, pool_name, recent_indices)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chat_hash, pool_name) DO UPDATE SET
                        recent_indices = excluded.recent_indices,
                        updated_at = datetime('now')
                    """,
                    (chat_hash, pool_name, serialized),
                )
