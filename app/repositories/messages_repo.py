from __future__ import annotations

from app.repositories.base_repo import BaseRepo


class MessagesRepo(BaseRepo):
    """Доступ к таблице messages: чтение и проверка наличия."""

    async def get_recent_normalized(self, chat_id: int, limit: int) -> list[str]:
        rows = await self._fetch_all(
            """
            SELECT normalized_text
            FROM messages
            WHERE chat_id = ? AND normalized_text != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        return [str(row[0]) for row in reversed(rows)]
