from __future__ import annotations

from app.core.text import sanitize_text
from app.repositories.base_repo import BaseRepo


class MessagesRepo(BaseRepo):
    """Доступ к таблице messages: чтение и проверка наличия."""

    async def exists(self, chat_id: int, text: str) -> bool:
        row = await self._fetch_one(
            "SELECT 1 FROM messages WHERE chat_id = ? AND normalized_text = ? LIMIT 1",
            (chat_id, sanitize_text(text)),
        )
        return row is not None

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
