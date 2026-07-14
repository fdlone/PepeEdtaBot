from __future__ import annotations

from app.repositories.base_repo import BaseRepo


class ChatMembersRepo(BaseRepo):
    """Доступ к таблице chat_members — каноническое хранилище участников чата.

    Сейчас единственный потребитель — `/pivo` (подписка / упоминания), но
    таблица намеренно не привязана к конкретной фиче: будущие функции,
    которым нужно персистентное состояние участника, ходят сюда же.
    """

    async def upsert(
        self,
        *,
        chat_hash: str,
        user_hash: str,
        encrypted_user_id: str,
        encrypted_username: str,
        encrypted_display_name: str,
    ) -> None:
        await self._execute(
            """
            INSERT INTO chat_members(
                chat_hash,
                user_hash,
                encrypted_user_id,
                encrypted_username,
                encrypted_display_name
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chat_hash, user_hash)
            DO UPDATE SET
                encrypted_user_id = excluded.encrypted_user_id,
                encrypted_username = excluded.encrypted_username,
                encrypted_display_name = excluded.encrypted_display_name,
                updated_at = datetime('now')
            """,
            (
                chat_hash,
                user_hash,
                encrypted_user_id,
                encrypted_username,
                encrypted_display_name,
            ),
        )

    async def refresh_profile(
        self,
        *,
        chat_hash: str,
        user_hash: str,
        encrypted_username: str,
        encrypted_display_name: str,
    ) -> None:
        """Обновляет профиль уже подписанного участника (без вставки новых строк).

        Username/display name — снимок на момент подписки, а Telegram позволяет
        менять и то, и другое: устаревший «@ник» в /pivo превращается в мёртвый
        текст, который никого не тегает. UPDATE без ON CONFLICT намеренный —
        участники, не нажавшие /pivo_on, в таблицу не попадают.
        """
        await self._execute(
            """
            UPDATE chat_members
            SET encrypted_username = ?,
                encrypted_display_name = ?,
                updated_at = datetime('now')
            WHERE chat_hash = ? AND user_hash = ?
            """,
            (encrypted_username, encrypted_display_name, chat_hash, user_hash),
        )

    async def remove_chat(self, chat_hash: str) -> None:
        """Удаляет всех участников чата (используется /clear)."""
        await self._execute(
            "DELETE FROM chat_members WHERE chat_hash = ?",
            (chat_hash,),
        )

    async def remove(self, chat_hash: str, user_hash: str) -> None:
        await self._execute(
            "DELETE FROM chat_members WHERE chat_hash = ? AND user_hash = ?",
            (chat_hash, user_hash),
        )

    async def list_members(self, chat_hash: str) -> list[dict[str, object]]:
        rows = await self._fetch_all(
            """
            SELECT encrypted_user_id, encrypted_username, encrypted_display_name
            FROM chat_members
            WHERE chat_hash = ?
            ORDER BY created_at, user_hash
            """,
            (chat_hash,),
        )
        return [
            {
                "encrypted_user_id": str(row[0]),
                "encrypted_username": str(row[1]),
                "encrypted_display_name": str(row[2]),
            }
            for row in rows
        ]
