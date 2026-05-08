from __future__ import annotations

from aiogram.types import User

from db import Database
from pivo import (
    PIVO_FALLBACK_MENTIONS,
    PivoMember,
    PivoSecurity,
    collect_pivo_mentions,
    display_name_from_user,
    get_random_pivo_message,
)


class PivoService:
    """Бизнес-логика opt-in /pivo: подписка, отписка, сборка сообщения с упоминаниями."""

    def __init__(self, db: Database, security: PivoSecurity) -> None:
        self._db = db
        self._security = security

    async def subscribe(self, chat_id: int, user: User) -> None:
        await self._db.upsert_pivo_member(
            chat_hash=self._security.hmac_value(chat_id),
            user_hash=self._security.hmac_value(user.id),
            encrypted_user_id=self._security.encrypt_value(user.id),
            encrypted_username=self._security.encrypt_value(user.username or ""),
            encrypted_display_name=self._security.encrypt_value(
                display_name_from_user(user)
            ),
            is_bot=user.is_bot,
        )

    async def unsubscribe(self, chat_id: int, user_id: int) -> None:
        await self._db.remove_pivo_member(
            chat_hash=self._security.hmac_value(chat_id),
            user_hash=self._security.hmac_value(user_id),
        )

    async def build_call_message(
        self, chat_id: int, caller_user_id: int
    ) -> tuple[str, int]:
        """Возвращает готовое сообщение для /pivo и число упомянутых участников."""
        chat_hash = self._security.hmac_value(chat_id)
        rows = await self._db.get_pivo_members(chat_hash)
        members = [
            PivoMember(
                encrypted_user_id=str(row["encrypted_user_id"]),
                encrypted_username=str(row["encrypted_username"]),
                encrypted_display_name=str(row["encrypted_display_name"]),
                is_bot=bool(row["is_bot"]),
            )
            for row in rows
        ]
        mention_items = collect_pivo_mentions(
            members=members,
            caller_user_id=caller_user_id,
            security=self._security,
        )
        mentions = " ".join(mention_items) if mention_items else PIVO_FALLBACK_MENTIONS
        return get_random_pivo_message(mentions), len(mention_items)
