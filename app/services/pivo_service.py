from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from aiogram.types import User

from app.services.pivo_message_builder import build_pivo_message
from db import Database
from pivo import (
    PIVO_FALLBACK_MENTIONS,
    PivoMember,
    PivoSecurity,
    collect_pivo_mentions,
    display_name_from_user,
)

PIVO_DAILY_LIMIT_USER = 3
PIVO_DAILY_LIMIT_ADMIN = 5
PIVO_DEFAULT_EXPLICIT_MENTIONS_LIMIT = 10
PIVO_DEFAULT_SUBSCRIBER_FANOUT_LIMIT = 20


class PivoCallLimitError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PivoQuotaResult:
    allowed: bool
    used_count: int
    limit: int
    usage_day: str


class PivoService:
    """Бизнес-логика opt-in /pivo: подписка, отписка, сборка сообщения с упоминаниями."""

    def __init__(self, db: Database, security: PivoSecurity) -> None:
        self._db = db
        self._security = security
        self._explicit_mentions_limit = PIVO_DEFAULT_EXPLICIT_MENTIONS_LIMIT
        self._subscriber_fanout_limit = PIVO_DEFAULT_SUBSCRIBER_FANOUT_LIMIT

    def configure_call_limits(
        self,
        *,
        explicit_mentions_limit: int,
        subscriber_fanout_limit: int,
    ) -> None:
        self._explicit_mentions_limit = explicit_mentions_limit
        self._subscriber_fanout_limit = subscriber_fanout_limit

    async def subscribe(self, chat_id: int, user: User) -> None:
        await self._db.upsert_chat_member(
            chat_hash=self._security.hmac_value(chat_id),
            user_hash=self._security.hmac_value(user.id),
            encrypted_user_id=self._security.encrypt_value(user.id),
            encrypted_username=self._security.encrypt_value(user.username or ""),
            encrypted_display_name=self._security.encrypt_value(
                display_name_from_user(user)
            ),
        )

    async def unsubscribe(self, chat_id: int, user_id: int) -> None:
        await self._db.remove_chat_member(
            chat_hash=self._security.hmac_value(chat_id),
            user_hash=self._security.hmac_value(user_id),
        )

    async def consume_daily_call_quota(
        self,
        *,
        chat_id: int,
        user_id: int,
        is_admin_or_owner: bool,
        today: date | None = None,
    ) -> PivoQuotaResult:
        """Списывает одну суточную квоту /pivo для пользователя в чате."""
        usage_day = (today or datetime.now(UTC).date()).isoformat()
        limit = PIVO_DAILY_LIMIT_ADMIN if is_admin_or_owner else PIVO_DAILY_LIMIT_USER
        allowed, used_count = await self._db.consume_pivo_daily_call(
            chat_hash=self._security.hmac_value(chat_id),
            user_hash=self._security.hmac_value(user_id),
            usage_day=usage_day,
            limit=limit,
        )
        return PivoQuotaResult(
            allowed=allowed,
            used_count=used_count,
            limit=limit,
            usage_day=usage_day,
        )

    async def refund_daily_call_quota(
        self,
        *,
        chat_id: int,
        user_id: int,
        usage_day: str,
    ) -> None:
        """Возвращает списанную квоту /pivo после сбоя до доставки ответа."""
        await self._db.refund_pivo_daily_call(
            chat_hash=self._security.hmac_value(chat_id),
            user_hash=self._security.hmac_value(user_id),
            usage_day=usage_day,
        )

    async def build_call_message(
        self,
        chat_id: int,
        caller_user_id: int,
        *,
        planned_time: str | None = None,
        target: str | None = None,
        explicit_mentions: Sequence[str] = (),
    ) -> tuple[str, int]:
        """Возвращает готовое сообщение для /pivo и число упомянутых участников."""
        if explicit_mentions:
            mention_items = list(explicit_mentions)
            if len(mention_items) > self._explicit_mentions_limit:
                raise PivoCallLimitError(
                    "В /pivo можно указывать не больше "
                    f"{self._explicit_mentions_limit} явных упоминаний за раз."
                )
        else:
            chat_hash = self._security.hmac_value(chat_id)
            rows = await self._db.get_chat_members(chat_hash)
            members = [
                PivoMember(
                    encrypted_user_id=str(row["encrypted_user_id"]),
                    encrypted_username=str(row["encrypted_username"]),
                    encrypted_display_name=str(row["encrypted_display_name"]),
                )
                for row in rows
            ]
            mention_items = collect_pivo_mentions(
                members=members,
                caller_user_id=caller_user_id,
                security=self._security,
            )
            if len(mention_items) > self._subscriber_fanout_limit:
                raise PivoCallLimitError(
                    "В списке подписчиков /pivo слишком много людей: "
                    f"{len(mention_items)} из {self._subscriber_fanout_limit}."
                )
        mentions = " ".join(mention_items) if mention_items else PIVO_FALLBACK_MENTIONS
        text = build_pivo_message(
            mentions,
            planned_time=planned_time,
            target=target,
            has_explicit_mentions=bool(explicit_mentions),
        )
        return text, len(mention_items)
