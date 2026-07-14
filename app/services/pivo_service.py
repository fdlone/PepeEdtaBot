from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from aiogram.types import User

from app.domain.pivo import (
    PIVO_FALLBACK_MENTIONS,
    PivoMember,
    PivoSecurity,
    collect_pivo_mentions,
    display_name_from_user,
)
from app.infrastructure.database import Database
from app.services.pivo_message_builder import (
    PivoMessageGenerator,
    build_pivo_message_context,
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
        # Дроссель refresh_member: (chat_hash, user_hash) → день последней
        # записи профиля. Ключи — те же хеши, что и в БД, сырых id в памяти нет.
        self._refreshed_on: dict[tuple[str, str], str] = {}

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

    async def refresh_member(
        self, chat_id: int, user: User, *, today: date | None = None
    ) -> None:
        """Обновляет @username и имя подписчика по свежему сообщению из чата.

        Без этого в /pivo уходит снимок, снятый при /pivo_on: сменивший ник
        участник получает в упоминании мёртвый «@старый_ник» — не подсвеченный
        и без уведомления. Не подписанных не трогает (UPDATE не найдёт строку).

        Вызывается на каждом сообщении, поэтому запись дросселируется: не чаще
        одного раза в сутки на пользователя в чате. Кэш живёт в памяти — после
        рестарта первое сообщение каждого участника снова обновит профиль.
        """
        chat_hash = self._security.hmac_value(chat_id)
        user_hash = self._security.hmac_value(user.id)
        usage_day = (today or datetime.now(UTC).date()).isoformat()
        if self._refreshed_on.get((chat_hash, user_hash)) == usage_day:
            return
        self._refreshed_on[(chat_hash, user_hash)] = usage_day
        await self._db.refresh_chat_member(
            chat_hash=chat_hash,
            user_hash=user_hash,
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

    async def clear_chat_data(self, chat_id: int) -> None:
        """Удаляет все /pivo-данные чата (подписки, квоты, анти-повтор).

        Вызывается из /clear: очистка чата должна затрагивать и opt-in
        подписки, а не только модель генерации.
        """
        await self._db.clear_pivo_chat_data(self._security.hmac_value(chat_id))

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
        recent_pool_window: int = 0,
        temporal_flavor_chance: float = 0.0,
        now: datetime | None = None,
    ) -> tuple[str, int, dict[str, int]]:
        """Возвращает сообщение /pivo, число упоминаний и выбранные индексы пулов.

        Метод не имеет побочных эффектов в БД: анти-повтор фиксируется отдельным
        вызовом ``record_pool_usage`` только после успешной отправки, чтобы
        отклонённые по квоте или недоставленные вызовы не крутили пулы.
        """
        chat_hash = self._security.hmac_value(chat_id)
        if explicit_mentions:
            mention_items = list(explicit_mentions)
            if len(mention_items) > self._explicit_mentions_limit:
                raise PivoCallLimitError(
                    "В /pivo можно указывать не больше "
                    f"{self._explicit_mentions_limit} явных упоминаний за раз."
                )
        else:
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
        context = build_pivo_message_context(
            mentions,
            planned_time=planned_time,
            target=target,
            has_explicit_mentions=bool(explicit_mentions),
        )
        recent_indices = (
            await self._db.get_pivo_pool_usage(chat_hash)
            if recent_pool_window > 0
            else {}
        )
        result = PivoMessageGenerator().build(
            context,
            recent_indices=recent_indices,
            now=now,
            temporal_flavor_chance=temporal_flavor_chance,
        )
        return result.text, len(mention_items), result.picks

    async def record_pool_usage(
        self,
        chat_id: int,
        picks: Mapping[str, int],
        *,
        recent_pool_window: int,
    ) -> None:
        """Фиксирует анти-повтор шаблонов после успешной отправки /pivo."""
        if recent_pool_window <= 0 or not picks:
            return
        await self._db.record_pivo_pool_usage(
            self._security.hmac_value(chat_id),
            picks,
            keep=recent_pool_window,
        )
