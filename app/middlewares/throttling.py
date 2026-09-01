from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject


class ThrottlingMiddleware(BaseMiddleware):
    """Per-user per-command cooldown.

    By default throttled updates are silently dropped. Commands listed in
    ``notify_on_throttle`` instead get a short reply telling the user how
    long is left — useful for explicit user-driven commands like ``/clear``,
    where silence reads as "the bot is broken".

    The notify reply itself is rate-limited per key (``notify_cooldown_sec``)
    so a user hammering a throttled command cannot turn every attempt into a
    bot reply (audit N6): after one notification further attempts inside the
    window are dropped silently.
    """

    def __init__(
        self,
        limits: dict[str, float],
        notify_on_throttle: set[str] | None = None,
        *,
        state_ttl_sec: int = 21600,
        state_max_keys: int = 4096,
        notify_cooldown_sec: float = 30.0,
        bot_username: str | None = None,
    ) -> None:
        # limits: command name (without /) -> cooldown in seconds
        self._limits = limits
        # Own username, lowercased: "/clear@OtherBot" is another bot's traffic
        # and must not burn this bot's cooldown. Telegram usernames are
        # case-insensitive. When unset (tests), any @mention is treated as ours.
        self._bot_username = (bot_username or "").lower() or None
        self._notify_on_throttle = notify_on_throttle or set()
        self._state_ttl_sec = float(state_ttl_sec)
        self._state_max_keys = state_max_keys
        self._notify_cooldown_sec = notify_cooldown_sec
        # (chat_id, user_id, command) -> last allowed timestamp
        self._last_used: dict[tuple[int, int, str], float] = {}
        # (chat_id, user_id, command) -> last throttle-notification timestamp
        self._last_notified: dict[tuple[int, int, str], float] = {}
        self._cleanup_tick = 0

    def _prune_state(self, now: float) -> None:
        cutoff = now - self._state_ttl_sec
        stale_keys = [
            key for key, last_used in self._last_used.items() if last_used < cutoff
        ]
        for key in stale_keys:
            self._last_used.pop(key, None)

        overflow = len(self._last_used) - self._state_max_keys
        if overflow > 0:
            oldest_keys = sorted(
                self._last_used.items(),
                key=lambda item: item[1],
            )[:overflow]
            for key, _ in oldest_keys:
                self._last_used.pop(key, None)

        stale_notified = [
            key
            for key, notified_at in self._last_notified.items()
            if notified_at < cutoff or key not in self._last_used
        ]
        for key in stale_notified:
            self._last_notified.pop(key, None)

        self._cleanup_tick = 0

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message):
            return await handler(event, data)

        # `text or caption` — ровно то, что читает фильтр `Command` в aiogram
        # (filters/command.py). Команда, отправленная подписью к медиа, имеет
        # text=None и caption="/stats": фильтр её принимает и хендлер
        # отрабатывает целиком. Пока здесь стоял один `event.text`, такая
        # команда проходила мимо окна — то есть кулдаун обходился для всех
        # команд сразу, а спека command-rate-limits требует, чтобы команда без
        # ограничения была явным исключением, а не следствием расхождения двух
        # разборов одного и того же сообщения.
        text = event.text or event.caption or ""
        if not text.startswith("/") or event.from_user is None:
            return await handler(event, data)

        parts = text.split(maxsplit=1)
        command, _, mention = parts[0].lstrip("/").partition("@")
        command = command.lower()
        if (
            mention
            and self._bot_username is not None
            and mention.lower() != self._bot_username
        ):
            # Addressed to a different bot: not our traffic, no cooldown.
            return await handler(event, data)
        limit = self._limits.get(command)
        if limit is None:
            return await handler(event, data)

        throttle_key = command
        if command == "clear" and len(parts) > 1 and parts[1].strip().lower() == "confirm":
            throttle_key = "clear:confirm"

        key = (event.chat.id, event.from_user.id, throttle_key)
        now = time.monotonic()
        self._cleanup_tick += 1
        if self._cleanup_tick >= 64 or len(self._last_used) > self._state_max_keys:
            self._prune_state(now)
        last_used = self._last_used.get(key)
        if last_used is not None and (now - last_used) < limit:
            if command in self._notify_on_throttle:
                last_notified = self._last_notified.get(key)
                if (
                    last_notified is None
                    or (now - last_notified) >= self._notify_cooldown_sec
                ):
                    self._last_notified[key] = now
                    remaining = max(1, math.ceil(limit - (now - last_used)))
                    await event.reply(
                        f"Слишком часто. Подождите ~{remaining} сек."
                    )
            return None  # throttled

        # Окно занимается ДО вызова хендлера. Пока стамп стоял после `await`,
        # момент фиксации был наблюдаем через конкурентность: aiogram заводит
        # задачу на каждый апдейт, `getUpdates` отдаёт их пачками, и пять
        # одинаковых команд успевали пройти проверку до первой записи —
        # хендлер отрабатывал пять раз при окне в одно.
        #
        # Отдельного примитива синхронизации здесь нет и не нужно: участок от
        # чтения `last_used` до этой строки не содержит `await`, а внутри
        # одного event loop такой участок не прерывается. Примитив
        # понадобится вместе со вторым инстансом — то есть вместе со всем
        # остальным из «осознанно отложено» (docs/OPEN.md).
        previous_used = last_used
        self._last_used[key] = now
        if len(self._last_used) > self._state_max_keys:
            self._prune_state(now)
        try:
            return await handler(event, data)
        except Exception:
            # Прежнее свойство сохраняется: падение хендлера (например,
            # SQLITE_BUSY на /clear) не запирает участника на всё окно.
            # Восстанавливается прежнее значение, а не удаляется ключ —
            # потому что это буквально «вернуть как было», а не ради иного
            # поведения: разницы между двумя формами здесь нет и быть не
            # может. Вызов, дошедший до хендлера, уже прошёл проверку окна,
            # значит прежнее значение либо отсутствует, либо лежит вне окна.
            # (Прежняя редакция этого комментария утверждала, что удаление
            # обнулило бы «накопленное окно» — такого окна в этой точке
            # существовать не может.)
            # Только если стамп всё ещё наш. Медленный хендлер мог упасть уже
            # после того, как окно истекло и следующий вызов занял его
            # законно, — его занятие отменять нельзя. Та же логика, что у
            # отката резервации ответа в RuntimeState.release_reply_slot.
            if self._last_used.get(key) != now:
                raise
            if previous_used is None:
                self._last_used.pop(key, None)
            else:
                self._last_used[key] = previous_used
            raise
