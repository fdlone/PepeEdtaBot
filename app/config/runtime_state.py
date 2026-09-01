from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

from app.config.registry import RUNTIME_FIELDS
from app.config.settings import RuntimeTunables, Settings
from app.core.mood import ChatMoodState, MoodConfig


@dataclass(slots=True)
class _MaintenanceCounter:
    """Счётчик обращений, по которому запускается периодическая уборка.

    Существует ради того, чтобы быть изменяемым объектом. Представление чата
    (см. ``RuntimeState.effective``) — поверхностная копия, и любое скалярное
    поле в ней своё: инкремент на копии-однодневке до базового состояния не
    доходит. Все остальные накопители — словари, поэтому общие; этот счётчик
    был единственным изменяемым скаляром и единственным исключением из
    инварианта «живое состояние у всех представлений одно».
    """

    value: int = 0


@dataclass(frozen=True, slots=True)
class ReplySlot:
    """Талон занятого бюджета ответа: чем откатывать, если ответ не состоялся.

    Хранит момент резервации и предыдущее значение кулдауна, потому что откат
    обязан снять **свою** метку. Между резервацией и откатом лежит генерация,
    за которую другой апдейт мог занять слот законно, и его занятие отменять
    нельзя.
    """

    chat_id: int
    now: float
    previous_last_ts: float | None
    counted: bool


@dataclass(slots=True)
class RuntimeState(RuntimeTunables):
    # Все runtime-mutable поля объявлены один раз в RuntimeTunables
    # (settings.py); здесь остаётся только живое состояние процесса.
    runtime_state_ttl_sec: int
    runtime_state_max_chats: int
    # env-only, не ручка /set: часовой пояс — свойство деплоя (O12). Двигает
    # человеческие временны́е ветки; служебные сутки остаются на UTC. Дефолт
    # повторяет дефолт CHAT_TIMEZONE; ZoneInfo иммутабелен, общий дефолт безопасен.
    chat_timezone: ZoneInfo = field(default_factory=lambda: ZoneInfo("UTC"))
    last_reply_ts: dict[int, float] = field(default_factory=dict)
    learned_messages: dict[int, int] = field(default_factory=dict)
    recent_short_replies: dict[int, deque[str]] = field(default_factory=dict)
    recent_replies: dict[int, deque[str]] = field(default_factory=dict)
    recent_fallbacks: dict[int, deque[str]] = field(default_factory=dict)
    chat_mood: dict[int, ChatMoodState] = field(default_factory=dict)
    # M2: timestamps (monotonic seconds) of recent bot replies per chat, used for
    # the per-hour reply cap. Trimmed to a one-hour window on each append.
    recent_reply_times: dict[int, deque[float]] = field(default_factory=dict)
    # Anti-flood gate for mention-triggered replies: (chat_id, user_id) ->
    # monotonic timestamp of the last reply this user got by addressing the bot.
    last_mention_reply_ts: dict[tuple[int, int], float] = field(default_factory=dict)
    # L3: (ISO day, fired count) per chat for the combined daily budget of
    # rare events + false starts.
    rare_events_today: dict[int, tuple[str, int]] = field(default_factory=dict)
    # L2: (chat_id, user_id) -> ISO UTC day of the last vocative quirk. The
    # once-a-day-per-user cap is fixed in code, not a knob — rarity is the
    # point. Raw user_id stays in memory only (never persisted; the DB side
    # is keyed by HMAC).
    last_user_quirk_day: dict[tuple[int, int], str] = field(default_factory=dict)
    _last_chat_activity: dict[int, float] = field(default_factory=dict)
    _cleanup: _MaintenanceCounter = field(default_factory=_MaintenanceCounter)
    # Per-chat setting overrides on top of the global values above:
    # chat_id -> {field name: value}. Only settings live here; the live
    # per-chat state (the dicts above) is never split — see ``effective``.
    chat_overrides: dict[int, dict[str, Any]] = field(default_factory=dict)

    def effective(self, chat_id: int) -> RuntimeState:
        """Return the state as this chat sees it.

        Without overrides this returns ``self`` — the very same object, not a
        copy. That is what keeps a chat that never used ``/set`` behaving
        byte-identically to before overlays existed (and costs nothing on the
        hot path).

        With overrides it is a shallow copy: every per-chat dict above is
        shared by reference, so anti-repeat history, mood and counters keep
        accumulating on the one true state no matter which view wrote them.
        Only the scalar settings differ.
        """
        overrides = self.chat_overrides.get(chat_id)
        if not overrides:
            return self
        view = copy.copy(self)
        for name, value in overrides.items():
            setattr(view, name, value)
        return view

    def set_override(self, chat_id: int, name: str, value: Any) -> None:
        self.chat_overrides.setdefault(chat_id, {})[name] = value

    def clear_override(self, chat_id: int, name: str) -> bool:
        """Drop one override; returns whether there was anything to drop."""
        overrides = self.chat_overrides.get(chat_id)
        if not overrides or name not in overrides:
            return False
        del overrides[name]
        if not overrides:
            self.chat_overrides.pop(chat_id, None)
        return True

    def mood_config(self) -> MoodConfig:
        return MoodConfig(
            ewma_alpha=self.mood_ewma_alpha,
            lively_rate_per_min=self.mood_lively_rate_per_min,
            sleepy_rate_per_min=self.mood_sleepy_rate_per_min,
            heated_intensity=self.mood_heated_intensity,
            max_rate_per_min=self.mood_max_rate_per_min,
            mention_heated_share=self.mood_mention_heated_share,
        )

    def note_reply_sent(
        self, chat_id: int, now: float, *, unprompted: bool = True
    ) -> None:
        """Record that the bot replied in ``chat_id`` at ``now`` (monotonic sec).

        Always updates ``last_reply_ts`` (cooldown + burst rhythm apply to every
        reply). Only ``unprompted`` replies are appended to the rolling per-hour
        history used by the reply cap: mention answers are always sent and must
        never count against the gate (see REPLY_MAX_PER_HOUR). Entries older than
        one hour are dropped so the deque stays bounded by the cap itself.
        """
        self.last_reply_ts[chat_id] = now
        if not unprompted:
            return
        history = self.recent_reply_times.get(chat_id)
        if history is None:
            history = deque()
            self.recent_reply_times[chat_id] = history
        history.append(now)
        cutoff = now - 3600.0
        while history and history[0] < cutoff:
            history.popleft()

    def reserve_reply_slot(
        self, chat_id: int, now: float, *, unprompted: bool = True
    ) -> ReplySlot:
        """Занять бюджет ответа немедленно, до подготовки самого ответа.

        Отличается от ``note_reply_sent`` только моментом вызова и тем, что
        возвращает талон для отката. Смысл переноса: между проверкой кулдауна
        и записью о факте ответа лежит вся генерация, а апдейты
        обрабатываются конкурентно (aiogram заводит задачу на каждый, а
        ``getUpdates`` отдаёт их пачками). Пока запись стояла после отправки,
        пачка сообщений успевала пройти все проверки до первой записи — и
        кулдаун с часовым капом не давали того, что обещают, ровно в
        burst-режиме, ради которого заведены.

        Примитива синхронизации здесь нет намеренно: участок от проверки до
        этой записи не содержит ``await``, а внутри одного event loop такой
        участок не прерывается. Он понадобится вместе со вторым инстансом —
        то есть вместе со всем остальным из «осознанно отложено».
        """
        previous_last_ts = self.last_reply_ts.get(chat_id)
        self.note_reply_sent(chat_id, now, unprompted=unprompted)
        return ReplySlot(
            chat_id=chat_id,
            now=now,
            previous_last_ts=previous_last_ts,
            counted=unprompted,
        )

    def release_reply_slot(self, slot: ReplySlot) -> None:
        """Вернуть бюджет: ответ не состоялся.

        Возврат разрешает ответ на **следующее** сообщение и не переигрывает
        текущее — иначе неудачная генерация превращалась бы в цикл ретраев на
        горячем пути.

        Снимается именно своя метка, а не последняя: между резервацией и
        откатом лежат ``await``, за которые другой апдейт мог занять слот, и
        его резервация остаётся в силе. По той же причине ``last_reply_ts``
        восстанавливается только если он всё ещё наш.
        """
        if self.last_reply_ts.get(slot.chat_id) == slot.now:
            if slot.previous_last_ts is None:
                self.last_reply_ts.pop(slot.chat_id, None)
            else:
                self.last_reply_ts[slot.chat_id] = slot.previous_last_ts
        if not slot.counted:
            return
        history = self.recent_reply_times.get(slot.chat_id)
        if history is None:
            return
        try:
            history.remove(slot.now)
        except ValueError:
            # Метки уже нет — её вытеснил часовой срез. Возвращать нечего.
            pass
        if not history:
            self.recent_reply_times.pop(slot.chat_id, None)

    def can_fire_rare_event(self, chat_id: int, today_iso: str) -> bool:
        """True while the chat's combined daily event budget is not exhausted."""
        day, count = self.rare_events_today.get(chat_id, (today_iso, 0))
        if day != today_iso:
            return self.rare_event_daily_cap > 0
        return count < self.rare_event_daily_cap

    def note_rare_event(self, chat_id: int, today_iso: str) -> None:
        """Count a fired rare event; the counter resets on day change."""
        day, count = self.rare_events_today.get(chat_id, (today_iso, 0))
        if day != today_iso:
            day, count = today_iso, 0
        self.rare_events_today[chat_id] = (day, count + 1)

    def note_mention_reply(self, chat_id: int, user_id: int, now: float) -> None:
        self.last_mention_reply_ts[(chat_id, user_id)] = now

    def can_fire_user_quirk(
        self, chat_id: int, user_id: int, today_iso: str
    ) -> bool:
        """True while the user has not received a quirk today (UTC day)."""
        return self.last_user_quirk_day.get((chat_id, user_id)) != today_iso

    def note_user_quirk(self, chat_id: int, user_id: int, today_iso: str) -> None:
        self.last_user_quirk_day[(chat_id, user_id)] = today_iso

    def note_chat_activity(self, chat_id: int, now: float) -> None:
        self._last_chat_activity[chat_id] = now
        self._cleanup.value += 1
        if (
            self._cleanup.value >= 64
            or len(self._last_chat_activity) > self.runtime_state_max_chats
        ):
            self.prune_inactive(now)

    def forget_chat(self, chat_id: int) -> None:
        self.last_reply_ts.pop(chat_id, None)
        self.learned_messages.pop(chat_id, None)
        self.recent_short_replies.pop(chat_id, None)
        self.recent_replies.pop(chat_id, None)
        self.recent_fallbacks.pop(chat_id, None)
        self.chat_mood.pop(chat_id, None)
        self.recent_reply_times.pop(chat_id, None)
        self.rare_events_today.pop(chat_id, None)
        for key in [k for k in self.last_mention_reply_ts if k[0] == chat_id]:
            self.last_mention_reply_ts.pop(key, None)
        for key in [k for k in self.last_user_quirk_day if k[0] == chat_id]:
            self.last_user_quirk_day.pop(key, None)
        self._last_chat_activity.pop(chat_id, None)
        # Overrides are per-chat state like everything else here: a chat the
        # bot forgot must not come back carrying its old tuning.
        self.chat_overrides.pop(chat_id, None)

    def prune_inactive(self, now: float) -> None:
        cutoff = now - self.runtime_state_ttl_sec
        stale_chat_ids = [
            chat_id
            for chat_id, last_seen in self._last_chat_activity.items()
            if last_seen < cutoff
        ]
        for chat_id in stale_chat_ids:
            self.forget_chat(chat_id)

        overflow = len(self._last_chat_activity) - self.runtime_state_max_chats
        if overflow > 0:
            oldest_chat_ids = sorted(
                self._last_chat_activity.items(),
                key=lambda item: item[1],
            )[:overflow]
            for chat_id, _ in oldest_chat_ids:
                self.forget_chat(chat_id)

        self._cleanup.value = 0


def runtime_state_from_settings(settings: Settings) -> RuntimeState:
    """Build a fresh RuntimeState from Settings via the config registry.

    Iterating over RUNTIME_FIELDS guarantees that adding a new
    runtime-mutable field requires editing only ``config_registry.py``.
    """
    return RuntimeState(
        **{spec.name: getattr(settings, spec.name) for spec in RUNTIME_FIELDS},
        runtime_state_ttl_sec=settings.runtime_state_ttl_sec,
        runtime_state_max_chats=settings.runtime_state_max_chats,
        chat_timezone=settings.chat_timezone,
    )
