from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from app.config.registry import RUNTIME_FIELDS
from app.config.settings import Settings


@dataclass(slots=True)
class RuntimeState:
    reply_probability: float
    min_cooldown_sec: int
    min_tokens_for_model: int
    max_reply_chars: int
    max_reply_tokens: int
    normalize_lower: bool
    auto_capitalize_replies: bool
    typing_min_ms: int
    typing_max_ms: int
    randomness_strength: float
    repetition_penalty_strength: float
    markov_order: int
    enable_backoff: bool
    backoff_min_order: int
    use_reply_context: bool
    fuzzy_context_casefold: bool
    fuzzy_context_prefix: bool
    reply_context_max_tokens: int
    reply_context_last_tokens: int
    reply_context_bias: float
    reply_context_start_bias: float
    reply_context_only_for_replies: bool
    reply_context_include_current_message: bool
    runtime_state_ttl_sec: int
    runtime_state_max_chats: int
    last_reply_ts: dict[int, float] = field(default_factory=dict)
    learned_messages: dict[int, int] = field(default_factory=dict)
    recent_short_replies: dict[int, deque[str]] = field(default_factory=dict)
    _last_chat_activity: dict[int, float] = field(default_factory=dict)
    _cleanup_tick: int = 0

    def note_chat_activity(self, chat_id: int, now: float) -> None:
        self._last_chat_activity[chat_id] = now
        self._cleanup_tick += 1
        if (
            self._cleanup_tick >= 64
            or len(self._last_chat_activity) > self.runtime_state_max_chats
        ):
            self.prune_inactive(now)

    def forget_chat(self, chat_id: int) -> None:
        self.last_reply_ts.pop(chat_id, None)
        self.learned_messages.pop(chat_id, None)
        self.recent_short_replies.pop(chat_id, None)
        self._last_chat_activity.pop(chat_id, None)

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

        self._cleanup_tick = 0


def runtime_state_from_settings(settings: Settings) -> RuntimeState:
    """Build a fresh RuntimeState from Settings via the config registry.

    Iterating over RUNTIME_FIELDS guarantees that adding a new
    runtime-mutable field requires editing only ``config_registry.py``.
    """
    return RuntimeState(
        **{spec.name: getattr(settings, spec.name) for spec in RUNTIME_FIELDS},
        runtime_state_ttl_sec=settings.runtime_state_ttl_sec,
        runtime_state_max_chats=settings.runtime_state_max_chats,
    )
