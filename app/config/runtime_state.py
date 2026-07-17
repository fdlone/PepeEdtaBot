from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from app.config.registry import RUNTIME_FIELDS
from app.config.settings import Settings
from app.core.mood import ChatMoodState, MoodConfig


def get_recent_deque(
    store: dict[int, deque[str]], chat_id: int, maxlen: int
) -> deque[str]:
    """Per-chat bounded history from ``store``, created on first access."""
    recent = store.get(chat_id)
    if recent is None:
        recent = deque(maxlen=maxlen)
        store[chat_id] = recent
    return recent


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
    typing_per_char_ms: int
    randomness_strength: float
    candidate_selection_temperature: float
    reply_flavor_strength: float
    emoji_append_chance: float
    repetition_penalty_strength: float
    recent_reply_penalty_strength: float
    verbatim_penalty_strength: float
    length_mode_weights: tuple[float, float, float]
    length_context_adaptation: float
    markov_order: int
    enable_backoff: bool
    markov_jump_probability: float
    context_jump_boost: float
    verbatim_extension_share: float
    order_mix_probability: float
    hot_ngram_seed_chance: float
    hot_ngram_min_count: int
    hot_ngram_recency_share: float
    rare_event_chance: float
    false_start_chance: float
    rare_event_daily_cap: int
    user_quirk_chance: float
    user_quirk_min_interactions: int
    user_quirk_name_share: float
    use_reply_context: bool
    fuzzy_context_casefold: bool
    reply_context_max_tokens: int
    reply_context_bias: float
    reply_context_start_bias: float
    context_start_affinity: float
    context_anchor_splice_probability: float
    reply_context_only_for_replies: bool
    reply_context_include_current_message: bool
    pivo_recent_pool_window: int
    pivo_temporal_flavor_chance: float
    mood_enabled: bool
    mood_modulation_strength: float
    mood_ewma_alpha: float
    mood_lively_rate_per_min: float
    mood_sleepy_rate_per_min: float
    mood_heated_intensity: float
    mood_mention_heated_share: float
    mood_max_rate_per_min: float
    reply_director_enabled: bool
    reply_probability_min: float
    reply_probability_max: float
    reply_burst_boost_sec: int
    reply_burst_boost_mult: float
    reply_burst_suppress_sec: int
    reply_burst_suppress_mult: float
    reply_max_per_hour: int
    mention_cooldown_sec: int
    runtime_state_ttl_sec: int
    runtime_state_max_chats: int
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
    _cleanup_tick: int = 0

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
