from __future__ import annotations

from dataclasses import dataclass, field

from settings import Settings


@dataclass(slots=True)
class RuntimeState:
    reply_probability: float
    min_cooldown_sec: int
    min_tokens_for_model: int
    max_reply_chars: int
    max_reply_tokens: int
    normalize_lower: bool
    typing_min_ms: int
    typing_max_ms: int
    randomness_strength: float
    repetition_penalty_strength: float
    markov_order: int
    enable_backoff: bool
    backoff_min_order: int
    use_reply_context: bool
    reply_context_max_tokens: int
    reply_context_last_tokens: int
    reply_context_bias: float
    reply_context_start_bias: float
    reply_context_only_for_replies: bool
    reply_context_include_current_message: bool
    last_reply_ts: dict[int, float] = field(default_factory=dict)
    pending_seed: dict[int, list[str]] = field(default_factory=dict)
    learned_messages: dict[int, int] = field(default_factory=dict)


def runtime_state_from_settings(settings: Settings) -> RuntimeState:
    return RuntimeState(
        reply_probability=settings.reply_probability,
        min_cooldown_sec=settings.min_cooldown_sec,
        min_tokens_for_model=settings.min_tokens_for_model,
        max_reply_chars=settings.max_reply_chars,
        max_reply_tokens=settings.max_reply_tokens,
        normalize_lower=settings.normalize_lower,
        typing_min_ms=settings.typing_min_ms,
        typing_max_ms=settings.typing_max_ms,
        randomness_strength=settings.randomness_strength,
        repetition_penalty_strength=settings.repetition_penalty_strength,
        markov_order=settings.markov_order,
        enable_backoff=settings.enable_backoff,
        backoff_min_order=settings.backoff_min_order,
        use_reply_context=settings.use_reply_context,
        reply_context_max_tokens=settings.reply_context_max_tokens,
        reply_context_last_tokens=settings.reply_context_last_tokens,
        reply_context_bias=settings.reply_context_bias,
        reply_context_start_bias=settings.reply_context_start_bias,
        reply_context_only_for_replies=settings.reply_context_only_for_replies,
        reply_context_include_current_message=settings.reply_context_include_current_message,
    )
