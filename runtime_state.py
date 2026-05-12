from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from config_registry import RUNTIME_FIELDS
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
    learned_messages: dict[int, int] = field(default_factory=dict)
    recent_short_replies: dict[int, deque[str]] = field(default_factory=dict)


def runtime_state_from_settings(settings: Settings) -> RuntimeState:
    """Build a fresh RuntimeState from Settings via the config registry.

    Iterating over RUNTIME_FIELDS guarantees that adding a new
    runtime-mutable field requires editing only ``config_registry.py``.
    """
    return RuntimeState(
        **{spec.name: getattr(settings, spec.name) for spec in RUNTIME_FIELDS}
    )
