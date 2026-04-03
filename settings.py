from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    bot_token: str
    reply_probability: float
    min_cooldown_sec: int
    min_tokens_for_model: int
    max_reply_chars: int
    owner_id: Optional[int]
    normalize_lower: bool
    db_path: str
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


def load_settings() -> Settings:
    load_dotenv()

    bot_token = os.getenv("BOT_TOKEN", "").strip()
    if not bot_token:
        raise ValueError("BOT_TOKEN is required")

    reply_probability = float(os.getenv("REPLY_PROBABILITY", "0.08"))
    if not 0.0 <= reply_probability <= 1.0:
        raise ValueError("REPLY_PROBABILITY must be in range [0..1]")

    min_cooldown_sec = int(os.getenv("MIN_COOLDOWN_SEC", "45"))
    min_tokens_for_model = int(os.getenv("MIN_TOKENS_FOR_MODEL", "200"))
    max_reply_chars = int(os.getenv("MAX_REPLY_CHARS", "280"))
    owner_raw = os.getenv("OWNER_ID", "").strip()
    owner_id = int(owner_raw) if owner_raw else None
    normalize_lower = _to_bool(os.getenv("NORMALIZE_LOWER", "false"), default=False)
    db_path = os.getenv("DB_PATH", os.path.join("data", "markov.db"))
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    typing_min_ms = int(os.getenv("TYPING_MIN_MS", "350"))
    typing_max_ms = int(os.getenv("TYPING_MAX_MS", "1100"))
    if typing_min_ms < 0 or typing_max_ms < 0 or typing_min_ms > typing_max_ms:
        raise ValueError("TYPING_MIN_MS/TYPING_MAX_MS are invalid")
    randomness_strength = float(os.getenv("RANDOMNESS_STRENGTH", "2.0"))
    if not 0.0 <= randomness_strength <= 3.0:
        raise ValueError("RANDOMNESS_STRENGTH must be in range [0..3]")
    repetition_penalty_strength = float(os.getenv("REPETITION_PENALTY_STRENGTH", "1.0"))
    if not 0.0 <= repetition_penalty_strength <= 3.0:
        raise ValueError("REPETITION_PENALTY_STRENGTH must be in range [0..3]")
    markov_order = int(os.getenv("MARKOV_ORDER", "3"))
    if markov_order not in {2, 3}:
        raise ValueError("MARKOV_ORDER must be 2 or 3")
    enable_backoff = _to_bool(os.getenv("ENABLE_BACKOFF", "true"), default=True)
    backoff_min_order = int(os.getenv("BACKOFF_MIN_ORDER", "1"))
    if backoff_min_order not in {1, 2}:
        raise ValueError("BACKOFF_MIN_ORDER must be 1 or 2")
    if backoff_min_order >= markov_order:
        raise ValueError("BACKOFF_MIN_ORDER must be lower than MARKOV_ORDER")
    use_reply_context = _to_bool(os.getenv("USE_REPLY_CONTEXT", "true"), default=True)
    reply_context_max_tokens = int(os.getenv("REPLY_CONTEXT_MAX_TOKENS", "12"))
    if reply_context_max_tokens < 2:
        raise ValueError("REPLY_CONTEXT_MAX_TOKENS must be >= 2")
    reply_context_last_tokens = int(os.getenv("REPLY_CONTEXT_LAST_TOKENS", "3"))
    if reply_context_last_tokens not in {2, 3}:
        raise ValueError("REPLY_CONTEXT_LAST_TOKENS must be 2 or 3")
    if reply_context_last_tokens > reply_context_max_tokens:
        raise ValueError("REPLY_CONTEXT_LAST_TOKENS must be <= REPLY_CONTEXT_MAX_TOKENS")
    reply_context_bias = float(os.getenv("REPLY_CONTEXT_BIAS", "1.8"))
    if not 1.0 <= reply_context_bias <= 4.0:
        raise ValueError("REPLY_CONTEXT_BIAS must be in range [1.0..4.0]")
    reply_context_start_bias = float(os.getenv("REPLY_CONTEXT_START_BIAS", "2.2"))
    if not 1.0 <= reply_context_start_bias <= 4.0:
        raise ValueError("REPLY_CONTEXT_START_BIAS must be in range [1.0..4.0]")
    reply_context_only_for_replies = _to_bool(
        os.getenv("REPLY_CONTEXT_ONLY_FOR_REPLIES", "true"), default=True
    )
    reply_context_include_current_message = _to_bool(
        os.getenv("REPLY_CONTEXT_INCLUDE_CURRENT_MESSAGE", "true"), default=True
    )

    return Settings(
        bot_token=bot_token,
        reply_probability=reply_probability,
        min_cooldown_sec=min_cooldown_sec,
        min_tokens_for_model=min_tokens_for_model,
        max_reply_chars=max_reply_chars,
        owner_id=owner_id,
        normalize_lower=normalize_lower,
        db_path=db_path,
        typing_min_ms=typing_min_ms,
        typing_max_ms=typing_max_ms,
        randomness_strength=randomness_strength,
        repetition_penalty_strength=repetition_penalty_strength,
        markov_order=markov_order,
        enable_backoff=enable_backoff,
        backoff_min_order=backoff_min_order,
        use_reply_context=use_reply_context,
        reply_context_max_tokens=reply_context_max_tokens,
        reply_context_last_tokens=reply_context_last_tokens,
        reply_context_bias=reply_context_bias,
        reply_context_start_bias=reply_context_start_bias,
        reply_context_only_for_replies=reply_context_only_for_replies,
        reply_context_include_current_message=reply_context_include_current_message,
    )
