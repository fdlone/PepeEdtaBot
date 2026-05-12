from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from bot_policy import DEFAULT_BOT_TEXT_ALIASES
from config_registry import RUNTIME_FIELDS, validate_cross_fields


@dataclass(slots=True)
class Settings:
    bot_token: str
    reply_probability: float
    min_cooldown_sec: int
    min_tokens_for_model: int
    max_reply_chars: int
    max_reply_tokens: int
    owner_id: int | None
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
    pivo_hmac_secret: str
    pivo_encryption_secret: str
    pivo_explicit_mentions_limit: int
    pivo_subscriber_fanout_limit: int
    runtime_state_ttl_sec: int
    runtime_state_max_chats: int
    throttle_state_ttl_sec: int
    throttle_state_max_keys: int
    prefix_cache_max_messages: int
    log_level: str
    bot_text_aliases: frozenset[str]


def _load_runtime_fields() -> dict[str, object]:
    """Read all runtime-mutable fields from the environment via the registry.

    Each FieldSpec encapsulates parsing + validation, so adding a new
    runtime-mutable field requires editing only ``config_registry.py``.
    """
    values: dict[str, object] = {}
    for spec in RUNTIME_FIELDS:
        raw = os.getenv(spec.env_var, spec.default)
        try:
            values[spec.name] = spec.parse(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec.env_var}: {exc}") from exc
    return values


def load_settings(load_env: bool = True) -> Settings:
    if load_env:
        load_dotenv()

    bot_token = os.getenv("BOT_TOKEN", "").strip()
    if not bot_token:
        raise ValueError("BOT_TOKEN is required")

    owner_raw = os.getenv("OWNER_ID", "").strip()
    try:
        owner_id = int(owner_raw) if owner_raw else None
    except ValueError as exc:
        raise ValueError("OWNER_ID must be an integer") from exc

    db_path = os.getenv("DB_PATH", os.path.join("data", "markov.db")).strip()
    if not db_path:
        raise ValueError("DB_PATH must not be empty")
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    pivo_hmac_secret = os.getenv("PIVO_HMAC_SECRET", "").strip()
    if len(pivo_hmac_secret) < 16:
        raise ValueError("PIVO_HMAC_SECRET must be at least 16 characters")
    pivo_encryption_secret = os.getenv("PIVO_ENCRYPTION_SECRET", "").strip()
    if len(pivo_encryption_secret) < 16:
        raise ValueError("PIVO_ENCRYPTION_SECRET must be at least 16 characters")
    pivo_explicit_mentions_limit_raw = os.getenv(
        "PIVO_EXPLICIT_MENTIONS_LIMIT",
        "10",
    ).strip()
    pivo_subscriber_fanout_limit_raw = os.getenv(
        "PIVO_SUBSCRIBER_FANOUT_LIMIT",
        "20",
    ).strip()
    try:
        pivo_explicit_mentions_limit = int(pivo_explicit_mentions_limit_raw)
    except ValueError as exc:
        raise ValueError("PIVO_EXPLICIT_MENTIONS_LIMIT must be an integer") from exc
    if pivo_explicit_mentions_limit < 1:
        raise ValueError("PIVO_EXPLICIT_MENTIONS_LIMIT must be at least 1")
    try:
        pivo_subscriber_fanout_limit = int(pivo_subscriber_fanout_limit_raw)
    except ValueError as exc:
        raise ValueError("PIVO_SUBSCRIBER_FANOUT_LIMIT must be an integer") from exc
    if pivo_subscriber_fanout_limit < 1:
        raise ValueError("PIVO_SUBSCRIBER_FANOUT_LIMIT must be at least 1")

    runtime_state_ttl_sec_raw = os.getenv("RUNTIME_STATE_TTL_SEC", "86400").strip()
    runtime_state_max_chats_raw = os.getenv("RUNTIME_STATE_MAX_CHATS", "2048").strip()
    throttle_state_ttl_sec_raw = os.getenv("THROTTLE_STATE_TTL_SEC", "21600").strip()
    throttle_state_max_keys_raw = os.getenv("THROTTLE_STATE_MAX_KEYS", "4096").strip()
    try:
        runtime_state_ttl_sec = int(runtime_state_ttl_sec_raw)
    except ValueError as exc:
        raise ValueError("RUNTIME_STATE_TTL_SEC must be an integer") from exc
    if runtime_state_ttl_sec < 1:
        raise ValueError("RUNTIME_STATE_TTL_SEC must be at least 1")
    try:
        runtime_state_max_chats = int(runtime_state_max_chats_raw)
    except ValueError as exc:
        raise ValueError("RUNTIME_STATE_MAX_CHATS must be an integer") from exc
    if runtime_state_max_chats < 1:
        raise ValueError("RUNTIME_STATE_MAX_CHATS must be at least 1")
    try:
        throttle_state_ttl_sec = int(throttle_state_ttl_sec_raw)
    except ValueError as exc:
        raise ValueError("THROTTLE_STATE_TTL_SEC must be an integer") from exc
    if throttle_state_ttl_sec < 1:
        raise ValueError("THROTTLE_STATE_TTL_SEC must be at least 1")
    try:
        throttle_state_max_keys = int(throttle_state_max_keys_raw)
    except ValueError as exc:
        raise ValueError("THROTTLE_STATE_MAX_KEYS must be an integer") from exc
    if throttle_state_max_keys < 1:
        raise ValueError("THROTTLE_STATE_MAX_KEYS must be at least 1")
    prefix_cache_max_messages_raw = os.getenv(
        "PREFIX_CACHE_MAX_MESSAGES", "2000"
    ).strip()
    try:
        prefix_cache_max_messages = int(prefix_cache_max_messages_raw)
    except ValueError as exc:
        raise ValueError("PREFIX_CACHE_MAX_MESSAGES must be an integer") from exc
    if prefix_cache_max_messages < 1:
        raise ValueError("PREFIX_CACHE_MAX_MESSAGES must be at least 1")

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(
            "LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )

    # BOT_TEXT_ALIASES: comma-separated. Empty / unset → built-in defaults
    # so deployments without .env access keep working.
    aliases_raw = os.getenv("BOT_TEXT_ALIASES", "").strip()
    if aliases_raw:
        bot_text_aliases = frozenset(
            chunk.strip().lower()
            for chunk in aliases_raw.split(",")
            if chunk.strip()
        )
        if not bot_text_aliases:
            bot_text_aliases = DEFAULT_BOT_TEXT_ALIASES
    else:
        bot_text_aliases = DEFAULT_BOT_TEXT_ALIASES

    runtime_values = _load_runtime_fields()

    settings = Settings(
        bot_token=bot_token,
        owner_id=owner_id,
        db_path=db_path,
        pivo_hmac_secret=pivo_hmac_secret,
        pivo_encryption_secret=pivo_encryption_secret,
        pivo_explicit_mentions_limit=pivo_explicit_mentions_limit,
        pivo_subscriber_fanout_limit=pivo_subscriber_fanout_limit,
        runtime_state_ttl_sec=runtime_state_ttl_sec,
        runtime_state_max_chats=runtime_state_max_chats,
        throttle_state_ttl_sec=throttle_state_ttl_sec,
        throttle_state_max_keys=throttle_state_max_keys,
        prefix_cache_max_messages=prefix_cache_max_messages,
        log_level=log_level,
        bot_text_aliases=bot_text_aliases,
        **runtime_values,  # type: ignore[arg-type]
    )
    validate_cross_fields(settings)
    return settings
