from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.config.defaults import (
    MESSAGES_RETENTION_PER_CHAT,
    SQLITE_BUSY_TIMEOUT_MS,
    SQLITE_WAL_AUTOCHECKPOINT_PAGES,
)
from app.config.registry import (
    RUNTIME_FIELDS,
    _parse_bool,
    validate_cross_fields,
)
from app.core.reply_policy import DEFAULT_BOT_TEXT_ALIASES


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
    auto_capitalize_replies: bool
    db_path: str
    messages_retention_per_chat: int
    sqlite_busy_timeout_ms: int
    sqlite_wal_autocheckpoint_pages: int
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
    hot_ngram_seed_chance: float
    hot_ngram_min_count: int
    hot_ngram_recency_share: float
    rare_event_chance: float
    false_start_chance: float
    rare_event_daily_cap: int
    user_quirk_chance: float
    user_quirk_min_interactions: int
    use_reply_context: bool
    fuzzy_context_casefold: bool
    reply_context_max_tokens: int
    reply_context_last_tokens: int
    reply_context_bias: float
    reply_context_start_bias: float
    context_start_affinity: float
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
    pivo_hmac_secret: str
    pivo_encryption_secret: str
    pivo_explicit_mentions_limit: int
    pivo_subscriber_fanout_limit: int
    runtime_state_ttl_sec: int
    runtime_state_max_chats: int
    throttle_state_ttl_sec: int
    throttle_state_max_keys: int
    text_cache_max_messages: int
    log_level: str
    gen_trace_log: bool
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


def _load_int(env_var: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(env_var, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer") from exc
    if value < minimum:
        qualifier = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{env_var} must be {qualifier}")
    return value


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

    messages_retention_per_chat = _load_int(
        "MESSAGES_RETENTION_PER_CHAT",
        MESSAGES_RETENTION_PER_CHAT,
        minimum=1,
    )
    sqlite_busy_timeout_ms = _load_int(
        "SQLITE_BUSY_TIMEOUT_MS",
        SQLITE_BUSY_TIMEOUT_MS,
        minimum=0,
    )
    sqlite_wal_autocheckpoint_pages = _load_int(
        "SQLITE_WAL_AUTOCHECKPOINT_PAGES",
        SQLITE_WAL_AUTOCHECKPOINT_PAGES,
        minimum=0,
    )

    pivo_hmac_secret = os.getenv("PIVO_HMAC_SECRET", "").strip()
    if len(pivo_hmac_secret) < 16:
        raise ValueError("PIVO_HMAC_SECRET must be at least 16 characters")
    if pivo_hmac_secret.lower().startswith("change_me"):
        raise ValueError(
            "PIVO_HMAC_SECRET still holds the .env.example placeholder; "
            "set a real random secret (see .env.example)"
        )
    pivo_encryption_secret = os.getenv("PIVO_ENCRYPTION_SECRET", "").strip()
    if len(pivo_encryption_secret) < 16:
        raise ValueError("PIVO_ENCRYPTION_SECRET must be at least 16 characters")
    if pivo_encryption_secret.lower().startswith("change_me"):
        raise ValueError(
            "PIVO_ENCRYPTION_SECRET still holds the .env.example placeholder; "
            "set a real random secret (see .env.example)"
        )
    if pivo_hmac_secret == pivo_encryption_secret:
        raise ValueError(
            "PIVO_HMAC_SECRET and PIVO_ENCRYPTION_SECRET must be different values"
        )
    pivo_explicit_mentions_limit = _load_int(
        "PIVO_EXPLICIT_MENTIONS_LIMIT", 10, minimum=1
    )
    pivo_subscriber_fanout_limit = _load_int(
        "PIVO_SUBSCRIBER_FANOUT_LIMIT", 20, minimum=1
    )

    runtime_state_ttl_sec = _load_int("RUNTIME_STATE_TTL_SEC", 86400, minimum=1)
    runtime_state_max_chats = _load_int("RUNTIME_STATE_MAX_CHATS", 2048, minimum=1)
    throttle_state_ttl_sec = _load_int("THROTTLE_STATE_TTL_SEC", 21600, minimum=1)
    throttle_state_max_keys = _load_int("THROTTLE_STATE_MAX_KEYS", 4096, minimum=1)
    text_cache_max_messages = _load_int("TEXT_CACHE_MAX_MESSAGES", 1000, minimum=1)
    if messages_retention_per_chat < text_cache_max_messages:
        raise ValueError(
            "MESSAGES_RETENTION_PER_CHAT must be greater than or equal to "
            "TEXT_CACHE_MAX_MESSAGES"
        )

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(
            "LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )

    # Verbose per-candidate generation trace (chat_markov.gen). Off by default:
    # it dumps candidate texts into the log on every reply -- a debugging tool,
    # not something a prod deployment should have to remember to switch off.
    try:
        gen_trace_log = _parse_bool(os.getenv("GEN_TRACE_LOG", "false"))
    except ValueError as exc:
        raise ValueError(f"GEN_TRACE_LOG: {exc}") from exc

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
        messages_retention_per_chat=messages_retention_per_chat,
        sqlite_busy_timeout_ms=sqlite_busy_timeout_ms,
        sqlite_wal_autocheckpoint_pages=sqlite_wal_autocheckpoint_pages,
        pivo_hmac_secret=pivo_hmac_secret,
        pivo_encryption_secret=pivo_encryption_secret,
        pivo_explicit_mentions_limit=pivo_explicit_mentions_limit,
        pivo_subscriber_fanout_limit=pivo_subscriber_fanout_limit,
        runtime_state_ttl_sec=runtime_state_ttl_sec,
        runtime_state_max_chats=runtime_state_max_chats,
        throttle_state_ttl_sec=throttle_state_ttl_sec,
        throttle_state_max_keys=throttle_state_max_keys,
        text_cache_max_messages=text_cache_max_messages,
        log_level=log_level,
        gen_trace_log=gen_trace_log,
        bot_text_aliases=bot_text_aliases,
        **runtime_values,  # type: ignore[arg-type]
    )
    validate_cross_fields(settings)
    return settings
