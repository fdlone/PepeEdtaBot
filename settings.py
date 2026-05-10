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
        log_level=log_level,
        bot_text_aliases=bot_text_aliases,
        **runtime_values,  # type: ignore[arg-type]
    )
    validate_cross_fields(settings)
    return settings
