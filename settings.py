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
    db_path = os.getenv("DB_PATH", "markov.db")

    return Settings(
        bot_token=bot_token,
        reply_probability=reply_probability,
        min_cooldown_sec=min_cooldown_sec,
        min_tokens_for_model=min_tokens_for_model,
        max_reply_chars=max_reply_chars,
        owner_id=owner_id,
        normalize_lower=normalize_lower,
        db_path=db_path,
    )
