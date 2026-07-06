"""
Single source of truth for runtime-mutable configuration fields.

Each FieldSpec describes one configuration field that lives both in
``Settings`` (loaded from environment) and ``RuntimeState`` (mutable at
runtime via ``/set``). The same parser is reused by ``load_settings`` and
``apply_runtime_setting``, eliminating triple duplication of validation
logic across ``settings.py``, ``runtime_state.py`` and ``runtime_config.py``.

Fields that are NOT runtime-mutable (BOT_TOKEN, OWNER_ID, DB_PATH,
PIVO secrets, LOG_LEVEL) intentionally live only in ``settings.py``.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _parse_bool(value: str) -> bool:
    val = value.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _int_in_range(min_v: int, max_v: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if parsed < min_v or parsed > max_v:
            raise ValueError(f"value must be in range [{min_v}..{max_v}]")
        return parsed
    return parse


def _int_min(min_v: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if parsed < min_v:
            raise ValueError(f"value must be >= {min_v}")
        return parsed
    return parse


def _int_in_set(allowed: set[int]) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if parsed not in allowed:
            raise ValueError(f"value must be one of {sorted(allowed)}")
        return parsed
    return parse


def _float_in_range(min_v: float, max_v: float) -> Callable[[str], float]:
    def parse(value: str) -> float:
        parsed = float(value)
        if parsed < min_v or parsed > max_v:
            raise ValueError(f"value must be in range [{min_v}..{max_v}]")
        return parsed
    return parse


def _parse_length_mode_weights(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(
            "value must be three comma-separated weights: short,medium,long"
        )
    try:
        short_w, medium_w, long_w = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"invalid weight in {value!r}") from exc
    if min(short_w, medium_w, long_w) < 0.0:
        raise ValueError("weights must be non-negative")
    if short_w + medium_w + long_w <= 0.0:
        raise ValueError("at least one weight must be positive")
    return short_w, medium_w, long_w


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Metadata for a single runtime-mutable configuration field."""

    name: str
    env_var: str
    default: str
    parse: Callable[[str], Any]


RUNTIME_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("reply_probability", "REPLY_PROBABILITY", "0.08",
              _float_in_range(0.0, 1.0)),
    FieldSpec("min_cooldown_sec", "MIN_COOLDOWN_SEC", "45", _int_min(0)),
    FieldSpec("min_tokens_for_model", "MIN_TOKENS_FOR_MODEL", "200", _int_min(0)),
    FieldSpec("max_reply_chars", "MAX_REPLY_CHARS", "280",
              _int_in_range(20, 4000)),
    FieldSpec("max_reply_tokens", "MAX_REPLY_TOKENS", "45",
              _int_in_range(1, 300)),
    FieldSpec("normalize_lower", "NORMALIZE_LOWER", "false", _parse_bool),
    FieldSpec(
        "auto_capitalize_replies",
        "AUTO_CAPITALIZE_REPLIES",
        "false",
        _parse_bool,
    ),
    FieldSpec("typing_min_ms", "TYPING_MIN_MS", "350", _int_min(0)),
    FieldSpec("typing_max_ms", "TYPING_MAX_MS", "1100", _int_min(0)),
    FieldSpec("typing_per_char_ms", "TYPING_PER_CHAR_MS", "12",
              _int_in_range(0, 200)),
    FieldSpec("randomness_strength", "RANDOMNESS_STRENGTH", "2.0",
              _float_in_range(0.0, 3.0)),
    FieldSpec("candidate_selection_temperature", "CANDIDATE_SELECTION_TEMPERATURE",
              "0.7", _float_in_range(0.0, 3.0)),
    FieldSpec("reply_flavor_strength", "REPLY_FLAVOR_STRENGTH", "1.0",
              _float_in_range(0.0, 2.0)),
    # M3 emoji channel: chance to append a frequency-sampled emoji (from this
    # chat's own usage) to a reply. 0 disables. Suppressed after a "?" ending;
    # a heated mood scales it up.
    FieldSpec("emoji_append_chance", "EMOJI_APPEND_CHANCE", "0.15",
              _float_in_range(0.0, 1.0)),
    FieldSpec("repetition_penalty_strength", "REPETITION_PENALTY_STRENGTH", "1.0",
              _float_in_range(0.0, 3.0)),
    # 0.5 chosen by eval sweep (2026-07-02): in the case-preserved profile
    # strength 1.0 collapsed context_token_overlap 0.21->0.09; 0.5 keeps it at
    # 0.14 with the same distinct-1/2 gain and ~1% empty-result rate.
    FieldSpec("recent_reply_penalty_strength", "RECENT_REPLY_PENALTY_STRENGTH",
              "0.5", _float_in_range(0.0, 3.0)),
    FieldSpec("length_mode_weights", "LENGTH_MODE_WEIGHTS", "0.25,0.55,0.2",
              _parse_length_mode_weights),
    FieldSpec("markov_order", "MARKOV_ORDER", "3", _int_in_set({2, 3})),
    FieldSpec("enable_backoff", "ENABLE_BACKOFF", "true", _parse_bool),
    FieldSpec("backoff_min_order", "BACKOFF_MIN_ORDER", "1", _int_in_set({1, 2})),
    # M4 topic drift: probability per generation step (only after the reply has
    # >8 tokens, order 3) of jumping to a new learned sentence start, splicing a
    # connective ("..., кстати ...") so the shift reads as a deliberate aside.
    # 0 disables (the pre-M4 behaviour).
    FieldSpec("markov_jump_probability", "MARKOV_JUMP_PROBABILITY", "0.04",
              _float_in_range(0.0, 1.0)),
    FieldSpec("use_reply_context", "USE_REPLY_CONTEXT", "true", _parse_bool),
    FieldSpec(
        "fuzzy_context_casefold",
        "FUZZY_CONTEXT_CASEFOLD",
        "true",
        _parse_bool,
    ),
    FieldSpec(
        "fuzzy_context_prefix",
        "FUZZY_CONTEXT_PREFIX",
        "false",
        _parse_bool,
    ),
    FieldSpec("reply_context_max_tokens", "REPLY_CONTEXT_MAX_TOKENS", "12",
              _int_min(2)),
    FieldSpec("reply_context_last_tokens", "REPLY_CONTEXT_LAST_TOKENS", "3",
              _int_in_set({2, 3})),
    FieldSpec("reply_context_bias", "REPLY_CONTEXT_BIAS", "1.8",
              _float_in_range(1.0, 4.0)),
    FieldSpec("reply_context_start_bias", "REPLY_CONTEXT_START_BIAS", "2.2",
              _float_in_range(1.0, 4.0)),
    FieldSpec("reply_context_only_for_replies", "REPLY_CONTEXT_ONLY_FOR_REPLIES",
              "true", _parse_bool),
    FieldSpec("reply_context_include_current_message",
              "REPLY_CONTEXT_INCLUDE_CURRENT_MESSAGE", "true", _parse_bool),
    # S2: how many recently used /pivo template indices to remember per pool per
    # chat and exclude from the next pick. 0 disables anti-repeat.
    FieldSpec("pivo_recent_pool_window", "PIVO_RECENT_POOL_WINDOW", "5",
              _int_in_range(0, 50)),
    # S4: probability of swapping the /pivo closing line for a time-aware variant
    # (late-night / Friday / Monday) when one applies. 0 disables temporal flavor.
    FieldSpec("pivo_temporal_flavor_chance", "PIVO_TEMPORAL_FLAVOR_CHANCE", "0.5",
              _float_in_range(0.0, 1.0)),
    # M1 chat mood: a hidden per-chat state (sleepy/calm/lively/heated) that drifts
    # the bot's behaviour with conversation rhythm.
    FieldSpec("mood_enabled", "MOOD_ENABLED", "true", _parse_bool),
    # Scales every mood-driven knob deviation: 0 tracks mood but changes nothing,
    # 1 uses the built-in table, higher exaggerates.
    FieldSpec("mood_modulation_strength", "MOOD_MODULATION_STRENGTH", "1.0",
              _float_in_range(0.0, 3.0)),
    # EWMA smoothing for the mood signals (higher = reacts faster).
    FieldSpec("mood_ewma_alpha", "MOOD_EWMA_ALPHA", "0.3",
              _float_in_range(0.01, 1.0)),
    # Message-rate thresholds (msg/min) separating sleepy < calm < lively.
    FieldSpec("mood_lively_rate_per_min", "MOOD_LIVELY_RATE_PER_MIN", "12.0",
              _float_in_range(0.0, 600.0)),
    FieldSpec("mood_sleepy_rate_per_min", "MOOD_SLEEPY_RATE_PER_MIN", "2.0",
              _float_in_range(0.0, 600.0)),
    # Smoothed emphatic-intensity (!/?/caps) at/above which a chat reads as heated.
    FieldSpec("mood_heated_intensity", "MOOD_HEATED_INTENSITY", "0.4",
              _float_in_range(0.0, 1.0)),
    # Clamp for the instantaneous message rate so bursts cannot spike the EWMA.
    FieldSpec("mood_max_rate_per_min", "MOOD_MAX_RATE_PER_MIN", "120.0",
              _float_in_range(1.0, 6000.0)),
    # M2 reply-probability director: master toggle. When off, the flat
    # ``reply_probability`` (× mood multiplier) governs unprompted replies exactly
    # as before; when on, conversation momentum maps into the [min, max] band below.
    FieldSpec("reply_director_enabled", "REPLY_DIRECTOR_ENABLED", "true",
              _parse_bool),
    # Momentum band: a quiet chat sits near the min, a busy/addressed one climbs to
    # the max. The legacy flat ``reply_probability`` (0.08) is roughly the midpoint.
    FieldSpec("reply_probability_min", "REPLY_PROBABILITY_MIN", "0.02",
              _float_in_range(0.0, 1.0)),
    FieldSpec("reply_probability_max", "REPLY_PROBABILITY_MAX", "0.30",
              _float_in_range(0.0, 1.0)),
    # Burst rhythm: right after the bot replies its probability is boosted for
    # ``boost_sec`` (join the exchange), then suppressed for the following
    # ``suppress_sec`` (withdraw). ``min_cooldown_sec`` remains the hard floor.
    FieldSpec("reply_burst_boost_sec", "REPLY_BURST_BOOST_SEC", "180",
              _int_min(0)),
    FieldSpec("reply_burst_boost_mult", "REPLY_BURST_BOOST_MULT", "2.0",
              _float_in_range(0.0, 10.0)),
    FieldSpec("reply_burst_suppress_sec", "REPLY_BURST_SUPPRESS_SEC", "600",
              _int_min(0)),
    FieldSpec("reply_burst_suppress_mult", "REPLY_BURST_SUPPRESS_MULT", "0.5",
              _float_in_range(0.0, 10.0)),
    # Safety cap on unprompted replies per rolling hour per chat (mentions always
    # answered, never counted against the gate). 0 disables the cap.
    FieldSpec("reply_max_per_hour", "REPLY_MAX_PER_HOUR", "20",
              _int_in_range(0, 1000)),
)


_SPECS_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in RUNTIME_FIELDS}


def get_spec(name: str) -> FieldSpec | None:
    return _SPECS_BY_NAME.get(name)


def runtime_field_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in RUNTIME_FIELDS)


def validate_cross_fields(obj: Any) -> None:
    """Cross-field validation shared by load_settings and /set.

    Raises ValueError if any invariant between fields is violated.
    """
    if obj.typing_min_ms > obj.typing_max_ms:
        raise ValueError("TYPING_MIN_MS must be <= TYPING_MAX_MS")
    if obj.backoff_min_order >= obj.markov_order:
        raise ValueError("BACKOFF_MIN_ORDER must be lower than MARKOV_ORDER")
    if obj.reply_context_last_tokens > obj.reply_context_max_tokens:
        raise ValueError(
            "REPLY_CONTEXT_LAST_TOKENS must be <= REPLY_CONTEXT_MAX_TOKENS"
        )
    if obj.mood_sleepy_rate_per_min >= obj.mood_lively_rate_per_min:
        raise ValueError(
            "MOOD_SLEEPY_RATE_PER_MIN must be lower than MOOD_LIVELY_RATE_PER_MIN"
        )
    if obj.reply_probability_min > obj.reply_probability_max:
        raise ValueError(
            "REPLY_PROBABILITY_MIN must be <= REPLY_PROBABILITY_MAX"
        )


def try_apply(state: Any, name: str, value: str) -> None:
    """Parse, validate and apply a runtime field on ``state``.

    Cross-field invariants are checked on a shallow copy first; the real
    state is mutated only if the resulting combination is valid.
    Raises ``KeyError`` for unknown names and ``ValueError`` for invalid
    values (either by spec.parse or by cross-field validation).
    """
    spec = _SPECS_BY_NAME.get(name)
    if spec is None:
        raise KeyError(name)
    parsed = spec.parse(value)
    candidate = copy.copy(state)
    setattr(candidate, name, parsed)
    validate_cross_fields(candidate)
    setattr(state, name, parsed)
