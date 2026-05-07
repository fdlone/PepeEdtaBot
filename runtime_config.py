from __future__ import annotations

from typing import Optional


ALLOWED_RUNTIME_KEYS = (
    "reply_probability",
    "min_cooldown_sec",
    "min_tokens_for_model",
    "max_reply_chars",
    "max_reply_tokens",
    "normalize_lower",
    "typing_min_ms",
    "typing_max_ms",
    "randomness_strength",
    "repetition_penalty_strength",
    "markov_order",
    "enable_backoff",
    "backoff_min_order",
    "use_reply_context",
    "reply_context_max_tokens",
    "reply_context_last_tokens",
    "reply_context_bias",
    "reply_context_start_bias",
    "reply_context_only_for_replies",
    "reply_context_include_current_message",
)

UNKNOWN_RUNTIME_KEY_MESSAGE = (
    "Неизвестный ключ.\n"
    "Доступно: " + ", ".join(ALLOWED_RUNTIME_KEYS)
)


class UnknownRuntimeSettingError(ValueError):
    pass


class InvalidRuntimeSettingValueError(ValueError):
    pass


def parse_bool(value: str) -> Optional[bool]:
    val = value.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_float_in_range(value: str, min_value: float, max_value: float) -> float:
    parsed = float(value)
    if not min_value <= parsed <= max_value:
        raise InvalidRuntimeSettingValueError
    return parsed


def _parse_int_min(value: str, min_value: int) -> int:
    parsed = int(value)
    if parsed < min_value:
        raise InvalidRuntimeSettingValueError
    return parsed


def _parse_bool_or_raise(value: str) -> bool:
    parsed = parse_bool(value)
    if parsed is None:
        raise InvalidRuntimeSettingValueError
    return parsed


def apply_runtime_setting(state: object, key: str, value: str) -> None:
    normalized_key = key.strip().lower()
    try:
        if normalized_key == "reply_probability":
            setattr(state, normalized_key, _parse_float_in_range(value, 0.0, 1.0))
        elif normalized_key == "min_cooldown_sec":
            setattr(state, normalized_key, _parse_int_min(value, 0))
        elif normalized_key == "min_tokens_for_model":
            setattr(state, normalized_key, _parse_int_min(value, 0))
        elif normalized_key == "max_reply_chars":
            parsed = int(value)
            if parsed < 20 or parsed > 4000:
                raise InvalidRuntimeSettingValueError
            setattr(state, normalized_key, parsed)
        elif normalized_key == "max_reply_tokens":
            parsed = int(value)
            if parsed < 3 or parsed > 300:
                raise InvalidRuntimeSettingValueError
            setattr(state, normalized_key, parsed)
        elif normalized_key == "normalize_lower":
            setattr(state, normalized_key, _parse_bool_or_raise(value))
        elif normalized_key == "typing_min_ms":
            parsed = _parse_int_min(value, 0)
            if parsed > state.typing_max_ms:
                raise InvalidRuntimeSettingValueError
            setattr(state, normalized_key, parsed)
        elif normalized_key == "typing_max_ms":
            parsed = int(value)
            if parsed < state.typing_min_ms:
                raise InvalidRuntimeSettingValueError
            setattr(state, normalized_key, parsed)
        elif normalized_key == "randomness_strength":
            setattr(state, normalized_key, _parse_float_in_range(value, 0.0, 3.0))
        elif normalized_key == "repetition_penalty_strength":
            setattr(state, normalized_key, _parse_float_in_range(value, 0.0, 3.0))
        elif normalized_key == "markov_order":
            parsed = int(value)
            if parsed not in {2, 3} or state.backoff_min_order >= parsed:
                raise InvalidRuntimeSettingValueError
            setattr(state, normalized_key, parsed)
        elif normalized_key == "enable_backoff":
            setattr(state, normalized_key, _parse_bool_or_raise(value))
        elif normalized_key == "backoff_min_order":
            parsed = int(value)
            if parsed not in {1, 2} or parsed >= state.markov_order:
                raise InvalidRuntimeSettingValueError
            setattr(state, normalized_key, parsed)
        elif normalized_key == "use_reply_context":
            setattr(state, normalized_key, _parse_bool_or_raise(value))
        elif normalized_key == "reply_context_max_tokens":
            parsed = _parse_int_min(value, 2)
            if parsed < state.reply_context_last_tokens:
                raise InvalidRuntimeSettingValueError
            setattr(state, normalized_key, parsed)
        elif normalized_key == "reply_context_last_tokens":
            parsed = int(value)
            if parsed not in {2, 3} or parsed > state.reply_context_max_tokens:
                raise InvalidRuntimeSettingValueError
            setattr(state, normalized_key, parsed)
        elif normalized_key == "reply_context_bias":
            setattr(state, normalized_key, _parse_float_in_range(value, 1.0, 4.0))
        elif normalized_key == "reply_context_start_bias":
            setattr(state, normalized_key, _parse_float_in_range(value, 1.0, 4.0))
        elif normalized_key == "reply_context_only_for_replies":
            setattr(state, normalized_key, _parse_bool_or_raise(value))
        elif normalized_key == "reply_context_include_current_message":
            setattr(state, normalized_key, _parse_bool_or_raise(value))
        else:
            raise UnknownRuntimeSettingError(normalized_key)
    except UnknownRuntimeSettingError:
        raise
    except (TypeError, ValueError) as exc:
        raise InvalidRuntimeSettingValueError from exc
