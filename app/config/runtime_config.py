from __future__ import annotations

from app.config.registry import runtime_field_names, try_apply

ALLOWED_RUNTIME_KEYS: tuple[str, ...] = runtime_field_names()

UNKNOWN_RUNTIME_KEY_MESSAGE = "Неизвестный ключ.\nДоступно: " + ", ".join(
    ALLOWED_RUNTIME_KEYS
)


class UnknownRuntimeSettingError(ValueError):
    pass


class InvalidRuntimeSettingValueError(ValueError):
    pass


def parse_bool(value: str) -> bool | None:
    val = value.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return None


def apply_runtime_setting(state: object, key: str, value: str) -> None:
    normalized_key = key.strip().lower()
    try:
        try_apply(state, normalized_key, value)
    except KeyError as exc:
        raise UnknownRuntimeSettingError(normalized_key) from exc
    except (TypeError, ValueError) as exc:
        raise InvalidRuntimeSettingValueError from exc
