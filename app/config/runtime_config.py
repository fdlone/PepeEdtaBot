from __future__ import annotations

from app.config.registry import (
    _parse_bool,
    get_spec,
    runtime_field_names,
    try_apply,
    try_apply_for_chat,
)

ALLOWED_RUNTIME_KEYS: tuple[str, ...] = runtime_field_names()

UNKNOWN_RUNTIME_KEY_MESSAGE = "Неизвестный ключ.\nДоступно: " + ", ".join(
    ALLOWED_RUNTIME_KEYS
)


class UnknownRuntimeSettingError(ValueError):
    pass


class InvalidRuntimeSettingValueError(ValueError):
    pass


def parse_bool(value: str) -> bool | None:
    """Lenient variant of the registry boolean parser: None instead of raising."""
    try:
        return _parse_bool(value)
    except ValueError:
        return None


def apply_runtime_setting(
    state: object, key: str, value: str, *, chat_id: int | None = None
) -> None:
    """Apply a setting globally, or to one chat when ``chat_id`` is given.

    ``chat_id=None`` keeps the original process-global behaviour, which is
    what ``load_settings`` and the explicit global form of ``/set`` want.
    """
    normalized_key = key.strip().lower()
    try:
        if chat_id is None:
            try_apply(state, normalized_key, value)
        else:
            try_apply_for_chat(state, chat_id, normalized_key, value)
    except KeyError as exc:
        raise UnknownRuntimeSettingError(normalized_key) from exc
    except (TypeError, ValueError) as exc:
        raise InvalidRuntimeSettingValueError from exc


def clear_runtime_setting(state: object, chat_id: int, key: str) -> bool:
    """Drop a chat's override so it follows the global value again."""
    normalized_key = key.strip().lower()
    if get_spec(normalized_key) is None:
        raise UnknownRuntimeSettingError(normalized_key)
    cleared: bool = state.clear_override(chat_id, normalized_key)  # type: ignore[attr-defined]
    return cleared
