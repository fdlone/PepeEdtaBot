"""Stable, short, irreversible mask for ``chat_id`` in log lines.

Goals:
- Operators sharing logs (screenshots, copy-paste) don't leak raw chat IDs.
- The mask stays the same within a deployment so two log lines about the
  same chat can be correlated.
- The mask changes when ``PIVO_HMAC_SECRET`` rotates — that is a desired
  property, because rotation should also break correlation across the
  rotation boundary.

The key is derived via HKDF-SHA256 from ``PIVO_HMAC_SECRET`` with the
domain label ``logging:chat_id``. The mask is the first 8 hex chars of
HMAC-SHA256(key, str(chat_id)) — 32 bits of entropy, more than enough
for a single bot with a few dozen chats.

Usage::

    # main.py, once at startup
    log_masking.init_masking(settings.pivo_hmac_secret)

    # anywhere
    from app.log_masking import mask_chat_id
    logger.info("chat=%s ...", mask_chat_id(message.chat.id))

Calling ``mask_chat_id`` before ``init_masking`` raises
``LogMaskingNotInitialized`` — failing fast is better than silently
leaking raw IDs because someone forgot to wire the masker.

The invariant covers more than our own log calls: a third-party
exception can carry a raw ``chat_id`` inside its message text (aiogram
bakes one into ``TelegramRetryAfter`` and ``TelegramMigrateToChat``).
``mask_chat_ids_in_text`` exists for that case — it masks identifiers
found in text we did not format ourselves.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
import traceback

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_DOMAIN = b"logging:chat_id"
_MASK_HEX_LEN = 8

# Stands in for a mask when the key was never derived. Keeps the raw id
# out of the log without making the caller handle an exception.
UNMASKED_PLACEHOLDER = "unmasked"

# A chat id inside foreign text: a negative integer long enough that
# nothing else in these messages can be confused for one. Legacy basic
# groups can have ids as short as -1xxxxx, so the bound is 6 digits: low
# enough to catch them, high enough that small negative numbers (offsets,
# deltas) in the same messages survive untouched.
_MIN_CHAT_ID_DIGITS = 6
_CHAT_ID_IN_TEXT = re.compile(rf"(?<![\d-])-\d{{{_MIN_CHAT_ID_DIGITS},}}(?!\d)")

_key: bytes | None = None


class LogMaskingNotInitialized(RuntimeError):
    """``mask_chat_id`` was called before ``init_masking``."""


def init_masking(secret: str) -> None:
    """Derive and cache the HKDF key for chat_id masking.

    Idempotent: calling twice with the same secret is a no-op; calling
    with a different secret replaces the key (handy for tests).
    """
    global _key
    _key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=_DOMAIN,
    ).derive(secret.encode("utf-8"))


def reset_masking() -> None:
    """Drop the cached key. Tests use this to assert fail-fast behavior."""
    global _key
    _key = None


def mask_chat_id(chat_id: int) -> str:
    """Return the deterministic 8-hex mask of ``chat_id``.

    Raises :class:`LogMaskingNotInitialized` if ``init_masking`` was
    never called for this process.
    """
    if _key is None:
        raise LogMaskingNotInitialized(
            "Call log_masking.init_masking(secret) at startup."
        )
    digest = hmac.new(_key, str(chat_id).encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:_MASK_HEX_LEN]


def mask_chat_ids_in_text(text: str) -> str:
    """Mask chat identifiers that appear inside a text we did not format.

    Recognition is by shape, not by phrasing: the wording belongs to
    aiogram and Telegram and changes between versions, while the shape of
    an identifier does not. A chat id is a negative integer with at least
    ``_MIN_CHAT_ID_DIGITS`` digits — supergroups (``-100…``) and basic
    groups, including short legacy ids, both match, while everything else
    these messages carry (retry delays, error codes, message ids) does
    not: those are positive, or far too short.

    Unlike :func:`mask_chat_id` this never raises. It is used on the error
    path, where losing the log line costs more than losing one field, so
    an underived key yields a placeholder instead of an exception — the
    raw identifier is still never written.
    """
    return _CHAT_ID_IN_TEXT.sub(_mask_match, text)


def _mask_match(match: re.Match[str]) -> str:
    try:
        return mask_chat_id(int(match.group(0)))
    except LogMaskingNotInitialized:
        return UNMASKED_PLACEHOLDER


class MaskingFilter(logging.Filter):
    """Маскирование `chat_id` на уровне логгера, а не на каждом вызове.

    Правило §4 CLAUDE.md — «маскировать везде, включая текст чужих
    исключений» — трижды держалось перечислением точек и трижды промахивалось:
    сперва общий обработчик ошибок (2026-08-05), потом фильтр прав и путь
    отправки (B-2/E-04), потом обёртка `PartialDeliveryError`, уводившая
    исключение мимо ветки маскирования (V-1), и наконец восемь `exc_info` в
    обработчиках `/pivo` и обучения (W3-1). Каждый раз правило было записано,
    и каждый раз новое место о нём не знало.

    Фильтр снимает зависимость от внимательности: он видит **все** записи
    логгера, включая трассы `exc_info`, которые точечная обёртка вокруг
    `str(exc)` не покрывает в принципе. Исключение форматируется здесь же и
    кладётся в `exc_text` уже замаскированным — форматтер использует готовый
    текст и повторно к `exc_info` не обращается.

    Записи ниже действующего уровня логирования сюда не доходят, поэтому
    отладочные вызовы на горячем пути за регулярку не платят.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and record.exc_info[1] is not None:
            record.exc_text = mask_chat_ids_in_text(
                "".join(traceback.format_exception(record.exc_info[1]))
            ).rstrip()
            record.exc_info = None
        try:
            message = record.getMessage()
        except Exception:
            # Кривой формат — не повод потерять запись целиком; пусть её
            # отформатирует стандартный путь, как и раньше.
            return True
        masked = mask_chat_ids_in_text(message)
        if masked != message:
            record.msg = masked
            record.args = ()
        return True
