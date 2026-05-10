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
"""
from __future__ import annotations

import hashlib
import hmac

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_DOMAIN = b"logging:chat_id"
_MASK_HEX_LEN = 8

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
