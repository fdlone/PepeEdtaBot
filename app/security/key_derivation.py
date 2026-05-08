"""HKDF-based domain key derivation from master secrets.

Usage:
    hmac_key  = derive_key(settings.pivo_hmac_secret,       "members:hmac")
    fernet_key = derive_fernet_key(settings.pivo_encryption_secret, "members:encryption")

pivo continues to use raw secrets directly (PivoSecurity unchanged).
New domains derive isolated keys with their own `domain` label.
"""
from __future__ import annotations

import base64

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def derive_key(master_secret: str, domain: str, length: int = 32) -> bytes:
    """Derive a domain-specific raw key via HKDF-SHA256."""
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=domain.encode("utf-8"),
    ).derive(master_secret.encode("utf-8"))


def derive_fernet_key(master_secret: str, domain: str) -> bytes:
    """Derive a URL-safe base64-encoded 32-byte key ready for Fernet."""
    return base64.urlsafe_b64encode(derive_key(master_secret, domain))
