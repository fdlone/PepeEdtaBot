from __future__ import annotations

import math
import re

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9_+%.-])"
    r"[A-Za-z0-9_+%.-]+@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,}"
    r"(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)

_PHONE_CANDIDATE_RE = re.compile(
    r"(?<![\w+])\+?[0-9](?:[0-9 ().-]*[0-9])?(?!\w)",
)
_PHONE_PUNCTUATION_RE = re.compile(r"[()+.-]")

_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"
    r"(?![A-Za-z0-9_-])"
)
_TELEGRAM_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])\d{6,}:[A-Za-z0-9_-]{30,}(?![A-Za-z0-9_-])"
)
_PREFIXED_SECRET_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:sk-|gh[pousr]_|xox[baprs]-|AIza|AKIA)"
    r"[A-Za-z0-9_-]{20,}"
    r"(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
_GENERIC_SECRET_RE = re.compile(r"(?<!\S)[A-Za-z0-9_-]{24,128}(?!\S)")
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_HEX_RE = re.compile(r"[0-9a-fA-F]{32,}")


def redact_emails(text: str) -> str:
    """Replace conservative email-address matches with one space."""
    return _EMAIL_RE.sub(" ", text)


def _is_phone_candidate(candidate: str) -> bool:
    digits = "".join(character for character in candidate if character.isdigit())
    digit_count = len(digits)
    has_plus = candidate.startswith("+")

    if has_plus and 8 <= digit_count <= 15:
        return True
    if digit_count == 10 and digits.startswith("0"):
        return True
    if digit_count == 11 and digits.startswith(("7", "8")):
        return True

    has_phone_punctuation = _PHONE_PUNCTUATION_RE.search(candidate) is not None
    return has_phone_punctuation and 10 <= digit_count <= 15


def redact_phones(text: str) -> str:
    """Replace validated phone-number candidates with one space."""

    def replace_candidate(match: re.Match[str]) -> str:
        candidate = match.group(0)
        return " " if _is_phone_candidate(candidate) else candidate

    return _PHONE_CANDIDATE_RE.sub(replace_candidate, text)


def _shannon_entropy(token: str) -> float:
    length = len(token)
    counts = {character: token.count(character) for character in set(token)}
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def _is_generic_secret(token: str) -> bool:
    if _UUID_RE.fullmatch(token) is not None:
        return False
    if _HEX_RE.fullmatch(token) is not None:
        return True

    class_count = sum(
        (
            any(character.islower() for character in token),
            any(character.isupper() for character in token),
            any(character.isdigit() for character in token),
            any(character in "_-" for character in token),
        )
    )
    return class_count >= 3 and _shannon_entropy(token) >= 3.5


def redact_secrets(text: str) -> str:
    """Replace known and high-confidence generic secret forms with one space."""
    redacted = _JWT_RE.sub(" ", text)
    redacted = _TELEGRAM_TOKEN_RE.sub(" ", redacted)
    redacted = _PREFIXED_SECRET_RE.sub(" ", redacted)

    def replace_generic(match: re.Match[str]) -> str:
        token = match.group(0)
        return " " if _is_generic_secret(token) else token

    return _GENERIC_SECRET_RE.sub(replace_generic, redacted)


def redact_sensitive_data(text: str) -> str:
    """Redact emails, then phone numbers, then secrets."""
    return redact_secrets(redact_phones(redact_emails(text)))
