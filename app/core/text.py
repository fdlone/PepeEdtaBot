from __future__ import annotations

import re

from app.core.privacy_filter import redact_sensitive_data

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
# `(?<!\w)` prevents matching the `@host` part of an email address like
# `user@example.com` — without it, `@example` would be stripped and leave
# `user.com` in the normalized text.
MENTION_RE = re.compile(r"(?<!\w)@\w+", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
REPEAT_RE = re.compile(r"(.)\1{2,}", re.UNICODE)


def remove_links(text: str) -> str:
    return URL_RE.sub("", text)


def remove_mentions(text: str) -> str:
    return MENTION_RE.sub("", text)


def normalize_repeats(text: str) -> str:
    return REPEAT_RE.sub(r"\1\1", text)


def sanitize_text(text: str) -> str:
    text = remove_links(text)
    # Redact PII (emails, phone numbers, secrets) before mention removal so the
    # corpus, stored normalized_text, reply context and verbatim-copy checks
    # never see sensitive data. Redaction replaces spans with spaces, which the
    # final whitespace normalization collapses.
    text = redact_sensitive_data(text)
    text = remove_mentions(text)
    text = normalize_repeats(text)
    text = SPACE_RE.sub(" ", text).strip()
    return text
