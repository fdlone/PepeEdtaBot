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
SENTENCE_ENDINGS = frozenset(".!?")


def remove_links(text: str) -> str:
    return URL_RE.sub("", text)


def remove_mentions(text: str) -> str:
    return MENTION_RE.sub("", text)


def normalize_repeats(text: str) -> str:
    return REPEAT_RE.sub(r"\1\1", text)


def capitalize_reply_sentences(text: str) -> str:
    """Capitalize eligible sentence-start letters without changing text length."""
    characters = list(text)
    sentence_start_pending = True

    for index, character in enumerate(characters):
        if character in SENTENCE_ENDINGS:
            sentence_start_pending = True
            continue
        if not sentence_start_pending or not character.isalpha():
            continue

        previous = characters[index - 1] if index > 0 else ""
        if previous.isalnum() or previous == "_":
            sentence_start_pending = False
            continue

        token_end = index
        while (
            token_end < len(characters)
            and (characters[token_end].isalnum() or characters[token_end] == "_")
        ):
            token_end += 1
        if "_" in characters[index:token_end]:
            sentence_start_pending = False
            continue

        uppercase = character.upper()
        if len(uppercase) == 1:
            characters[index] = uppercase
        sentence_start_pending = False

    return "".join(characters)


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
