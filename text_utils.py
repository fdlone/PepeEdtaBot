from __future__ import annotations

import re

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+", re.UNICODE)
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
    text = remove_mentions(text)
    text = normalize_repeats(text)
    text = SPACE_RE.sub(" ", text).strip()
    return text
