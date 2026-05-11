from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Sequence

from aiogram.enums import MessageEntityType
from aiogram.types import Message, MessageEntity, User


@dataclass(frozen=True, slots=True)
class PivoCommandArgs:
    planned_time: str | None
    target: str | None
    explicit_mentions: tuple[str, ...]


_USERNAME_RE = re.compile(r"(?<!\S)@([A-Za-z0-9_]{1,32})(?![A-Za-z0-9_])")
_TIME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:today|tomorrow)\s+(?:[01]?\d|2[0-3]):[0-5]\d\b", re.IGNORECASE),
    re.compile(r"^(?:сегодня|завтра)\s+(?:[01]?\d|2[0-3]):[0-5]\d\b", re.IGNORECASE),
    re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d\b"),
    re.compile(r"^(?:today|tomorrow)\s+evening\b", re.IGNORECASE),
    re.compile(r"^(?:сегодня|завтра)\s+вечером\b", re.IGNORECASE),
    re.compile(r"^evening\b", re.IGNORECASE),
    re.compile(r"^вечером\b", re.IGNORECASE),
)


def parse_pivo_command(message: Message) -> PivoCommandArgs:
    text = message.text or ""
    args_start = _find_args_start(text)
    spans: list[tuple[int, int]] = []
    explicit_mentions = list(
        _collect_entity_mentions(
            text=text,
            entities=message.entities or (),
            args_start=args_start,
            spans=spans,
        )
    )

    text_without_entity_mentions = _remove_spans(text, spans)
    args_text = text_without_entity_mentions[args_start:].strip()
    args_text, fallback_mentions = _extract_plain_username_mentions(args_text)
    explicit_mentions.extend(fallback_mentions)

    planned_time, target = _extract_leading_time(args_text.strip())
    return PivoCommandArgs(
        planned_time=planned_time,
        target=target or None,
        explicit_mentions=tuple(explicit_mentions),
    )


def _find_args_start(text: str) -> int:
    match = re.match(r"\S+", text)
    if match is None:
        return 0
    return match.end()


def _collect_entity_mentions(
    *,
    text: str,
    entities: Sequence[MessageEntity],
    args_start: int,
    spans: list[tuple[int, int]],
) -> tuple[str, ...]:
    mentions: list[str] = []
    for entity in entities:
        start = _utf16_offset_to_index(text, entity.offset)
        end = _utf16_offset_to_index(text, entity.offset + entity.length)
        if end <= args_start:
            continue

        entity_type = _entity_type_value(entity.type)
        if entity_type == MessageEntityType.MENTION.value:
            mention_text = entity.extract_from(text).strip()
            if mention_text:
                mentions.append(html.escape(mention_text, quote=False))
                spans.append((start, end))
        elif entity_type == MessageEntityType.TEXT_MENTION.value and entity.user is not None:
            mentions.append(_format_text_mention(entity.user))
            spans.append((start, end))
    return tuple(mentions)


def _entity_type_value(entity_type: object) -> str:
    value = getattr(entity_type, "value", entity_type)
    return str(value)


def _format_text_mention(user: User) -> str:
    display_name = user.full_name or user.first_name or "participant"
    safe_user_id = html.escape(str(user.id), quote=True)
    safe_display_name = html.escape(display_name, quote=False)
    return f'<a href="tg://user?id={safe_user_id}">{safe_display_name}</a>'


def _extract_plain_username_mentions(text: str) -> tuple[str, tuple[str, ...]]:
    mentions: list[str] = []

    def replace(match: re.Match[str]) -> str:
        mentions.append(html.escape(match.group(0), quote=False))
        return " "

    cleaned = _USERNAME_RE.sub(replace, text)
    return _normalize_spaces(cleaned), tuple(mentions)


def _extract_leading_time(text: str) -> tuple[str | None, str | None]:
    for pattern in _TIME_PATTERNS:
        match = pattern.match(text)
        if match is None:
            continue
        planned_time = match.group(0).strip()
        target = text[match.end() :].lstrip(" \t-,:;").strip()
        return planned_time, target or None
    return None, text or None


def _remove_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    result = text
    for start, end in sorted(spans, reverse=True):
        result = f"{result[:start]} {' ' * (end - start)} {result[end:]}"
    return result


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _utf16_offset_to_index(text: str, offset: int) -> int:
    current_offset = 0
    for index, char in enumerate(text):
        if current_offset >= offset:
            return index
        current_offset += len(char.encode("utf-16-le")) // 2
    return len(text)
