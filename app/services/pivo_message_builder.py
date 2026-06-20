from __future__ import annotations

import html
import random
import re
from dataclasses import asdict, dataclass

from app.domain.pivo_templates import (
    PIVO_DEFAULT_BODY_PARTS,
    PIVO_DEFAULT_BOTTOM_PARTS,
    PIVO_DEFAULT_TARGET_INTROS,
    PIVO_DEFAULT_TOP_PARTS,
    PIVO_NOTIFICATION_LINES,
    PIVO_TARGET_BODY_PARTS,
    PIVO_TARGET_BOTTOM_PARTS,
    PIVO_TARGET_INTROS,
    PIVO_TARGET_TOP_PARTS,
)

MENTIONS_INLINE_FALLBACKS: tuple[str, ...] = (
    "местные дегенераты",
    "подозрительные личности",
    "конченый состав чата",
    "морально уставшие участники",
)

PIVO_FALLBACK_MENTIONS_TEXT = "Господа дегенераты"
DEFAULT_TARGET_PHRASE = "пиво, игры и коллективная деградация"
DEFAULT_TARGET_CONTEXT = "по классической программе: пиво, игры, споры и Discord"


@dataclass(frozen=True, slots=True)
class PivoMessageContext:
    mentions_inline: str
    notification_line: str
    time_phrase: str
    time_phrase_soft: str
    target_phrase: str
    target_bullet: str
    target_context: str
    has_explicit_target: bool

    def template_values(self) -> dict[str, object]:
        return asdict(self)


def build_pivo_message(
    mentions: str,
    *,
    planned_time: str | None = None,
    target: str | None = None,
    has_explicit_mentions: bool = False,
) -> str:
    context = build_pivo_message_context(
        mentions,
        planned_time=planned_time,
        target=target,
        has_explicit_mentions=has_explicit_mentions,
    )
    return PivoMessageGenerator().build(context)


def build_pivo_message_context(
    mentions: str,
    *,
    planned_time: str | None,
    target: str | None,
    has_explicit_mentions: bool,
) -> PivoMessageContext:
    raw_target = target.strip() if target is not None else ""
    has_explicit_target = bool(raw_target)
    target_phrase = html.escape(raw_target, quote=False) if has_explicit_target else ""
    target_bullet = _build_target_bullet(
        target_phrase=target_phrase,
        has_explicit_target=has_explicit_target,
    )

    return PivoMessageContext(
        mentions_inline=_build_mentions_inline(
            mentions,
            has_explicit_mentions=has_explicit_mentions,
        ),
        notification_line=_build_notification_line(
            mentions,
            has_explicit_mentions=has_explicit_mentions,
        ),
        time_phrase=_format_time_phrase(planned_time),
        time_phrase_soft=_format_time_phrase_soft(planned_time),
        target_phrase=target_phrase or DEFAULT_TARGET_PHRASE,
        target_bullet=target_bullet,
        target_context=(
            f"по теме {target_phrase}" if has_explicit_target else DEFAULT_TARGET_CONTEXT
        ),
        has_explicit_target=has_explicit_target,
    )


class PivoMessageGenerator:
    def build(self, context: PivoMessageContext) -> str:
        values = context.template_values()
        if context.has_explicit_target:
            top_pool = PIVO_TARGET_TOP_PARTS
            body_pool = PIVO_TARGET_BODY_PARTS
            bottom_pool = PIVO_TARGET_BOTTOM_PARTS
        else:
            top_pool = PIVO_DEFAULT_TOP_PARTS
            body_pool = PIVO_DEFAULT_BODY_PARTS
            bottom_pool = PIVO_DEFAULT_BOTTOM_PARTS

        parts = [
            random.choice(top_pool),
            random.choice(body_pool),
            random.choice(bottom_pool),
        ]
        if context.notification_line:
            parts.append("{notification_line}")
        template = "\n\n".join(part.strip() for part in parts if part.strip())
        return template.format(**values).strip()


def _build_target_bullet(*, target_phrase: str, has_explicit_target: bool) -> str:
    if has_explicit_target:
        return random.choice(PIVO_TARGET_INTROS).format(target_phrase=target_phrase)
    return random.choice(PIVO_DEFAULT_TARGET_INTROS)


def _build_mentions_inline(mentions: str, *, has_explicit_mentions: bool) -> str:
    if has_explicit_mentions:
        return mentions.strip()
    return random.choice(MENTIONS_INLINE_FALLBACKS)


def _build_notification_line(mentions: str, *, has_explicit_mentions: bool) -> str:
    value = mentions.strip()
    if has_explicit_mentions or not value or value == PIVO_FALLBACK_MENTIONS_TEXT:
        return ""
    template = random.choice(PIVO_NOTIFICATION_LINES)
    return template.format(mentions=value)


def _format_time_phrase(planned_time: str | None) -> str:
    if not planned_time:
        return "сегодня"
    value = _format_time_value(planned_time)
    if _time_value_has_own_preposition(value):
        return value
    return f"в {value}"


def _format_time_phrase_soft(planned_time: str | None) -> str:
    if not planned_time:
        return "когда коллектив перестанет изображать занятых людей"
    value = _format_time_value(planned_time)
    if _time_value_has_own_preposition(value):
        return f"примерно {value}"
    return f"примерно в {value}"


def _format_time_value(planned_time: str) -> str:
    raw = planned_time.strip()
    normalized = raw.lower()

    if re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", normalized):
        return html.escape(raw, quote=False)
    if normalized.startswith("today at "):
        return f"сегодня {_normalize_time_tail(raw[9:])}"
    if normalized.startswith("today "):
        return f"сегодня {_normalize_time_tail(raw[6:])}"
    if normalized.startswith("tomorrow at "):
        return f"завтра {_normalize_time_tail(raw[12:])}"
    if normalized.startswith("tomorrow "):
        return f"завтра {_normalize_time_tail(raw[9:])}"
    if normalized == "today":
        return "сегодня"
    if normalized == "tomorrow":
        return "завтра"
    if normalized.startswith("сегодня в "):
        return f"сегодня {_normalize_time_tail(raw[10:])}"
    if normalized.startswith("завтра в "):
        return f"завтра {_normalize_time_tail(raw[9:])}"
    if normalized == "сегодня":
        return "сегодня"
    if normalized == "завтра":
        return "завтра"
    if normalized == "evening":
        return "вечером"
    if normalized == "вечером":
        return "вечером"
    return html.escape(raw, quote=False)


def _normalize_time_tail(value: str) -> str:
    tail = value.strip()
    if tail.lower() == "evening":
        return "вечером"
    if re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", tail):
        return f"в {html.escape(tail, quote=False)}"
    return html.escape(tail, quote=False)


def _time_value_has_own_preposition(value: str) -> bool:
    return value in {"сегодня", "завтра", "вечером"} or value.startswith(
        ("сегодня ", "завтра ")
    )
