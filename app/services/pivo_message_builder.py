from __future__ import annotations

import html
import random
import re

from pivo_templates import ENDING_PHRASES, PIVO_TEMPLATES

NEUTRAL_TARGET_LINES: tuple[str, ...] = (
    "Повестка вечера: {target}.",
    "План на вечер: {target}.",
    "Основная программа: {target}.",
    "Сегодняшний повод для сбора: {target}.",
    "В программе: {target}.",
)

NEUTRAL_SUPPORT_LINES: tuple[str, ...] = (
    "Discord, напитки по желанию и коллективное моральное разложение прилагаются.",
    "Споры допускаются, но без попыток делать вид, что кто-то здесь взрослый.",
    "Пиво приветствуется. Другой напиток тоже переживём.",
    "Главное - зайти в Discord и не изображать занятого человека.",
    "Уныние не приносить, его и так хватает.",
)

TIME_LINES: tuple[str, ...] = (
    "{time_phrase} сбор в Discord.",
    "Собираемся в Discord {time_phrase}.",
    "Discord открывается для морально сомнительных личностей {time_phrase}.",
)


def build_pivo_message(
    mentions: str,
    *,
    planned_time: str | None = None,
    target: str | None = None,
) -> str:
    template = random.choice(PIVO_TEMPLATES)
    text = template.format(mentions=mentions).strip()
    if target:
        return _build_target_message(
            text,
            html.escape(target, quote=False),
            planned_time=planned_time,
        )
    if planned_time:
        text = _apply_time(text, _format_time_phrase(planned_time))
    return text


def _build_target_message(
    source_text: str,
    target: str,
    *,
    planned_time: str | None,
) -> str:
    mentions, intro = _split_mentions_and_intro(source_text)
    lines = [mentions, "", intro, ""]

    if planned_time:
        lines.extend(
            [
                random.choice(TIME_LINES).format(
                    time_phrase=_format_time_phrase(planned_time)
                ),
                "",
            ]
        )

    lines.extend(
        [
            random.choice(NEUTRAL_TARGET_LINES).format(target=target),
            random.choice(NEUTRAL_SUPPORT_LINES),
            "",
            random.choice(ENDING_PHRASES),
        ]
    )
    return "\n".join(lines).strip()


def _split_mentions_and_intro(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", "Сегодня общий сбор в Discord."
    mentions = lines[0]
    intro = next((line for line in lines[1:] if "Discord" not in line), "")
    return mentions, intro or "Сегодня общий сбор в Discord."


def _apply_time(text: str, time_phrase: str) -> str:
    replacements = (
        (r"\bСегодня вечером\b", _capitalize(time_phrase)),
        (r"\bСегодня\b", _capitalize(time_phrase)),
        (r"\bсегодня вечером\b", time_phrase),
        (r"\bсегодня\b", time_phrase),
    )
    for pattern, replacement in replacements:
        updated, count = re.subn(pattern, replacement, text, count=1)
        if count:
            return updated
    return text.replace("Discord", f"Discord {time_phrase}", 1)


def _format_time_phrase(planned_time: str) -> str:
    raw = planned_time.strip()
    normalized = raw.lower()
    escaped = html.escape(raw, quote=False)

    if re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", normalized):
        return f"сегодня в {escaped}"
    if normalized.startswith("today "):
        return f"сегодня {_normalize_time_tail(raw[6:])}"
    if normalized.startswith("tomorrow "):
        return f"завтра {_normalize_time_tail(raw[9:])}"
    if normalized == "evening":
        return "сегодня вечером"
    return escaped


def _normalize_time_tail(value: str) -> str:
    tail = value.strip()
    if tail.lower() == "evening":
        return "вечером"
    if re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", tail):
        return f"в {html.escape(tail, quote=False)}"
    return html.escape(tail, quote=False)


def _capitalize(value: str) -> str:
    if not value:
        return value
    return f"{value[0].upper()}{value[1:]}"
