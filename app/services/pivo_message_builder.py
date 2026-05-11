from __future__ import annotations

import html
import random
import re

from pivo_templates import PIVO_TEMPLATES


def build_pivo_message(
    mentions: str,
    *,
    planned_time: str | None = None,
    target: str | None = None,
) -> str:
    template = random.choice(PIVO_TEMPLATES)
    text = template.format(mentions=mentions).strip()
    if target:
        text = _apply_target(text, html.escape(target, quote=False))
    if planned_time:
        text = _apply_time(text, _format_time_phrase(planned_time))
    return text


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


def _apply_target(text: str, target: str) -> str:
    updated = _replace_activity_blocks(text, target)
    updated = _replace_inline_activity_mentions(updated, target)
    if updated != text:
        return updated
    return _insert_target_after_discord_intro(text, target)


def _replace_activity_blocks(text: str, target: str) -> str:
    block_replacements = (
        (
            r"(?m)^Возможные дисциплины: .+$",
            f"Повестка вечера: {target}.",
        ),
        (
            r"(?m)^Варианты игр: .+$",
            f"Повестка вечера: {target}.",
        ),
        (
            r"(?m)^Игры на выбор: .+$",
            f"Повестка вечера: {target}.",
        ),
        (
            r"(?m)^Игры: .+$",
            f"Повестка вечера: {target}.",
        ),
        (
            r"(?m)^Рекомендуемые развлечения: .+$",
            f"Повестка вечера: {target}.",
        ),
        (
            r"(?m)^СИГейм, Codenames, рисовалка.*$",
            f"Повестка вечера: {target}.",
        ),
        (
            r"(?m)^СИГейм;\nCodenames;\nрисовалка;",
            f"{target};",
        ),
        (
            r"(?m)^Рекомендуемые дисциплины:\nСИГейм для .+\nCodenames для .+\nрисовалка для .+",
            f"Рекомендуемая дисциплина:\n{target}.",
        ),
    )
    updated = text
    for pattern, replacement in block_replacements:
        updated = re.sub(pattern, replacement, updated, count=1)
    return updated


def _replace_inline_activity_mentions(text: str, target: str) -> str:
    inline_patterns = (
        r"СИГейм, Codenames, рисовалка или [^.]+",
        r"СИГейм, рисовалка, Codenames или [^.]+",
        r"СИГейм, Codenames, рисовалка, Gartic Phone, [^.]+",
        r"СИГейм, рисовалка",
        r"СИГейма или рисовалки",
        r"Codenames",
        r"СИГейм",
        r"рисовалка",
        r"игры",
    )
    updated = text
    for pattern in inline_patterns:
        updated = re.sub(pattern, target, updated, count=1, flags=re.IGNORECASE)
    return updated


def _insert_target_after_discord_intro(text: str, target: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "Discord" in line:
            lines.insert(index + 1, f"Повестка вечера: {target}.")
            return "\n".join(lines)
    return f"{text}\n\nПовестка вечера: {target}."


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
