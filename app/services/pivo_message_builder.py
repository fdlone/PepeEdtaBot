from __future__ import annotations

import html
import random
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime

from app.domain.pivo_templates import (
    PIVO_DEFAULT_BODY_PARTS,
    PIVO_DEFAULT_BOTTOM_PARTS,
    PIVO_DEFAULT_TARGET_INTROS,
    PIVO_DEFAULT_TOP_PARTS,
    PIVO_FRIDAY_BOTTOM_PARTS,
    PIVO_LATE_NIGHT_BOTTOM_PARTS,
    PIVO_MONDAY_BOTTOM_PARTS,
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
    rng: random.Random | None = None,
) -> str:
    context = build_pivo_message_context(
        mentions,
        planned_time=planned_time,
        target=target,
        has_explicit_mentions=has_explicit_mentions,
        rng=rng,
    )
    return PivoMessageGenerator().build(context, rng=rng).text


def build_pivo_message_context(
    mentions: str,
    *,
    planned_time: str | None,
    target: str | None,
    has_explicit_mentions: bool,
    rng: random.Random | None = None,
) -> PivoMessageContext:
    raw_target = target.strip() if target is not None else ""
    has_explicit_target = bool(raw_target)
    target_phrase = html.escape(raw_target, quote=False) if has_explicit_target else ""
    target_bullet = _build_target_bullet(
        target_phrase=target_phrase,
        has_explicit_target=has_explicit_target,
        rng=rng,
    )

    return PivoMessageContext(
        mentions_inline=_build_mentions_inline(
            mentions,
            has_explicit_mentions=has_explicit_mentions,
            rng=rng,
        ),
        notification_line=_build_notification_line(
            mentions,
            has_explicit_mentions=has_explicit_mentions,
            rng=rng,
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


def _choice(pool: tuple[str, ...], rng: random.Random | None) -> str:
    """Pick from ``pool`` using the injected RNG, or the module RNG if None.

    Falling back to ``random.choice`` (not a fresh ``Random``) keeps the
    function patchable in tests and preserves the historical default.
    """
    return random.choice(pool) if rng is None else rng.choice(pool)


def _pick(
    pool: tuple[str, ...],
    rng: random.Random | None,
    avoid: frozenset[int],
) -> tuple[int, str]:
    """Pick a pool entry, excluding recently used indices when any remain.

    Selection goes through ``_choice`` over the (optionally filtered) pool, so
    the ``random.choice`` patch point used across the pivo tests keeps working.
    With an empty ``avoid`` set the filtered pool is the pool itself, so seeded
    output is byte-for-byte identical to the previous ``_choice(pool)`` call.

    Assumes ``pool`` has no duplicate entries: the drawn index is recovered via
    ``pool.index(chosen)``, which for a duplicated string would always report the
    first occurrence and thus slightly skew anti-repeat. All current pivo pools
    are duplicate-free; keep them so, or switch to indexing ``(i, part)`` pairs.
    """
    filtered = pool
    if avoid:
        candidates = tuple(part for i, part in enumerate(pool) if i not in avoid)
        if candidates:
            filtered = candidates
    chosen = _choice(filtered, rng)
    return pool.index(chosen), chosen


def _temporal_bottoms(now: datetime) -> tuple[str, ...]:
    """Collect the time-aware closing lines that apply at ``now`` (may be empty)."""
    pools: list[str] = []
    if 0 <= now.hour < 6:
        pools.extend(PIVO_LATE_NIGHT_BOTTOM_PARTS)
    weekday = now.weekday()
    if weekday == 4:
        pools.extend(PIVO_FRIDAY_BOTTOM_PARTS)
    elif weekday == 0:
        pools.extend(PIVO_MONDAY_BOTTOM_PARTS)
    return tuple(pools)


@dataclass(frozen=True, slots=True)
class PivoBuildResult:
    text: str
    # Indices actually drawn from the anti-repeat-tracked base pools, keyed by
    # "<variant>_<slot>". Omits the bottom slot when a temporal line was used.
    picks: dict[str, int]


class PivoMessageGenerator:
    def build(
        self,
        context: PivoMessageContext,
        *,
        rng: random.Random | None = None,
        recent_indices: Mapping[str, tuple[int, ...]] | None = None,
        now: datetime | None = None,
        temporal_flavor_chance: float = 0.0,
    ) -> PivoBuildResult:
        values = context.template_values()
        variant = "target" if context.has_explicit_target else "default"
        if context.has_explicit_target:
            top_pool = PIVO_TARGET_TOP_PARTS
            body_pool = PIVO_TARGET_BODY_PARTS
            bottom_pool = PIVO_TARGET_BOTTOM_PARTS
        else:
            top_pool = PIVO_DEFAULT_TOP_PARTS
            body_pool = PIVO_DEFAULT_BODY_PARTS
            bottom_pool = PIVO_DEFAULT_BOTTOM_PARTS

        recent = recent_indices or {}
        picks: dict[str, int] = {}

        top_idx, top_text = _pick(
            top_pool, rng, frozenset(recent.get(f"{variant}_top", ()))
        )
        picks[f"{variant}_top"] = top_idx
        body_idx, body_text = _pick(
            body_pool, rng, frozenset(recent.get(f"{variant}_body", ()))
        )
        picks[f"{variant}_body"] = body_idx

        bottom_text = self._pick_bottom(
            variant=variant,
            bottom_pool=bottom_pool,
            recent=recent,
            picks=picks,
            rng=rng,
            now=now,
            temporal_flavor_chance=temporal_flavor_chance,
        )

        parts = [top_text, body_text, bottom_text]
        if context.notification_line:
            parts.append("{notification_line}")
        template = "\n\n".join(part.strip() for part in parts if part.strip())
        return PivoBuildResult(text=template.format(**values).strip(), picks=picks)

    def _pick_bottom(
        self,
        *,
        variant: str,
        bottom_pool: tuple[str, ...],
        recent: Mapping[str, tuple[int, ...]],
        picks: dict[str, int],
        rng: random.Random | None,
        now: datetime | None,
        temporal_flavor_chance: float,
    ) -> str:
        """Pick the closing line, optionally swapping in a time-aware variant.

        A temporal line replaces the neutral bottom part only when a time bucket
        applies and the flavor roll passes; in that case no base-pool index is
        recorded (temporal pools are small and time-gated, so they are left out
        of anti-repeat). The neutral pool is always the fallback.
        """
        temporal = _temporal_bottoms(now) if now is not None else ()
        if temporal and temporal_flavor_chance > 0.0:
            roll = random.random() if rng is None else rng.random()
            if roll < temporal_flavor_chance:
                return _choice(temporal, rng)
        bottom_idx, bottom_text = _pick(
            bottom_pool, rng, frozenset(recent.get(f"{variant}_bottom", ()))
        )
        picks[f"{variant}_bottom"] = bottom_idx
        return bottom_text


def _build_target_bullet(
    *, target_phrase: str, has_explicit_target: bool, rng: random.Random | None = None
) -> str:
    if has_explicit_target:
        return _choice(PIVO_TARGET_INTROS, rng).format(target_phrase=target_phrase)
    return _choice(PIVO_DEFAULT_TARGET_INTROS, rng)


def _build_mentions_inline(
    mentions: str, *, has_explicit_mentions: bool, rng: random.Random | None = None
) -> str:
    if has_explicit_mentions:
        return mentions.strip()
    return _choice(MENTIONS_INLINE_FALLBACKS, rng)


def _build_notification_line(
    mentions: str, *, has_explicit_mentions: bool, rng: random.Random | None = None
) -> str:
    value = mentions.strip()
    if has_explicit_mentions or not value or value == PIVO_FALLBACK_MENTIONS_TEXT:
        return ""
    template = _choice(PIVO_NOTIFICATION_LINES, rng)
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


# Localized day-prefixes: "<en/ru day marker> <time tail>" -> "<ru day> <tail>".
# The tail offset is always len(prefix), so it need not be stored. Order matters:
# the longer "... at "/"... в " forms must precede their bare-space variants.
_TIME_DAY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("today at ", "сегодня"),
    ("today ", "сегодня"),
    ("tomorrow at ", "завтра"),
    ("tomorrow ", "завтра"),
    ("сегодня в ", "сегодня"),
    ("завтра в ", "завтра"),
)
# Standalone day/part-of-day words with a fixed localized form.
_TIME_EXACT_WORDS: dict[str, str] = {
    "today": "сегодня",
    "tomorrow": "завтра",
    "сегодня": "сегодня",
    "завтра": "завтра",
    "evening": "вечером",
    "вечером": "вечером",
}


def _format_time_value(planned_time: str) -> str:
    raw = planned_time.strip()
    normalized = raw.lower()

    if re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", normalized):
        return html.escape(raw, quote=False)
    for prefix, day_word in _TIME_DAY_PREFIXES:
        if normalized.startswith(prefix):
            return f"{day_word} {_normalize_time_tail(raw[len(prefix):])}"
    exact = _TIME_EXACT_WORDS.get(normalized)
    if exact is not None:
        return exact
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
