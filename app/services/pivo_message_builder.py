from __future__ import annotations

import html
import random
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime

from app.domain.pivo import PIVO_FALLBACK_MENTIONS
from app.domain.pivo_templates import (
    PIVO_AUTUMN_BOTTOM_PARTS,
    PIVO_DEFAULT_BODY_PARTS,
    PIVO_DEFAULT_BOTTOM_PARTS,
    PIVO_DEFAULT_TARGET_INTROS,
    PIVO_DEFAULT_TOP_PARTS,
    PIVO_FRIDAY_BOTTOM_PARTS,
    PIVO_LATE_NIGHT_BOTTOM_PARTS,
    PIVO_MONDAY_BOTTOM_PARTS,
    PIVO_NOTIFICATION_LINES,
    PIVO_SPRING_BOTTOM_PARTS,
    PIVO_SUB_POOLS,
    PIVO_SUMMER_BOTTOM_PARTS,
    PIVO_TARGET_BODY_PARTS,
    PIVO_TARGET_BOTTOM_PARTS,
    PIVO_TARGET_INTROS,
    PIVO_TARGET_TOP_PARTS,
    PIVO_WINTER_BOTTOM_PARTS,
)

DEFAULT_TARGET_PHRASE = "пиво, игры и коллективная деградация"


def format_truncated_mentions_note(omitted: int) -> str:
    """Приписка о том, что упомянуты не все подписчики.

    Предел на число упоминаний защищает чат от рассылки на весь список, но
    молчаливое усечение выглядело бы как потерянные люди. Формулировка без
    согласования по числу — числительное здесь произвольное.
    """
    return f"(упомянуты не все — ещё {omitted} в списке подписчиков)"


@dataclass(frozen=True, slots=True)
class PivoMessageContext:
    # Optional mention block glued to the vocative a top template already has:
    # either "" or a mention list carrying its own leading space (" @a @b").
    # See the invariants documented in app/domain/pivo_templates.py.
    mentions_inline: str
    notification_line: str
    time_phrase: str
    time_phrase_soft: str
    target_phrase: str
    target_bullet: str
    has_explicit_target: bool

    def template_values(self) -> dict[str, object]:
        return asdict(self)


def build_pivo_message_context(
    mentions: str,
    *,
    planned_time: str | None,
    target: str | None,
    has_explicit_mentions: bool,
    rng: random.Random | None = None,
) -> PivoMessageContext:
    raw_target = _normalize_target(target)
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


_SUB_SLOT_RE = re.compile(r"\{([a-z_]+)\}")
# Two levels are used today (chaos_bullet -> dispute_topic -> literal); the
# third is headroom. The cap only matters if someone writes a reference cycle.
_SUB_SLOT_MAX_DEPTH = 3


def _expand_sub_slots(text: str, rng: random.Random | None) -> str:
    """Expand ``{sub_pool}`` slots recursively, leaving context slots alone.

    Runs BEFORE the final ``str.format`` with context values, so user-supplied
    text (target_phrase and friends) is never rescanned for slots — braces
    typed by a user stay literal. Only names present in PIVO_SUB_POOLS are
    touched; anything else ("{target_bullet}") passes through untouched for the
    context substitution. Each replacement may itself contain sub-pool slots;
    expansion is depth-capped, and any slot still unresolved at the cap (an
    authoring mistake: a reference cycle) is dropped rather than leaked into
    the message or the final format call.
    """

    replaced = False

    def _substitute(match: re.Match[str]) -> str:
        nonlocal replaced
        pool = PIVO_SUB_POOLS.get(match.group(1))
        if pool is None:
            return match.group(0)
        replaced = True
        return _choice(pool, rng)

    for _ in range(_SUB_SLOT_MAX_DEPTH):
        replaced = False
        text = _SUB_SLOT_RE.sub(_substitute, text)
        # Comparing texts instead would exit early on a self-cycle ("{a}" ->
        # "{a}" reads as "nothing changed") and leak the slot past the cap.
        if not replaced:
            return text
    return _SUB_SLOT_RE.sub(
        lambda m: "" if m.group(1) in PIVO_SUB_POOLS else m.group(0), text
    )


# Meteorological seasons: month -> seasonal closing-line pool.
_SEASON_BOTTOM_POOLS: dict[int, tuple[str, ...]] = {
    12: PIVO_WINTER_BOTTOM_PARTS, 1: PIVO_WINTER_BOTTOM_PARTS,
    2: PIVO_WINTER_BOTTOM_PARTS,
    3: PIVO_SPRING_BOTTOM_PARTS, 4: PIVO_SPRING_BOTTOM_PARTS,
    5: PIVO_SPRING_BOTTOM_PARTS,
    6: PIVO_SUMMER_BOTTOM_PARTS, 7: PIVO_SUMMER_BOTTOM_PARTS,
    8: PIVO_SUMMER_BOTTOM_PARTS,
    9: PIVO_AUTUMN_BOTTOM_PARTS, 10: PIVO_AUTUMN_BOTTOM_PARTS,
    11: PIVO_AUTUMN_BOTTOM_PARTS,
}


def _temporal_bottoms(now: datetime) -> tuple[str, ...]:
    """Collect the time-aware closing lines that apply at ``now``.

    Day/hour buckets may be empty; the seasonal pool never is, so with a
    positive flavor chance there is always at least one temporal candidate.
    Rarer buckets (night, friday/monday) stay ahead of the season only by
    contributing their own entries to the same draw.
    """
    pools: list[str] = []
    if 0 <= now.hour < 6:
        pools.extend(PIVO_LATE_NIGHT_BOTTOM_PARTS)
    weekday = now.weekday()
    if weekday == 4:
        pools.extend(PIVO_FRIDAY_BOTTOM_PARTS)
    elif weekday == 0:
        pools.extend(PIVO_MONDAY_BOTTOM_PARTS)
    pools.extend(_SEASON_BOTTOM_POOLS[now.month])
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
        template = _expand_sub_slots(template, rng)
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


_TARGET_TRAILING_PUNCTUATION = " .,;:!?…"


def _normalize_target(target: str | None) -> str:
    """Collapse whitespace and strip trailing sentence punctuation off the target.

    The templates bring their own punctuation around the slot ("Цель заявлена
    так: {target_phrase}.", "{target_bullet};"), so a target typed as "го в
    дотку!" would otherwise come out as "го в дотку!.". A target made of nothing
    but punctuation collapses to "" and the message falls back to default mode.
    """
    if target is None:
        return ""
    collapsed = re.sub(r"\s+", " ", target).strip()
    return collapsed.rstrip(_TARGET_TRAILING_PUNCTUATION)


def _build_target_bullet(
    *, target_phrase: str, has_explicit_target: bool, rng: random.Random | None = None
) -> str:
    if has_explicit_target:
        return _choice(PIVO_TARGET_INTROS, rng).format(target_phrase=target_phrase)
    return _choice(PIVO_DEFAULT_TARGET_INTROS, rng)


def _build_mentions_inline(mentions: str, *, has_explicit_mentions: bool) -> str:
    """Render the inline mention block: " @a @b", or nothing at all.

    Only explicitly mentioned users go inline; subscribers are pinged by the
    separate notification line. Without explicit mentions the slot collapses to
    an empty string and the top template falls back to its own vocative.

    The slot used to be filled with a noun phrase ("подозрительные личности")
    whenever mentions were absent, but the surrounding sentence governs that
    slot — as apposition, accusative or dative — so a noun phrase came out in
    the wrong case ("Дорогой конченый коллектив подозрительные личности") or
    doubled the template's own vocative ("Так, конченые конченый состав чата").
    A mention list is grammatically opaque and has no such problem.
    """
    if not has_explicit_mentions:
        return ""
    value = mentions.strip()
    return f" {value}" if value else ""


def _build_notification_line(
    mentions: str, *, has_explicit_mentions: bool, rng: random.Random | None = None
) -> str:
    value = mentions.strip()
    if has_explicit_mentions or not value or value == PIVO_FALLBACK_MENTIONS:
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
    """Vague arrival time. Templates place it sentence-final, so the no-time
    default must open with an adverbial ("ближе к вечеру") rather than with the
    bare subordinate clause it used to be: "сбор в Discord когда коллектив ..."
    was missing the comma its "когда" required."""
    if not planned_time:
        return "ближе к вечеру, как только все перестанут изображать занятых людей"
    value = _format_time_value(planned_time)
    # A day word already reads as an approximation ("где-то завтра вечером");
    # only a bare clock time takes "примерно в".
    if _time_value_has_own_preposition(value):
        return f"где-то {value}"
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
    # Голые русские формы. Парсер принимает их наравне с остальными
    # (`pivo_parser.py`, паттерн `^(?:сегодня|завтра)\\s+HH:MM`), а таблица их
    # не знала — и «/pivo завтра 19:00» давало «собираемся завтра 19:00» без
    # предлога, тогда как «/pivo tomorrow 19:00» давало правильное «завтра в
    # 19:00». Английский ввод был обслужен лучше русского в русском чате.
    # Стоят последними: длинные формы «... в » обязаны проверяться раньше,
    # иначе они схлопнутся сюда и предлог продублируется.
    ("сегодня ", "сегодня"),
    ("завтра ", "завтра"),
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
