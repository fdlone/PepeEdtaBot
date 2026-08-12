"""Build a temporal snapshot from a snapshot's retained messages (doc 05 §1.1).

Phase 3 gives the chain observation times, but only going forward: historical
``first_seen`` values are unrecoverable and rebuilding the live chain is
forbidden (ADR-005). Waiting weeks for a live accumulation window before Phase 3
can be measured at all was the alternative; this builds the measurable corpus
instead, by replaying the messages the snapshot still retains — which do carry
real ``created_at`` values — through the real learning arithmetic into a
SEPARATE database.

What this is NOT, stated here because a reader of the report has to know:

- It is not the frozen C0 corpus. The retention window keeps ~1000 messages
  while the chain holds tens of thousands of transitions, so the fixture's chain
  is an order of magnitude smaller. Deltas are only ever valid *within* the
  fixture.
- It is not live accumulation. The "historical" slice is only as old as the
  retention window, so the fresh/historical split is a construction with a known
  bias, not an observation.
- It never touches the source. The source is opened read-only and the replay
  writes to a new file.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.markov import tokenize
from app.infrastructure.database import Database

# A token counts as "fresh" when the corpus first saw it inside this window.
# 14 days is two half-lives at the default 3-day setting scaled to the retention
# window's span — long enough that the fresh vocabulary is not a handful of
# tokens, short enough that it is not most of the corpus. Reported in every run.
DEFAULT_FRESH_DAYS = 14.0


@dataclass(frozen=True, slots=True)
class FixtureInfo:
    """Provenance of a reconstructed temporal snapshot, printed in the report."""

    messages: int
    first_moment: int
    last_moment: int
    fresh_days: float
    fresh_tokens: frozenset[str]
    vocabulary: int

    @property
    def span_days(self) -> float:
        return (self.last_moment - self.first_moment) / 86_400.0

    def describe(self) -> str:
        first = datetime.fromtimestamp(self.first_moment, UTC).date()
        last = datetime.fromtimestamp(self.last_moment, UTC).date()
        return (
            f"reconstructed temporal snapshot: {self.messages} replayed messages "
            f"spanning {first}..{last} ({self.span_days:.0f} days); fresh slice = "
            f"tokens first seen in the last {self.fresh_days:.0f} days = "
            f"{len(self.fresh_tokens)} of {self.vocabulary} vocabulary tokens. "
            "Built from the retention window, NOT from live accumulation: its C0 "
            "is not the frozen baseline and deltas are only valid within this "
            "snapshot."
        )


def _parse_created_at(value: str) -> int | None:
    """SQLite ``datetime('now')`` text to Unix seconds; None if unparseable."""
    try:
        return int(
            datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=UTC)
            .timestamp()
        )
    except (ValueError, AttributeError):
        return None


def read_source_messages(
    source: Path, chat_id: int
) -> list[tuple[int, str]]:
    """``(moment, text)`` for one chat, oldest first, read-only."""
    uri = f"file:{source.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            "SELECT created_at, normalized_text FROM messages "
            "WHERE chat_id = ? AND normalized_text != '' ORDER BY id",
            (chat_id,),
        ).fetchall()
    finally:
        connection.close()
    parsed = []
    for created_at, text in rows:
        moment = _parse_created_at(str(created_at))
        if moment is not None and str(text).strip():
            parsed.append((moment, str(text)))
    parsed.sort(key=lambda item: item[0])
    return parsed


async def build_temporal_fixture(
    *,
    source: Path,
    target: Path,
    chat_id: int,
    fresh_days: float = DEFAULT_FRESH_DAYS,
    short_half_life_days: float = 3.0,
) -> FixtureInfo:
    """Replay ``source``'s retained messages into ``target`` with real times.

    Deterministic: the same source, chat and parameters produce the same
    database, because the replay is ordered by timestamp and every write takes
    its moment from the message rather than from the clock.
    """
    messages = read_source_messages(source, chat_id)
    if len(messages) < 10:
        raise ValueError(
            f"chat {chat_id} has too few retained messages to replay: {len(messages)}"
        )
    if target.exists():
        target.unlink()

    first_seen_by_token: dict[str, int] = {}
    db = Database(str(target))
    await db.init()
    try:
        for moment, text in messages:
            tokens = tokenize(text)
            if len(tokens) < 2:
                continue
            await db.save_message_and_update_model(
                chat_id=chat_id,
                raw_text=text,
                tokens=tokens,
                now=moment,
                short_half_life_days=short_half_life_days,
            )
            for token in tokens:
                folded = token.casefold()
                if folded not in first_seen_by_token:
                    first_seen_by_token[folded] = moment
    finally:
        await db.close()

    last_moment = messages[-1][0]
    cutoff = last_moment - int(fresh_days * 86_400)
    fresh = frozenset(
        token for token, first in first_seen_by_token.items() if first >= cutoff
    )
    return FixtureInfo(
        messages=len(messages),
        first_moment=messages[0][0],
        last_moment=last_moment,
        fresh_days=fresh_days,
        fresh_tokens=fresh,
        vocabulary=len(first_seen_by_token),
    )


def fresh_share(reply_content: tuple[str, ...], fresh_tokens: frozenset[str]) -> float | None:
    """§3.5 ``freshness_reflection`` for one generation.

    Share of the reply's content tokens that belong to the fresh slice. ``None``
    for an empty reply — a generation that produced nothing is not evidence of
    zero freshness.
    """
    if not reply_content:
        return None
    hits = sum(1 for token in reply_content if token in fresh_tokens)
    return hits / len(reply_content)
