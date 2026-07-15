"""Cumulative content-4-gram index for the verbatim layer (closes O4).

The anti-quote machinery (scorer's verbatim penalty, the near-quote extension
trigger) used to build its 4-gram index from the ``messages`` window (last
1000 texts), while the Markov model itself accumulates forever — so the chain
could replay anything older than the window and the penalty was blind to it.
Measured 2026-07-15: the window covered ~5k of the chain's ~19.5k 4-grams
(~75% blind), a third of replies were pure all-time-corpus recombinations.

This migration creates ``chat_verbatim_ngrams`` — one row per distinct
casefolded content 4-gram ever learned per chat, same lifecycle as the
transition tables (never trimmed by message retention, wiped by /clear) —
and backfills it from two sources:

1. ``transitions3``: every stored quadruple without punctuation tokens is a
   contiguous content 4-gram of some past message (all-time memory; windows
   that crossed punctuation are unrecoverable — a lower bound by design).
2. ``messages``: content windows of the current retention window, including
   the punctuation-crossing ones the transitions cannot provide.

Privacy: the table stores a *subset* of n-grams already persisted in
``transitions3``, keyed the same way — no new information is retained.
"""
from __future__ import annotations

import re

import aiosqlite

# Mirrors app.core.markov.TOKEN_RE / PUNCT_SET (kept inline: migrations are
# frozen history and must not drift with the app code they backfilled for).
_TOKEN_RE = re.compile(r"\w+|[.,!?;:]", re.UNICODE)
_PUNCT = {".", ",", "!", "?", ";", ":"}
_NGRAM_SIZE = 4


def _content_windows(text: str) -> list[tuple[str, str, str, str]]:
    content = [
        token.casefold() for token in _TOKEN_RE.findall(text) if token not in _PUNCT
    ]
    return [
        (content[i], content[i + 1], content[i + 2], content[i + 3])
        for i in range(len(content) - _NGRAM_SIZE + 1)
    ]


async def apply(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_verbatim_ngrams (
            chat_id INTEGER NOT NULL,
            w1 TEXT NOT NULL,
            w2 TEXT NOT NULL,
            w3 TEXT NOT NULL,
            w4 TEXT NOT NULL,
            PRIMARY KEY (chat_id, w1, w2, w3, w4)
        ) WITHOUT ROWID
        """
    )

    # Source 1: the chain's whole memory (quadruples without punctuation).
    cursor = await conn.execute(
        "SELECT chat_id, w1, w2, w3, w4 FROM transitions3"
    )
    rows = await cursor.fetchall()
    await conn.executemany(
        """
        INSERT OR IGNORE INTO chat_verbatim_ngrams(chat_id, w1, w2, w3, w4)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (chat_id, w1.casefold(), w2.casefold(), w3.casefold(), w4.casefold())
            for chat_id, w1, w2, w3, w4 in rows
            if not any(w in _PUNCT for w in (w1, w2, w3, w4))
        ],
    )

    # Source 2: the current message window (adds punctuation-crossing windows).
    cursor = await conn.execute(
        "SELECT chat_id, normalized_text FROM messages "
        "WHERE normalized_text IS NOT NULL AND normalized_text != ''"
    )
    messages = await cursor.fetchall()
    await conn.executemany(
        """
        INSERT OR IGNORE INTO chat_verbatim_ngrams(chat_id, w1, w2, w3, w4)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (chat_id, *window)
            for chat_id, text in messages
            for window in _content_windows(text)
        ],
    )
