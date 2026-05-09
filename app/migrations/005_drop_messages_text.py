"""
Remove the raw text column from messages.

The Markov model lives in starts*/transitions* tables and does not use messages.text.
normalized_text is kept for short-message exact matching and near-repeat heuristics.

Uses ALTER TABLE DROP COLUMN (SQLite >= 3.35, Python >= 3.10).
Guards against repeated runs: skips if column is already absent.
"""
from __future__ import annotations

import aiosqlite


async def apply(conn: aiosqlite.Connection) -> None:
    cur = await conn.execute("PRAGMA table_info(messages)")
    columns = {row[1] for row in await cur.fetchall()}
    if "text" not in columns:
        return

    # Drop any index that touches the text column before dropping the column.
    cur = await conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='messages'"
    )
    for name, sql in await cur.fetchall():
        if sql and "text" in sql and "normalized_text" not in sql:
            await conn.execute(f"DROP INDEX IF EXISTS [{name}]")

    await conn.execute("ALTER TABLE messages DROP COLUMN text")
