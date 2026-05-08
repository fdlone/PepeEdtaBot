"""Zeroes out any non-anonymized author_id values left from pre-migration data."""
from __future__ import annotations

import aiosqlite


async def apply(conn: aiosqlite.Connection) -> None:
    await conn.execute("UPDATE messages SET author_id = 0 WHERE author_id != 0")
