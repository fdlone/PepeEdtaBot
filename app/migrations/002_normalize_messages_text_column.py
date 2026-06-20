"""
Adds normalized_text column to messages if absent (pre-existing DBs),
then backfills any rows where it is still empty.
"""
from __future__ import annotations

import aiosqlite

from app.core.text import sanitize_text


async def apply(conn: aiosqlite.Connection) -> None:
    cursor = await conn.execute("PRAGMA table_info(messages)")
    columns = {str(row[1]) for row in await cursor.fetchall()}
    if "normalized_text" not in columns:
        await conn.execute(
            "ALTER TABLE messages ADD COLUMN normalized_text TEXT NOT NULL DEFAULT ''"
        )

    cursor = await conn.execute(
        "SELECT id, text FROM messages WHERE normalized_text = ''"
    )
    rows = await cursor.fetchall()
    if rows:
        await conn.executemany(
            "UPDATE messages SET normalized_text = ? WHERE id = ?",
            [(sanitize_text(str(text)), int(row_id)) for row_id, text in rows],
        )

    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_normalized_lookup"
        " ON messages(chat_id, normalized_text)"
    )
