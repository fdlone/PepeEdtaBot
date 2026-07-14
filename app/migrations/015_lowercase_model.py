"""Casefolds the legacy mixed-case model data (NORMALIZE_LOWER backfill).

``NORMALIZE_LOWER`` only lowercases tokens on the way *in* (see
``tokenize``), so chats that learned before the flag was turned on keep both
"Пиво" and "пиво" as separate states: the chain still walks into the
capitalized copies, and the counts of one word are split across two rows.
This one-off backfill lowercases every token already in the model and merges
the resulting duplicates by summing ``cnt``.

Note on ``lower()``: SQLite's built-in is ASCII-only and would leave Cyrillic
untouched, which is exactly the data we need to fix — hence the Python
function registered below.

Chats that keep ``NORMALIZE_LOWER=false`` will accumulate mixed case again;
the flag defaults to true precisely so the config and the stored model agree.
"""
from __future__ import annotations

import aiosqlite

# (table, key columns) — cnt is summed over the lowercased key.
_MODEL_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("starts", ("w1", "w2")),
    ("transitions", ("w1", "w2", "w3")),
    ("starts3", ("w1", "w2", "w3")),
    ("transitions3", ("w1", "w2", "w3", "w4")),
)


def _lower(value: str | None) -> str | None:
    return value.lower() if value is not None else None


async def apply(conn: aiosqlite.Connection) -> None:
    await conn.create_function("py_lower", 1, _lower, deterministic=True)

    for table, key_columns in _MODEL_TABLES:
        keys = ", ".join(f"py_lower({column}) AS {column}" for column in key_columns)
        group_by = ", ".join(f"py_lower({column})" for column in key_columns)
        columns = ", ".join(key_columns)
        await conn.execute(
            f"""
            CREATE TEMP TABLE _lc_{table} AS
            SELECT chat_id, {keys}, SUM(cnt) AS cnt
            FROM {table}
            GROUP BY chat_id, {group_by}
            """  # nosec B608 - table/columns come from the _MODEL_TABLES literal
        )
        await conn.execute(f"DELETE FROM {table}")  # nosec B608 - literal table name
        await conn.execute(
            f"""
            INSERT INTO {table}(chat_id, {columns}, cnt)
            SELECT chat_id, {columns}, cnt FROM _lc_{table}
            """  # nosec B608 - table/columns come from the _MODEL_TABLES literal
        )
        await conn.execute(f"DROP TABLE _lc_{table}")

    # Hot n-grams carry an updated_at column: the merged row keeps the most
    # recent timestamp so the sliding-window decay still expires it correctly.
    await conn.execute(
        """
        CREATE TEMP TABLE _lc_hot AS
        SELECT
            chat_id,
            py_lower(w1) AS w1,
            py_lower(w2) AS w2,
            py_lower(w3) AS w3,
            SUM(cnt) AS cnt,
            MAX(updated_at) AS updated_at
        FROM chat_hot_ngrams
        GROUP BY chat_id, py_lower(w1), py_lower(w2), py_lower(w3)
        """
    )
    await conn.execute("DELETE FROM chat_hot_ngrams")
    await conn.execute(
        """
        INSERT INTO chat_hot_ngrams(chat_id, w1, w2, w3, cnt, updated_at)
        SELECT chat_id, w1, w2, w3, cnt, updated_at FROM _lc_hot
        """
    )
    await conn.execute("DROP TABLE _lc_hot")

    # messages.normalized_text is deliberately left as-is: it never reaches a
    # reply, and its readers (verbatim check, quoting penalty, token IDF) already
    # casefold both sides. Lowercasing it would only desync the stored corpus
    # from what save_message_and_update_model keeps writing.
