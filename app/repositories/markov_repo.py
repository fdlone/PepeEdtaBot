from __future__ import annotations

import aiosqlite

from app.core.temporal import TransitionRow
from app.repositories.base_repo import BaseRepo


def _transition_row(row: aiosqlite.Row) -> TransitionRow:
    """One pool row with both layers (M2R-200).

    ``s_updated_at`` stays None for rows that predate migration 018 or have not
    been observed since it ran — the reader treats that as an empty short layer
    rather than inventing a date (design D6).
    """
    return (
        str(row[0]),
        int(row[1]),
        float(row[2] or 0.0),
        None if row[3] is None else int(row[3]),
    )


# Вынесены в константы, чтобы тест плана запроса читал тот же текст, что
# уходит в базу. Прежний гард строил EXPLAIN по рукописной копии и потому
# ничего не охранял: копия разошлась с кодом, а тест остался зелёным.
REVERSE_TRANSITIONS_SQL = """
    SELECT w1, cnt, s_value, s_updated_at
    FROM transitions INDEXED BY idx_transitions_reverse
    WHERE chat_id = ? AND w2 = ? AND w3 = ?
    ORDER BY w1
"""

REVERSE_BRANCH_SQL = (
    "SELECT COUNT(*) FROM (SELECT DISTINCT w1 FROM transitions "
    "INDEXED BY idx_transitions_reverse WHERE chat_id = ? AND w2 = ?)"
)


class MarkovRepo(BaseRepo):
    """Read-only доступ к таблицам starts/transitions/transitions3/messages."""

    async def get_recent_normalized(self, chat_id: int, limit: int) -> list[str]:
        rows = await self._fetch_all(
            """
            SELECT normalized_text
            FROM messages
            WHERE chat_id = ? AND normalized_text != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        return [str(row[0]) for row in reversed(rows)]

    async def get_starts(self, chat_id: int) -> list[tuple[str, str, int]]:
        rows = await self._fetch_all(
            """
            SELECT w1, w2, cnt
            FROM starts
            WHERE chat_id = ?
            ORDER BY w1, w2
            """,
            (chat_id,),
        )
        return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]

    async def get_starts3(self, chat_id: int) -> list[tuple[str, str, str, int]]:
        rows = await self._fetch_all(
            """
            SELECT w1, w2, w3, cnt
            FROM starts3
            WHERE chat_id = ?
            ORDER BY w1, w2, w3
            """,
            (chat_id,),
        )
        return [(str(r[0]), str(r[1]), str(r[2]), int(r[3])) for r in rows]

    async def get_start_if_exists(
        self, chat_id: int, w1: str, w2: str
    ) -> tuple[str, str, int] | None:
        row = await self._fetch_one(
            "SELECT w1, w2, cnt FROM starts WHERE chat_id = ? AND w1 = ? AND w2 = ?",
            (chat_id, w1, w2),
        )
        if not row:
            return None
        return str(row[0]), str(row[1]), int(row[2])

    async def get_start3_if_exists(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> tuple[str, str, str, int] | None:
        row = await self._fetch_one(
            """
            SELECT w1, w2, w3, cnt
            FROM starts3
            WHERE chat_id = ? AND w1 = ? AND w2 = ? AND w3 = ?
            """,
            (chat_id, w1, w2, w3),
        )
        if not row:
            return None
        return str(row[0]), str(row[1]), str(row[2]), int(row[3])

    async def get_transitions(
        self, chat_id: int, w1: str, w2: str
    ) -> list[TransitionRow]:
        rows = await self._fetch_all(
            """
            SELECT w3, cnt, s_value, s_updated_at
            FROM transitions
            WHERE chat_id = ? AND w1 = ? AND w2 = ?
            ORDER BY w3
            """,
            (chat_id, w1, w2),
        )
        return [_transition_row(r) for r in rows]

    async def get_transitions3(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> list[TransitionRow]:
        rows = await self._fetch_all(
            """
            SELECT w4, cnt, s_value, s_updated_at
            FROM transitions3
            WHERE chat_id = ? AND w1 = ? AND w2 = ? AND w3 = ?
            ORDER BY w4
            """,
            (chat_id, w1, w2, w3),
        )
        return [_transition_row(r) for r in rows]

    async def get_reverse_transitions(
        self, chat_id: int, w2: str, w3: str
    ) -> list[TransitionRow]:
        """Which tokens preceded state ``(w2, w3)``, with the temporal record.

        M2R-400 (TZ §9.2, design D1): a reverse order-2 transition IS the
        forward row read by its last two columns. No production caller until
        seeded generation (M2R-410) lands behind its own gate.

        ``INDEXED BY`` — не оптимизация, а принуждение. Утверждение «served by
        ``idx_transitions_reverse``, never a scan`` стояло здесь с миграции 020
        и было неверным: в базе нет ``sqlite_stat1`` (``ANALYZE`` не
        выполняется нигде в проекте), и без статистики SQLite предпочитает
        PK-индекс — скан всего чата в порядке ``w1`` бесплатно удовлетворяет
        ``ORDER BY``. Замер на прод-копии 2026-08-26: 1.42 мс против 0.036 мс
        с индексом, то есть 40×. Результат идентичен — при фиксированных
        ``(chat_id, w2, w3)`` значение ``w1`` уникально по первичному ключу,
        поэтому ``ORDER BY w1`` остаётся тотальным. Гейт ``generation_hash``
        такую разницу не видит по построению: байты те же, отличается время.
        """
        rows = await self._fetch_all(REVERSE_TRANSITIONS_SQL, (chat_id, w2, w3))
        return [_transition_row(r) for r in rows]

    async def get_seed_forward(
        self, chat_id: int, token: str
    ) -> list[tuple[str, int]]:
        """Continuations of ``token`` as a first token: ``(w2, total_cnt)``.

        M2R-410 tail bootstrap: a seed is a single token, but the chain's state
        is a pair — this picks the second token to start the tail's order-2
        walk from ``(token, w2)``. Long counts only (the temporal blend applies
        to the per-step walk, not to this one bootstrap draw). Served by the
        primary-key prefix ``(chat_id, w1)``. ``len`` is the seed's forward
        branching for the seed score.
        """
        rows = await self._fetch_all(
            """
            SELECT w2, SUM(cnt)
            FROM transitions
            WHERE chat_id = ? AND w1 = ?
            GROUP BY w2
            ORDER BY w2
            """,
            (chat_id, token),
        )
        return [(str(r[0]), int(r[1])) for r in rows]

    async def get_reverse_branch(self, chat_id: int, token: str) -> int:
        """How many distinct tokens the seed can be preceded by (M2R-410).

        The seed's reverse branching for the seed score: distinct ``w1`` over
        rows where the seed is the pair's first member (``w2 = token``).

        ``INDEXED BY`` по тем же основаниям, что у ``get_reverse_transitions``:
        без ``sqlite_stat1`` планировщик выбирал PK-индекс и сканировал чат
        целиком. Замер на прод-копии 2026-08-26: 1.16 мс против 0.086 мс, 13×.
        Этот запрос зовётся по одному разу на каждый eligible-токен входящего
        сообщения (``rank_seeds``), поэтому цена умножается на длину
        сообщения — при включении ``markov_seeded_candidate_ratio`` (M2R-430)
        это десятки миллисекунд из бюджета p95 150 мс.
        """
        rows = await self._fetch_all(REVERSE_BRANCH_SQL, (chat_id, token))
        return int(rows[0][0]) if rows else 0

    async def get_token_df(self, chat_id: int, token: str) -> int:
        """In how many learned messages the token appeared (M2R-400, TZ §9.3)."""
        rows = await self._fetch_all(
            "SELECT messages_seen FROM markov_token_df "
            "WHERE chat_id = ? AND token = ?",
            (chat_id, token),
        )
        return int(rows[0][0]) if rows else 0

    async def get_n_docs(self, chat_id: int) -> int:
        """Total learned messages of the chat — the IDF denominator's N."""
        rows = await self._fetch_all(
            "SELECT n_docs FROM chat_model_volume WHERE chat_id = ?",
            (chat_id,),
        )
        return int(rows[0][0]) if rows else 0

    async def get_states(
        self,
        chat_id: int,
        order: int,
    ) -> list[tuple[tuple[str, ...], int]]:
        if order not in {2, 3}:
            raise ValueError("order must be 2 or 3")

        table, columns = (
            ("transitions3", "w1, w2, w3") if order == 3 else ("transitions", "w1, w2")
        )
        # table/columns are picked from the two literals above, never from a caller.
        rows = await self._fetch_all(
            f"""
            SELECT {columns}, SUM(cnt)
            FROM {table}
            WHERE chat_id = ?
            GROUP BY {columns}
            ORDER BY {columns}
            """,  # nosec B608
            (chat_id,),
        )
        return [
            (
                tuple(str(row[index]) for index in range(order)),
                int(row[order]),
            )
            for row in rows
        ]

    async def get_word_frequencies(
        self, chat_id: int, *, min_word_len: int
    ) -> dict[str, int]:
        """Occurrence counts of chat words over the whole model memory.

        Counted as continuations (``w3``) of the order-2 chain, so every
        learned token occurrence except the two openers of each message is
        represented; short tokens (punctuation, particles) are trimmed in SQL
        to keep the result compact.

        Ordered by word: this read feeds slot-mutation sampling, and every read
        that feeds a draw carries a total order (``generation-sampling-
        determinism``). ``w3`` is the grouping key, so ordering by it is total.

        The clause is a contract pin, not a fix: SQLite already returns grouped
        rows in group-key order under both plans this query gets (temp B-tree
        and index scan), so adding it changes no result today. It is here so a
        later rewrite of the query cannot silently drop the property — and
        because the property was never actually the divergence. The warm cache
        appends newly learned words at the end while this read is sorted, and
        that gap is closed where the order becomes a draw, in
        ``pick_replacement``.
        """
        rows = await self._fetch_all(
            """
            SELECT w3, SUM(cnt)
            FROM transitions
            WHERE chat_id = ? AND LENGTH(w3) >= ?
            GROUP BY w3
            ORDER BY w3
            """,
            (chat_id, min_word_len),
        )
        return {str(r[0]): int(r[1]) for r in rows}

    async def get_model_volume(self, chat_id: int) -> tuple[int, int] | None:
        """``(volume2, volume3)`` из счётчика, или None, если строки ещё нет.

        Счётчик поддерживается инкрементально на записи
        (``save_message_and_update_model``) и сбрасывается вместе с чатом в
        ``clear_chat``, поэтому он точно равен ``SUM(cnt)`` по таблицам
        переходов — что и проверяется тестом.
        """
        row = await self._fetch_one(
            "SELECT volume2, volume3 FROM chat_model_volume WHERE chat_id = ?",
            (chat_id,),
        )
        if row is None:
            return None
        return int(row[0] or 0), int(row[1] or 0)

    async def sum_model_volume(self, chat_id: int) -> tuple[int, int]:
        """``(volume2, volume3)`` полным агрегатом по таблицам переходов."""
        async with self._connection() as db:
            volume2 = await self._sum_cnt(db, "transitions", chat_id)
            volume3 = await self._sum_cnt(db, "transitions3", chat_id)
        return volume2, volume3

    async def backfill_model_volume(self, chat_id: int) -> tuple[int, int]:
        """Считает объём агрегатом и заводит строку счётчика для чата.

        Нужно ровно один раз на чат — для тех, кто набрал данные до появления
        счётчика (миграция 009 сделала бэкфилл, но чат мог появиться и позже
        по другому пути) и для чата, у которого строки ещё нет.
        """
        async with self._transaction() as db:
            volume2 = await self._sum_cnt(db, "transitions", chat_id)
            volume3 = await self._sum_cnt(db, "transitions3", chat_id)
            await db.execute(
                """
                INSERT INTO chat_model_volume (chat_id, volume2, volume3)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO NOTHING
                """,
                (chat_id, volume2, volume3),
            )
        return volume2, volume3

    @staticmethod
    async def _sum_cnt(
        db: aiosqlite.Connection, table: str, chat_id: int
    ) -> int:
        # table is a hardcoded caller constant, never user input.
        cursor = await db.execute(
            f"SELECT COALESCE(SUM(cnt), 0) FROM {table} WHERE chat_id = ?",  # nosec B608
            (chat_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("COALESCE query returned None in get_chat_token_volume")
        return int(row[0] or 0)
