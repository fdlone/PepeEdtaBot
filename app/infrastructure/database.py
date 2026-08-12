from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import TypeVar

import aiosqlite

from app.config.defaults import (
    MESSAGES_RETENTION_PER_CHAT,
    SQLITE_BUSY_TIMEOUT_MS,
    SQLITE_WAL_AUTOCHECKPOINT_PAGES,
)
from app.core.candidate_scorer import verbatim_ngram_windows
from app.core.temporal import short_observed
from app.core.text import sanitize_text
from app.infrastructure import migrator
from app.log_masking import mask_chat_ids_in_text
from app.repositories import (
    ChatEmojiStatsRepo,
    ChatHotNgramsRepo,
    ChatMembersRepo,
    ChatUserInteractionsRepo,
    MarkovRepo,
    MessagesRepo,
    PivoPoolUsageRepo,
    PivoUsageRepo,
)

PIVO_DAILY_USAGE_RETENTION_DAYS = 7
CHAT_EMOJI_DECAY_DAYS = 7
CHAT_HOT_NGRAM_DECAY_DAYS = 7
# L2: окно «завсегдатая» намеренно медленнее мемных (7d) — недельный отпуск
# не должен разжаловать постоянного собеседника.
CHAT_USER_INTERACTION_DECAY_DAYS = 30
# Минимальный интервал между прогонами decay эмодзи/n-грамм. Планировщика в
# проекте нет, поэтому decay перезапускается лениво с learn-пути (см.
# decay_flavor_stats_if_due): при аптайме в недели окно «горячести» продолжает
# скользить, а не замирает до рестарта.
FLAVOR_DECAY_INTERVAL_SEC = 86400.0
# Пауза после неудачного прогона. Промежуточная величина выбрана намеренно:
# сутки заморозили бы окно «горячести» на день из-за одной занятой базы, а
# повтор на следующем же сообщении при типичной причине сбоя (SQLITE_BUSY)
# стоил бы до busy_timeout на каждом statement — и всё это под общей
# блокировкой соединения.
FLAVOR_DECAY_RETRY_INTERVAL_SEC = 600.0

_RepoT = TypeVar("_RepoT")

logger = logging.getLogger("chat_markov")


class MaintenanceOutcome(enum.StrEnum):
    """Чем закончился вызов отложенного обслуживания.

    «Срок не подошёл» и «попытались и не смогли» — разные вещи для того, кто
    решает, пора ли беспокоить владельца, поэтому это не bool.
    """

    SKIPPED = "skipped"
    DONE = "done"
    FAILED = "failed"


class Database:
    """Фасад над SQLite: владеет соединением, миграциями и репозиториями."""

    def __init__(
        self,
        path: str,
        *,
        messages_retention_per_chat: int = MESSAGES_RETENTION_PER_CHAT,
        busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
        wal_autocheckpoint_pages: int = SQLITE_WAL_AUTOCHECKPOINT_PAGES,
    ) -> None:
        if messages_retention_per_chat < 1:
            raise ValueError("messages_retention_per_chat must be at least 1")
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        if wal_autocheckpoint_pages < 0:
            raise ValueError("wal_autocheckpoint_pages must be non-negative")
        self.path = path
        self.messages_retention_per_chat = messages_retention_per_chat
        self.busy_timeout_ms = busy_timeout_ms
        self.wal_autocheckpoint_pages = wal_autocheckpoint_pages
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._markov: MarkovRepo | None = None
        self._messages: MessagesRepo | None = None
        self._chat_members: ChatMembersRepo | None = None
        self._pivo_usage: PivoUsageRepo | None = None
        self._pivo_pool_usage: PivoPoolUsageRepo | None = None
        self._chat_emoji_stats: ChatEmojiStatsRepo | None = None
        self._chat_hot_ngrams: ChatHotNgramsRepo | None = None
        self._chat_user_interactions: ChatUserInteractionsRepo | None = None
        # Monotonic-время, раньше которого следующий прогон decay не начинается
        # (None до init()).
        self._next_flavor_decay_monotonic: float | None = None
        self._last_maintenance_error: str | None = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not initialized. Call init() first.")
        return self._conn

    @staticmethod
    def _require(repo: _RepoT | None) -> _RepoT:
        """Возвращает репозиторий или бросает, если init() ещё не вызван."""
        if repo is None:
            raise RuntimeError(
                "Database not initialized: call await Database.init() first"
            )
        return repo

    # Репозитории отдаются свойствами, а не полями: проверка «init() вызван»
    # живёт в одной точке доступа, а тип остаётся неопциональным — вызывающему
    # не нужно ни assert, ни защита от None на каждом обращении.

    @property
    def markov(self) -> MarkovRepo:
        return self._require(self._markov)

    @property
    def messages(self) -> MessagesRepo:
        return self._require(self._messages)

    @property
    def chat_members(self) -> ChatMembersRepo:
        return self._require(self._chat_members)

    @property
    def pivo_usage(self) -> PivoUsageRepo:
        return self._require(self._pivo_usage)

    @property
    def pivo_pool_usage(self) -> PivoPoolUsageRepo:
        return self._require(self._pivo_pool_usage)

    @property
    def chat_emoji_stats(self) -> ChatEmojiStatsRepo:
        return self._require(self._chat_emoji_stats)

    @property
    def chat_hot_ngrams(self) -> ChatHotNgramsRepo:
        return self._require(self._chat_hot_ngrams)

    @property
    def chat_user_interactions(self) -> ChatUserInteractionsRepo:
        return self._require(self._chat_user_interactions)

    async def init(self) -> None:
        if self._conn is not None:
            return

        self._conn = await aiosqlite.connect(self.path)
        db = await self._get_conn()

        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms};")
        await db.execute(
            f"PRAGMA wal_autocheckpoint = {self.wal_autocheckpoint_pages};"
        )
        await db.execute("PRAGMA foreign_keys=ON;")

        await migrator.run(db)

        self._markov = MarkovRepo(self._get_conn, self._lock)
        self._messages = MessagesRepo(self._get_conn, self._lock)
        self._chat_members = ChatMembersRepo(self._get_conn, self._lock)
        self._pivo_usage = PivoUsageRepo(self._get_conn, self._lock)
        self._pivo_pool_usage = PivoPoolUsageRepo(self._get_conn, self._lock)
        self._chat_emoji_stats = ChatEmojiStatsRepo(self._get_conn, self._lock)
        self._chat_hot_ngrams = ChatHotNgramsRepo(self._get_conn, self._lock)
        self._chat_user_interactions = ChatUserInteractionsRepo(
            self._get_conn, self._lock
        )
        await self.cleanup_pivo_daily_usage()
        await self.decay_chat_emoji_stats()
        await self.decay_chat_hot_ngrams()
        await self.decay_chat_user_interactions()
        self._next_flavor_decay_monotonic = (
            time.monotonic() + FLAVOR_DECAY_INTERVAL_SEC
        )

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        self._markov = None
        self._messages = None
        self._chat_members = None
        self._pivo_usage = None
        self._pivo_pool_usage = None
        self._chat_emoji_stats = None
        self._chat_hot_ngrams = None
        self._chat_user_interactions = None

    # One statement's worth of keys. Bounded so a very long message cannot
    # approach SQLite's bound-parameter ceiling; the loop costs nothing at
    # normal message lengths, where a single chunk covers everything.
    _TEMPORAL_CHUNK = 200

    @staticmethod
    async def _upsert_temporal(
        db: aiosqlite.Connection,
        *,
        table: str,
        key_columns: tuple[str, ...],
        chat_id: int,
        entries: Sequence[tuple[tuple[str, ...], int]],
        moment: int,
        half_life_days: float,
    ) -> None:
        """Upsert both layers for one message's n-grams (M2R-200/210).

        Read-modify-write rather than pure SQL arithmetic: the short layer's new
        value depends on the row's own ``s_updated_at`` through ``2^(-Δt/hl)``,
        and SQLite's ``pow`` is a compile-time option we do not control across
        the CI matrix and the Docker image. Doing it in Python also keeps one
        implementation of the decay identity (design D1) and keeps the clock out
        of SQL, which determinism needs.

        The extra read is a primary-key lookup over the keys of a single
        message, inside the transaction and the lock that already exist.

        ``table`` and ``key_columns`` come from this module's call sites only,
        never from user input.
        """
        keys = [key for key, _ in entries]
        stored: dict[tuple[str, ...], tuple[float, int | None]] = {}
        key_list = ", ".join(key_columns)
        placeholder = f"({', '.join('?' * len(key_columns))})"
        for start in range(0, len(keys), Database._TEMPORAL_CHUNK):
            chunk = keys[start : start + Database._TEMPORAL_CHUNK]
            values = ", ".join([placeholder] * len(chunk))
            cursor = await db.execute(
                f"""
                SELECT {key_list}, s_value, s_updated_at
                FROM {table}
                WHERE chat_id = ? AND ({key_list}) IN (VALUES {values})
                """,  # nosec B608
                (chat_id, *[word for key in chunk for word in key]),
            )
            for row in await cursor.fetchall():
                arity = len(key_columns)
                stored[tuple(str(row[i]) for i in range(arity))] = (
                    float(row[arity] or 0.0),
                    None if row[arity + 1] is None else int(row[arity + 1]),
                )

        payload = []
        for key, delta in entries:
            s_value, s_updated_at = stored.get(key, (0.0, None))
            payload.append(
                (
                    chat_id,
                    *key,
                    delta,
                    moment,
                    moment,
                    short_observed(s_value, s_updated_at, moment, half_life_days, delta),
                    moment,
                )
            )
        columns = f"chat_id, {key_list}, cnt, first_seen, last_seen, s_value, s_updated_at"
        row_placeholders = ", ".join("?" * (len(key_columns) + 6))
        await db.executemany(
            f"""
            INSERT INTO {table}({columns})
            VALUES ({row_placeholders})
            ON CONFLICT(chat_id, {key_list})
            DO UPDATE SET
                cnt = cnt + excluded.cnt,
                first_seen = COALESCE(first_seen, excluded.first_seen),
                last_seen = excluded.last_seen,
                s_value = excluded.s_value,
                s_updated_at = excluded.s_updated_at
            """,  # nosec B608
            payload,
        )

    async def save_message_and_update_model(
        self,
        chat_id: int,
        raw_text: str,
        tokens: list[str],
        now: int | None = None,
        short_half_life_days: float = 3.0,
    ) -> int:
        """Атомарно сохраняет raw-сообщение и обновляет счётчики переходов n=3/n=2/n=1.

        С M2R-200 в той же транзакции ведётся и временная запись: долгий
        счётчик, ``first_seen``/``last_seen`` и затухающая пара короткого слоя
        (ТЗ §7.1). Момент наблюдения — параметр, а не ``strftime('now')`` в SQL:
        тесты и eval должны уметь задать его, а арифметика затухания живёт в
        одном месте (``app/core/temporal.py``), а не во второй реализации на
        SQL — математические функции SQLite это опция сборки.
        """
        moment = int(time.time()) if now is None else now
        starts2_pair: tuple[str, str] | None = None
        starts3_triplet: tuple[str, str, str] | None = None

        trans2_counter: Counter[tuple[str, str, str]] = Counter()
        trans3_counter: Counter[tuple[str, str, str, str]] = Counter()

        if len(tokens) >= 2:
            starts2_pair = (tokens[0], tokens[1])
        if len(tokens) >= 3:
            starts3_triplet = (tokens[0], tokens[1], tokens[2])
            trans2_counter = Counter(
                (tokens[i], tokens[i + 1], tokens[i + 2])
                for i in range(len(tokens) - 2)
            )
        if len(tokens) >= 4:
            trans3_counter = Counter(
                (tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])
                for i in range(len(tokens) - 3)
            )

        async with self._lock:
            db = await self._get_conn()

            await db.execute(
                "INSERT INTO messages(chat_id, author_id, normalized_text) VALUES (?, ?, ?)",
                (chat_id, 0, sanitize_text(raw_text)),
            )
            await db.execute(
                """
                DELETE FROM messages
                WHERE chat_id = ?
                  AND id <= (
                      SELECT id
                      FROM messages
                      WHERE chat_id = ?
                      ORDER BY id DESC
                      LIMIT 1 OFFSET ?
                  )
                """,
                (
                    chat_id,
                    chat_id,
                    self.messages_retention_per_chat,
                ),
            )

            if starts2_pair:
                await self._upsert_temporal(
                    db,
                    table="starts",
                    key_columns=("w1", "w2"),
                    chat_id=chat_id,
                    entries=[(starts2_pair, 1)],
                    moment=moment,
                    half_life_days=short_half_life_days,
                )
            if starts3_triplet:
                await self._upsert_temporal(
                    db,
                    table="starts3",
                    key_columns=("w1", "w2", "w3"),
                    chat_id=chat_id,
                    entries=[(starts3_triplet, 1)],
                    moment=moment,
                    half_life_days=short_half_life_days,
                )
            if trans2_counter:
                await self._upsert_temporal(
                    db,
                    table="transitions",
                    key_columns=("w1", "w2", "w3"),
                    chat_id=chat_id,
                    entries=list(trans2_counter.items()),
                    moment=moment,
                    half_life_days=short_half_life_days,
                )
            if trans3_counter:
                await self._upsert_temporal(
                    db,
                    table="transitions3",
                    key_columns=("w1", "w2", "w3", "w4"),
                    chat_id=chat_id,
                    entries=list(trans3_counter.items()),
                    moment=moment,
                    half_life_days=short_half_life_days,
                )

            # Cumulative verbatim index (O4): distinct casefolded content
            # 4-grams, same lifecycle as the transition tables (message
            # retention never trims it) so the quote penalty and the
            # near-quote extension see the chain's whole memory, not the
            # last-1000-messages window.
            verbatim_windows = verbatim_ngram_windows(tokens)
            if verbatim_windows:
                await db.executemany(
                    """
                    INSERT OR IGNORE INTO chat_verbatim_ngrams
                        (chat_id, w1, w2, w3, w4)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [(chat_id, *window) for window in verbatim_windows],
                )

            # Maintain the per-chat model volume incrementally instead of
            # re-summing the whole transitions tables on every message (audit
            # D2). The delta is exactly the number of transition occurrences
            # this message contributed (sum of the per-message counters).
            volume2_delta = sum(trans2_counter.values())
            volume3_delta = sum(trans3_counter.values())
            await db.execute(
                """
                INSERT INTO chat_model_volume (chat_id, volume2, volume3)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    volume2 = volume2 + excluded.volume2,
                    volume3 = volume3 + excluded.volume3
                """,
                (chat_id, volume2_delta, volume3_delta),
            )
            cursor = await db.execute(
                "SELECT volume2, volume3 FROM chat_model_volume WHERE chat_id = ?",
                (chat_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError(
                    "chat_model_volume row missing after upsert in "
                    "save_message_and_update_model"
                )
            volume2 = int(row[0] or 0)
            volume3 = int(row[1] or 0)

            await db.commit()
            return volume3 if volume3 > 0 else volume2

    # --- Объём модели чата ---

    async def get_chat_token_volume(self, chat_id: int) -> int:
        """Объём модели чата — то, по чему проверяется готовность отвечать.

        Читается из инкрементального счётчика ``chat_model_volume``, а не
        агрегатом по таблицам переходов: этот вызов происходит на КАЖДОЕ
        входящее сообщение, а полный ``SUM(cnt)`` — скан, растущий вместе с
        корпусом (на живом чате это десятки тысяч строк). Счётчик и заводился
        под это (миграция 009), но читатель остался на агрегате.
        """
        volumes = await self._model_volumes(chat_id, persist=True)
        volume2, volume3 = volumes
        return volume3 if volume3 > 0 else volume2

    async def _model_volumes(
        self, chat_id: int, *, persist: bool
    ) -> tuple[int, int]:
        """``(volume2, volume3)`` из счётчика, с падением на полный агрегат.

        ``persist`` заводит недостающую строку счётчика — это делает только
        горячий путь чтения. Диагностика (``get_stats``) остаётся read-only:
        побочная запись из диагностического вызова портила бы наблюдаемое
        состояние таблицы.
        """
        volumes = await self.markov.get_model_volume(chat_id)
        if volumes is not None:
            return volumes
        if persist:
            return await self.markov.backfill_model_volume(chat_id)
        return await self.markov.sum_model_volume(chat_id)

    # --- Суточные затухания ---

    async def decay_chat_emoji_stats(
        self,
        *,
        decay_days: int = CHAT_EMOJI_DECAY_DAYS,
        now: datetime | None = None,
    ) -> int:
        """Halve emoji counts not bumped within ``decay_days`` so dead memes fade.

        Runs at init and then lazily once per ``FLAVOR_DECAY_INTERVAL_SEC`` from
        the learn path (``decay_flavor_stats_if_due``) — no scheduler exists.
        Stale rows are halved and their clock reset; rows reaching 0 are removed.
        Returns the number of rows deleted.
        """
        cutoff = self._decay_cutoff(decay_days, now)
        return await self.chat_emoji_stats.decay_stale(cutoff)

    # --- Делегаты к ChatHotNgramsRepo (L1 running jokes) ---

    async def decay_chat_hot_ngrams(
        self,
        *,
        decay_days: int = CHAT_HOT_NGRAM_DECAY_DAYS,
        now: datetime | None = None,
    ) -> int:
        """Halve n-gram counts not bumped within ``decay_days`` so jokes get old.

        Same cadence and contract as ``decay_chat_emoji_stats`` (init + lazy
        daily re-run via ``decay_flavor_stats_if_due``). Returns the number of
        purged rows.
        """
        cutoff = self._decay_cutoff(decay_days, now)
        return await self.chat_hot_ngrams.decay_stale(cutoff)

    async def decay_chat_user_interactions(
        self,
        *,
        decay_days: int = CHAT_USER_INTERACTION_DECAY_DAYS,
        now: datetime | None = None,
    ) -> int:
        """Halve interaction counts of users quiet for ``decay_days``.

        Same cadence and contract as the other flavor decays (init + lazy
        daily re-run via ``decay_flavor_stats_if_due``), but a slower window:
        regular status should outlive a vacation, not just a meme cycle.
        Returns the number of purged rows.
        """
        cutoff = self._decay_cutoff(decay_days, now)
        return await self.chat_user_interactions.decay_stale(cutoff)

    @staticmethod
    def _decay_cutoff(decay_days: int, now: datetime | None) -> str:
        """Cutoff timestamp for decay_stale, in SQLite's datetime('now') text
        format ("YYYY-MM-DD HH:MM:SS", UTC) so the string comparison there is
        well-defined."""
        if decay_days < 0:
            raise ValueError("decay_days must be non-negative")
        current = now or datetime.now(UTC)
        return (current - timedelta(days=decay_days)).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def last_maintenance_error(self) -> str | None:
        """Текст последней ошибки обслуживания — для сигнала владельцу.

        Логи владельцу недоступны, а «база занята» и «диск кончился» требуют
        разных действий, поэтому причина доходит до него вместе с сигналом.
        """
        return self._last_maintenance_error

    async def decay_flavor_stats_if_due(self) -> MaintenanceOutcome:
        """Прокручивает decay эмодзи/n-грамм, если подошёл срок.

        Заменяет планировщик, которого в проекте нет: дешёвая монотонная
        проверка вызывается с learn-пути, а сам decay выполняется не чаще раза
        в ``FLAVOR_DECAY_INTERVAL_SEC``, так что окно «горячести» продолжает
        скользить и без рестартов. Возвращает True, если decay выполнился.

        Сбой не пробрасывается вызывающему. Обслуживание не должно стоить
        сообщения: пока этот вызов стоял внутри
        ``save_message_and_update_model``, занятая база (бэкап, ``VACUUM``,
        внешний клиент) означала не отложенное затухание, а остановку обучения
        во всех чатах сразу — и молчаливую, потому что единственным сигналом
        была строка в логе. Затухание же переживает пропущенный прогон без
        последствий: мемы поблёкнут позже.

        Следующая попытка после сбоя откладывается на
        ``FLAVOR_DECAY_RETRY_INTERVAL_SEC``, а не на сутки и не на следующее
        сообщение: сутки заморозили бы окно из-за одной занятой базы, а повтор
        на каждом сообщении при типичной причине (``SQLITE_BUSY``) стоил бы до
        busy_timeout на каждом statement под общей блокировкой соединения.
        """
        now = time.monotonic()
        if (
            self._next_flavor_decay_monotonic is not None
            and now < self._next_flavor_decay_monotonic
        ):
            return MaintenanceOutcome.SKIPPED
        try:
            await self.decay_chat_emoji_stats()
            await self.decay_chat_hot_ngrams()
            await self.decay_chat_user_interactions()
        except Exception as error:
            self._next_flavor_decay_monotonic = (
                now + FLAVOR_DECAY_RETRY_INTERVAL_SEC
            )
            self._last_maintenance_error = mask_chat_ids_in_text(str(error))
            logger.exception(
                "Суточное затухание статистики не выполнено; повтор через %.0f с",
                FLAVOR_DECAY_RETRY_INTERVAL_SEC,
            )
            return MaintenanceOutcome.FAILED
        self._next_flavor_decay_monotonic = now + FLAVOR_DECAY_INTERVAL_SEC
        self._last_maintenance_error = None
        return MaintenanceOutcome.DONE

    async def cleanup_pivo_daily_usage(
        self,
        *,
        retention_days: int = PIVO_DAILY_USAGE_RETENTION_DAYS,
        today: date | None = None,
    ) -> int:
        """Deletes /pivo daily quota rows older than retention_days."""
        if retention_days < 0:
            raise ValueError("retention_days must be non-negative")
        current_day = today or datetime.now(UTC).date()
        cutoff_day = (current_day - timedelta(days=retention_days)).isoformat()
        return await self.pivo_usage.delete_usage_before(cutoff_day)

    # --- Кросс-доменные операции ---

    async def clear_pivo_chat_data(self, chat_hash: str) -> None:
        """Удаляет все /pivo-данные чата: подписки, квоты, анти-повтор пулов."""
        await self.chat_members.remove_chat(chat_hash)
        await self.pivo_usage.delete_chat_usage(chat_hash)
        await self.pivo_pool_usage.delete_chat(chat_hash)

    async def _fetch_int(
        self, db: aiosqlite.Connection, sql: str, params: tuple[object, ...]
    ) -> int:
        row = await (await db.execute(sql, params)).fetchone()
        return int(row[0] or 0) if row else 0

    async def get_stats(self, chat_id: int) -> dict[str, int]:
        """Полный срез по чату — диагностика, а не путь ответа.

        Команда ``/stats`` показывает только объём модели и берёт его через
        ``get_chat_token_volume``; счётчики строк здесь остаются, потому что
        по ним проверяется, что записалось в базу. Объёмы читаются из
        инкрементального счётчика — сравнение с ``SUM(cnt)`` живёт в тесте, а
        не в каждом вызове.
        """
        p = (chat_id,)
        volume2, volume3 = await self._model_volumes(chat_id, persist=False)
        async with self._lock:
            db = await self._get_conn()
            f = self._fetch_int

            msg_count    = await f(db, "SELECT COUNT(*) FROM messages WHERE chat_id = ?", p)
            starts2      = await f(db, "SELECT COUNT(*) FROM starts WHERE chat_id = ?", p)
            starts3      = await f(db, "SELECT COUNT(*) FROM starts3 WHERE chat_id = ?", p)
            trans2_count = await f(db, "SELECT COUNT(*) FROM transitions WHERE chat_id = ?", p)
            trans3_count = await f(db, "SELECT COUNT(*) FROM transitions3 WHERE chat_id = ?", p)

        return {
            "messages":     msg_count,
            "starts2":      starts2,
            "starts3":      starts3,
            "transitions2": trans2_count,
            "transitions3": trans3_count,
            "volume2":      volume2,
            "volume3":      volume3,
            "volume":       volume3 if volume3 > 0 else volume2,
        }

    async def get_verbatim_ngrams(
        self, chat_id: int
    ) -> list[tuple[str, str, str, str]]:
        """All distinct content 4-grams the chat has ever learned (O4 index)."""
        async with self._lock:
            db = await self._get_conn()
            cursor = await db.execute(
                "SELECT w1, w2, w3, w4 FROM chat_verbatim_ngrams "
                "WHERE chat_id = ?",
                (chat_id,),
            )
            rows = await cursor.fetchall()
        return [(row[0], row[1], row[2], row[3]) for row in rows]

    async def clear_chat(self, chat_id: int) -> None:
        tables = (
            "messages",
            "starts",
            "starts3",
            "transitions",
            "transitions3",
            "chat_model_volume",
            "chat_emoji_stats",
            "chat_hot_ngrams",
            "chat_user_interactions",
            "chat_verbatim_ngrams",
        )
        async with self._lock:
            db = await self._get_conn()
            # `tables` is the literal tuple above, never user input.
            for table in tables:
                await db.execute(
                    f"DELETE FROM {table} WHERE chat_id = ?",  # nosec B608
                    (chat_id,),
                )
            await db.commit()
