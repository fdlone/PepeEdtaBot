from __future__ import annotations

import logging
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Set
from dataclasses import dataclass

from app.config.defaults import SHORT_HALF_LIFE_DAYS
from app.core.candidate_scorer import build_token_idf, verbatim_ngram_windows
from app.core.intonation import IntonationProfile, build_intonation_profile
from app.core.markov import MarkovGenerator, tokenize
from app.core.slot_mutation import (
    ENDING_MATCH_LEN,
    MIN_MUTABLE_WORD_LEN,
    frequencies_by_ending,
)
from app.core.text import sanitize_text
from app.infrastructure.database import Database, MaintenanceOutcome
from app.log_masking import mask_chat_id
from app.services.meme_analyzer import MemeSettings, analyze_chat_memes

# Trailing punctuation stripped before verbatim comparison: the reply pipeline
# (finalize_reply_ending / reply flavor) only ever rewrites the ending cluster,
# so "текст." must match the learned "текст" — previously that dot let full
# quotes slip through the gate.
_VERBATIM_TRAILING_CHARS = " .!?…"


def normalize_for_verbatim(text: str) -> str:
    """Normalize a text for full-message verbatim comparison."""
    return sanitize_text(text).lower().rstrip(_VERBATIM_TRAILING_CHARS)


# Сколько выученных сообщений копится, прежде чем тяжёлая статистика чата
# пересобирается. IDF и интонационный профиль считаются по окну в сотни
# сообщений: одно новое сообщение сдвигает доли на доли процента, а стоит
# пересборка полного прохода по выборке с токенизацией и стеммингом. Дешёвые
# кэши (тексты, 4-граммы, частоты) обновляются инкрементально и этой отсрочки
# не знают.
STATS_REBUILD_EVERY_MESSAGES = 50

logger = logging.getLogger("chat_markov")


def _fold_shadow_order4(
    index: dict[tuple[str, str, str, str], dict[str, int]],
    tokens: list[str],
) -> None:
    """Учесть одно сообщение в теневом order-4 индексе (casefold, M2R-020)."""
    if len(tokens) < 5:
        return
    folded = [token.casefold() for token in tokens]
    for i in range(len(folded) - 4):
        state = (folded[i], folded[i + 1], folded[i + 2], folded[i + 3])
        bucket = index.setdefault(state, {})
        continuation = folded[i + 4]
        bucket[continuation] = bucket.get(continuation, 0) + 1

# Сколько сбой обслуживания должен продержаться, прежде чем о нём стоит
# беспокоить владельца. Разовая неудача — это обычная занятая база (бэкап,
# VACUUM, внешний клиент); она проходит сама и ничего не стоит. Час непрерывных
# неудач — это уже не окно обслуживания.
MAINTENANCE_ALERT_AFTER_SEC = 3600.0
# Как часто напоминать, пока не починили. Сигнал не должен сам стать потоком
# сообщений от бота — та же осторожность, что с отказом /pivo по квоте.
MAINTENANCE_ALERT_REPEAT_SEC = 86400.0


@dataclass(frozen=True, slots=True)
class MaintenanceAlert:
    """Сигнал владельцу о состоянии отложенного обслуживания базы.

    ``recovered=False`` — обслуживание не проходит уже ``failing_for_sec``;
    ``recovered=True`` — снова проходит, и об этом стоит сказать, потому что о
    поломке уже сообщали.
    """

    recovered: bool
    failing_for_sec: float
    reason: str | None


class LearningService:
    """Сохранение сообщения, обновление цепи Маркова и проверка дословного повтора."""

    def __init__(
        self,
        db: Database,
        generator: MarkovGenerator,
        *,
        text_cache_max_messages: int = 1000,
        user_hasher: Callable[[int], str] | None = None,
        cache_ttl_sec: int = 86400,
        cache_max_chats: int = 2048,
        meme_settings: MemeSettings | None = None,
    ) -> None:
        self._db = db
        self._generator = generator
        # M2R-300: settings for the daily meme pass. Falls back to defaults
        # rather than disabling itself — the registry should fill from the day
        # the code ships, because the manual rating round needs a ranking to
        # rate long before any scoring knob is raised.
        self._meme_settings = meme_settings or MemeSettings()
        self._text_cache_max_messages = text_cache_max_messages
        self._cache_ttl_sec = float(cache_ttl_sec)
        self._cache_max_chats = cache_max_chats
        # L2: анонимизация user_id для счётчика взаимодействий — main.py
        # внедряет PivoSecurity.hmac_value (та же схема, что /pivo). None
        # оставляет прежних вызывающих нетронутыми: quirk-методы достижимы
        # только при включённом канале, а main всегда передаёт хешер.
        self._user_hasher = user_hasher
        # Кэш последних N нормализованных текстов чата — та же выборка, что
        # отдаёт база. Хранится очередью (порядок и вытеснение старого) плюс
        # счётчиком вхождений: проверка дословного повтора спрашивает «есть ли
        # такой текст среди последних N», а один и тот же текст мог быть
        # выучен дважды.
        self._text_window: dict[int, deque[str]] = {}
        self._text_counts: dict[int, Counter[str]] = {}
        # Кумулятивный индекс контент-4-грамм чата. Пополняется окнами нового
        # сообщения; полностью читается из базы только при первом обращении.
        self._ngram_index: dict[int, set[tuple[str, ...]]] = {}
        # IDF контент-токенов по выборке: нужен, чтобы совпадение ответа с
        # контекстом по редкому имени весило больше, чем по частому «кто».
        # Пересобирается не чаще, чем раз в STATS_REBUILD_EVERY_MESSAGES.
        self._token_idf: dict[int, dict[str, float]] = {}
        self._idf_pending: dict[int, int] = {}
        # Интонационный профиль чата (P4) по той же выборке; None кэшируется
        # тоже — «мало сообщений» не должно перечитывать выборку на каждый
        # ответ. Пересобирается на той же отсрочке, что IDF.
        self._intonation: dict[int, IntonationProfile | None] = {}
        self._intonation_pending: dict[int, int] = {}
        # Частотный словарь чата для слот-мутаций: агрегат по всей памяти
        # order-2 цепи (не только окно ретенции messages). Пополняется словами
        # нового сообщения.
        self._word_frequencies: dict[int, dict[str, int]] = {}
        # Теневой order-4 индекс (M2R-020): casefold-4-грамма -> продолжения
        # по окну удержанных сообщений. Оценка снизу против полной истории —
        # ровно то, что нужно консервативному гейту Phase 7. Строится лениво,
        # пополняется в _absorb_message, вытесняется наравне с остальными.
        self._shadow_order4: dict[
            int, dict[tuple[str, str, str, str], dict[str, int]]
        ] = {}
        # Тот же словарь, разложенный по двухбуквенному окончанию: подбор
        # замены смотрит только корзину своего окончания, а не весь словарь.
        # Держится в синхроне с плоским словарём в одном месте — _absorb_message.
        self._frequencies_by_ending: dict[int, dict[str, dict[str, int]]] = {}
        # Монотонное время последнего обращения к кэшам чата — для вытеснения.
        self._last_touch: dict[int, float] = {}
        self._cleanup_tick = 0
        # Состояние сигнала владельцу об обслуживании: когда сбой начался и
        # когда о нём в последний раз сообщали. Живёт здесь, а не в конвейере:
        # конвейер собирается заново на каждое сообщение.
        self._maintenance_failing_since: float | None = None
        self._maintenance_alerted_at: float | None = None

    async def get_token_volume(self, chat_id: int) -> int:
        return await self._db.get_chat_token_volume(chat_id)

    async def run_due_maintenance(
        self, meme_settings: MemeSettings | None = None
    ) -> MaintenanceAlert | None:
        """Прокручивает отложенное обслуживание базы, если подошёл срок.

        ``meme_settings`` — эффективные значения /set-ручек мемо-пасса на
        момент вызова (передаёт конвейер); без них пасс идёт на значениях,
        полученных конструктором, — то есть /set не действовал бы до
        перезапуска.

        Планировщика в проекте нет, и learn-путь — единственное место, куда
        регулярно приходит управление. Обслуживание вызывается отдельным шагом,
        а не изнутри записи сообщения: его сбой не должен стоить выученного
        текста. Само обслуживание сбоев не пробрасывает (см.
        ``Database.decay_flavor_stats_if_due``).

        Возвращает сигнал, если о состоянии обслуживания пора сказать
        владельцу, иначе ``None``. Отправку делает вызывающий: сервис не знает
        про Telegram.
        """
        outcome = await self._db.decay_flavor_stats_if_due()
        now = time.monotonic()

        if outcome is MaintenanceOutcome.SKIPPED:
            return None

        # M2R-300: the meme pass rides the same daily cadence. It runs only when
        # the flavor decay actually ran, so it inherits the "once a day" rule
        # instead of inventing a second schedule. Failures are swallowed the
        # same way the decay swallows its own: a meme ranking is worth less than
        # the message the learn path is in the middle of handling.
        if outcome is MaintenanceOutcome.DONE:
            await self._run_meme_analysis(meme_settings or self._meme_settings)

        if outcome is MaintenanceOutcome.DONE:
            failing_since = self._maintenance_failing_since
            alerted = self._maintenance_alerted_at is not None
            self._maintenance_failing_since = None
            self._maintenance_alerted_at = None
            if not alerted:
                # О поломке не сообщали — сообщать о починке нечего.
                return None
            return MaintenanceAlert(
                recovered=True,
                failing_for_sec=now - (failing_since or now),
                reason=None,
            )

        if self._maintenance_failing_since is None:
            self._maintenance_failing_since = now
        failing_for = now - self._maintenance_failing_since
        if failing_for < MAINTENANCE_ALERT_AFTER_SEC:
            return None
        if (
            self._maintenance_alerted_at is not None
            and now - self._maintenance_alerted_at < MAINTENANCE_ALERT_REPEAT_SEC
        ):
            return None
        self._maintenance_alerted_at = now
        return MaintenanceAlert(
            recovered=False,
            failing_for_sec=failing_for,
            reason=self._db.last_maintenance_error,
        )

    async def record_message(
        self,
        chat_id: int,
        raw_text: str,
        tokens: list[str],
        *,
        incremental_cache: bool = False,
        short_half_life_days: float = SHORT_HALF_LIFE_DAYS,
    ) -> int:
        """Атомарно записывает сообщение и обновляет модель.

        Возвращает текущий объём токенов модели для чата.
        ``incremental_cache`` (M2R-030, ручка ``markov_cache_incremental``)
        вкатывает дельты сообщения в кэши распределений генератора вместо
        их полного сброса; дефолт false сохраняет прежнее поведение для
        вызывающих, не передающих ручку.

        Момент наблюдения берётся здесь один раз и передаётся обоим — SQL и
        свёртке кэша (M2R-210). Два вызова ``time.time()`` разошлись бы на
        доли секунды, и свёрнутый кэш перестал бы совпадать со свежим чтением:
        именно это гарантирует тест эквивалентности.
        """
        now = int(time.time())
        token_volume = await self._db.save_message_and_update_model(
            chat_id=chat_id,
            raw_text=raw_text,
            tokens=tokens,
            now=now,
            short_half_life_days=short_half_life_days,
        )
        if incremental_cache:
            self._generator.apply_learning_deltas(
                chat_id, tokens, now, short_half_life_days
            )
        else:
            self._generator.invalidate_chat_cache(chat_id)
        self._absorb_message(chat_id, raw_text, tokens)
        return token_volume

    async def _run_meme_analysis(self, settings: MemeSettings) -> None:
        """One meme pass per known chat, failures contained (M2R-300).

        Deliberately does not raise: this is background bookkeeping riding on
        the learn path, and a failed ranking must not cost the message being
        learned. The previous registry stays usable, and the next pass retries.
        """
        now = int(time.time())
        try:
            chat_ids = await self._db.list_chat_ids()
        except Exception:
            # Even enumerating the chats is background bookkeeping. If it
            # fails, the learn path must carry on — the spec's contract is that
            # a failing pass leaves the previous registry usable, and that has
            # to hold for the whole pass, not just for its inner loop.
            logger.exception("meme analysis could not list chats")
            return
        for chat_id in chat_ids:
            try:
                result = await analyze_chat_memes(
                    self._db.collocations,
                    chat_id,
                    now=now,
                    min_joint_count=settings.min_joint_count,
                    min_support=settings.min_support,
                    recency_days=settings.recency_days,
                    max_entries=settings.max_entries,
                )
            except Exception:
                logger.exception(
                    "meme analysis failed for chat %s", mask_chat_id(chat_id)
                )
                continue
            logger.info(
                "meme analysis chat=%s scored=%d stored=%d took=%.1fms",
                mask_chat_id(chat_id),
                result.scored_pairs,
                result.stored_pairs,
                result.duration_ms,
            )
            # Task 4.3 / generation-telemetry spec: duration and scored-pair
            # count per pass, visible in /stats — no chat identifiers.
            self._generator.telemetry.note_meme_pass(
                scored_pairs=result.scored_pairs,
                duration_ms=result.duration_ms,
            )

    async def reset_short_layer(self, chat_id: int | None) -> None:
        """Discard the short layer (TZ §7.2), leaving the long one intact.

        Called when the half-life changes: a decayed counter only means
        something against the half-life it accumulated under, so keeping it
        would silently mix two scales. The long counts, ``first_seen`` and
        ``last_seen`` are untouched — this throws away the chat's sense of
        *recent*, never its memory.

        ``chat_id=None`` resets every chat, which is what a global ``/set``
        means; the distribution caches are dropped so the change is visible
        without a restart.
        """
        await self._db.reset_short_layer(chat_id)
        if chat_id is None:
            self._generator.invalidate_all_caches()
        else:
            self._generator.invalidate_chat_cache(chat_id)

    async def record_emojis(self, chat_id: int, counts: Mapping[str, int]) -> None:
        """Fold a message's emoji frequencies into the chat's emoji stats (M3)."""
        await self._db.chat_emoji_stats.bump(chat_id, counts)

    async def get_emoji_stats(self, chat_id: int) -> dict[str, int]:
        """Per-chat emoji frequencies for the emoji-append channel (M3)."""
        return await self._db.chat_emoji_stats.get_stats(chat_id)

    async def record_hot_ngrams(
        self, chat_id: int, ngrams: Iterable[tuple[str, ...]]
    ) -> None:
        """Fold a learned message's content n-grams into the hot-ngram window (L1)."""
        await self._db.chat_hot_ngrams.bump(chat_id, ngrams)

    async def get_hot_ngrams(
        self,
        chat_id: int,
        *,
        min_count: int,
        recency_share: float,
        meme_ordering: bool = False,
    ) -> list[tuple[str, ...]]:
        """Currently-hot n-grams for unprompted-reply seeding (L1).

        ``meme_ordering`` (M2R-310) reorders the same selection by the
        collocation registry's association score; the frequency ordering
        stays the default so the eval can run both paths side by side.
        """
        return await self._db.chat_hot_ngrams.get_hot(
            chat_id,
            min_count=min_count,
            recency_share=recency_share,
            meme_ordering=meme_ordering,
        )

    async def get_active_collocations(
        self, chat_id: int
    ) -> frozenset[tuple[str, str]]:
        """Active collocation pairs for candidate scoring (M2R-320).

        Read once per reply by the generation pipeline — and only when a
        scoring weight is non-zero, so the neutral defaults add no query.
        """
        return frozenset(await self._db.collocations.get_active(chat_id))

    def _hash_user(self, user_id: int) -> str:
        if self._user_hasher is None:
            raise RuntimeError(
                "LearningService has no user_hasher: user-interaction methods "
                "require one (main.py injects PivoSecurity.hmac_value)"
            )
        return self._user_hasher(user_id)

    async def record_user_interaction(self, chat_id: int, user_id: int) -> None:
        """Count one answered mention for the user (L2, anonymized)."""
        await self._db.chat_user_interactions.bump(chat_id, self._hash_user(user_id))

    async def get_user_interaction_count(self, chat_id: int, user_id: int) -> int:
        """Answered-mention count for the user in the chat (L2, anonymized)."""
        return await self._db.chat_user_interactions.get_count(
            chat_id, self._hash_user(user_id)
        )

    async def get_user_interaction_stats(
        self, chat_id: int, threshold: int
    ) -> tuple[int, int, int]:
        """``(people, max_count, at_or_above)`` for the chat — numbers only.

        Answers "is the regulars threshold reachable in this chat?" without
        touching the anonymity of the counters themselves.
        """
        return await self._db.chat_user_interactions.get_stats(chat_id, threshold)

    async def is_verbatim_copy(self, chat_id: int, text: str) -> bool:
        """True если текст дословно совпадает с одним из последних обучающих сообщений.

        Сравнение игнорирует хвостовую пунктуацию с обеих сторон, чтобы
        добавленная финализацией точка не пропускала цитату через гейт.
        """
        normalized = normalize_for_verbatim(text)
        if not normalized:
            return False
        self._touch(chat_id)
        if chat_id not in self._text_counts:
            await self._build_text_window(chat_id)
        return normalized in self._text_counts[chat_id]

    async def get_verbatim_ngram_index(
        self, chat_id: int
    ) -> Set[tuple[str, ...]]:
        """Every content 4-gram the chat has ever learned (cumulative, O4).

        Used by the response generator to penalize candidates whose windows all
        come verbatim from the corpus (quote detection, not a hard gate) and to
        trigger the near-quote extension. Reads the cumulative
        ``chat_verbatim_ngrams`` table — the message-window variant was blind
        to ~75% of what the chain can replay (quotes older than retention).

        The table is never trimmed, so a full read grows with the chat's whole
        history. It happens once per chat: afterwards the index is kept up to
        date by folding in the windows of each learned message.
        """
        self._touch(chat_id)
        cached = self._ngram_index.get(chat_id)
        if cached is not None:
            return cached
        index: set[tuple[str, ...]] = set(
            await self._db.get_verbatim_ngrams(chat_id)
        )
        self._ngram_index[chat_id] = index
        return index

    async def get_context_idf(self, chat_id: int) -> Mapping[str, float]:
        """IDF контент-токенов по последним N сообщениям чата.

        Выборка та же, что у ``get_verbatim_ngram_index``, и она уже́ ретенции
        модели: ``messages`` подрезается, а ``transitions3`` — нет. Токены,
        которых в выборке не оказалось, считаются максимально редкими
        (см. ``idf_context_relevance``), что для имён собственных верно.
        """
        self._touch(chat_id)
        cached = self._token_idf.get(chat_id)
        if cached is not None and not self._is_stale(self._idf_pending, chat_id):
            return cached
        recent_texts = await self._get_recent_texts(chat_id)
        idf = build_token_idf(tokenize(text) for text in recent_texts)
        self._token_idf[chat_id] = idf
        self._idf_pending[chat_id] = 0
        return idf

    async def get_intonation_profile(
        self, chat_id: int
    ) -> IntonationProfile | None:
        """Интонационный профиль чата (P4) или None, пока сообщений мало."""
        self._touch(chat_id)
        if chat_id in self._intonation and not self._is_stale(
            self._intonation_pending, chat_id
        ):
            return self._intonation[chat_id]
        profile = build_intonation_profile(await self._get_recent_texts(chat_id))
        self._intonation[chat_id] = profile
        self._intonation_pending[chat_id] = 0
        return profile

    async def get_word_frequencies(self, chat_id: int) -> Mapping[str, int]:
        """Частоты слов чата по всей памяти цепи (для слот-мутаций).

        Слова короче ``MIN_MUTABLE_WORD_LEN`` отрезаются ещё в SQL — они всё
        равно не проходят гард мутаций.
        """
        self._touch(chat_id)
        cached = self._word_frequencies.get(chat_id)
        if cached is not None:
            return cached
        frequencies = await self._db.markov.get_word_frequencies(
            chat_id, min_word_len=MIN_MUTABLE_WORD_LEN
        )
        self._word_frequencies[chat_id] = frequencies
        self._frequencies_by_ending[chat_id] = frequencies_by_ending(frequencies)
        return frequencies

    async def get_word_frequencies_by_ending(
        self, chat_id: int
    ) -> Mapping[str, Mapping[str, int]]:
        """Частоты слов чата, разложенные по окончанию (для слот-мутаций).

        Замена обязана совпадать с оригиналом по окончанию, и раскладка
        превращает обход всего словаря чата в обход одной корзины. Порядок
        внутри корзины — тот же, что в плоском словаре, поэтому набор и
        порядок кандидатов не меняются.
        """
        await self.get_word_frequencies(chat_id)
        return self._frequencies_by_ending[chat_id]

    def forget_chat(self, chat_id: int) -> None:
        """Полностью забыть кэши чата.

        Вызывается после ``/clear``: данные чата в базе стёрты, и кэш, который
        их пережил, показывал бы боту удалённые сообщения — например, гейт
        дословного повтора продолжал бы считать цитатой то, чего больше нет.
        """
        self._text_window.pop(chat_id, None)
        self._text_counts.pop(chat_id, None)
        self._ngram_index.pop(chat_id, None)
        self._token_idf.pop(chat_id, None)
        self._idf_pending.pop(chat_id, None)
        self._intonation.pop(chat_id, None)
        self._intonation_pending.pop(chat_id, None)
        self._word_frequencies.pop(chat_id, None)
        self._frequencies_by_ending.pop(chat_id, None)
        self._shadow_order4.pop(chat_id, None)
        self._last_touch.pop(chat_id, None)

    # --- приватные методы ---

    def _touch(self, chat_id: int) -> None:
        """Отметить обращение к кэшам чата и изредка вытеснить неактивные.

        Та же политика, что у остальных состояний в памяти: срок хранения плюс
        предел по числу чатов. Иначе замолчавший чат держал бы свой индекс
        4-грамм и частотный словарь до перезапуска — а число чатов задаётся
        снаружи.
        """
        now = time.monotonic()
        self._last_touch[chat_id] = now
        self._cleanup_tick += 1
        if (
            self._cleanup_tick >= 64
            or len(self._last_touch) > self._cache_max_chats
        ):
            self._cleanup_tick = 0
            self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._cache_ttl_sec
        for chat_id in [
            chat_id
            for chat_id, seen in self._last_touch.items()
            if seen < cutoff
        ]:
            self.forget_chat(chat_id)

        overflow = len(self._last_touch) - self._cache_max_chats
        if overflow > 0:
            oldest = sorted(self._last_touch.items(), key=lambda item: item[1])
            for chat_id, _ in oldest[:overflow]:
                self.forget_chat(chat_id)

    def _is_stale(self, pending: dict[int, int], chat_id: int) -> bool:
        return pending.get(chat_id, 0) >= STATS_REBUILD_EVERY_MESSAGES

    def _absorb_message(
        self, chat_id: int, raw_text: str, tokens: list[str]
    ) -> None:
        """Учесть выученное сообщение в кэшах вместо их сброса.

        Сброс на каждое сообщение означал, что в активном чате почти каждый
        ответ идёт по холодному пути: следующий же ответ пересобирал выборку
        из тысячи сообщений, IDF по ней, весь кумулятивный индекс 4-грамм и
        частотный словарь агрегатом по всей цепи — и всё это синхронно.
        """
        self._touch(chat_id)

        window = self._text_window.get(chat_id)
        if window is not None:
            self._push_text(chat_id, window, normalize_for_verbatim(raw_text))

        index = self._ngram_index.get(chat_id)
        if index is not None:
            index.update(verbatim_ngram_windows(tokens))

        frequencies = self._word_frequencies.get(chat_id)
        if frequencies is not None:
            # Частоты считаются по продолжениям order-2 цепи (w3), то есть по
            # токенам начиная с третьего — ровно то, что записывает
            # save_message_and_update_model.
            by_ending = self._frequencies_by_ending[chat_id]
            for token in tokens[2:]:
                if len(token) >= MIN_MUTABLE_WORD_LEN:
                    frequencies[token] = frequencies.get(token, 0) + 1
                    bucket = by_ending.setdefault(
                        token.casefold()[-ENDING_MATCH_LEN:], {}
                    )
                    bucket[token] = frequencies[token]

        shadow = self._shadow_order4.get(chat_id)
        if shadow is not None:
            _fold_shadow_order4(shadow, tokens)

        if chat_id in self._token_idf:
            self._idf_pending[chat_id] = self._idf_pending.get(chat_id, 0) + 1
        if chat_id in self._intonation:
            self._intonation_pending[chat_id] = (
                self._intonation_pending.get(chat_id, 0) + 1
            )

    def _push_text(self, chat_id: int, window: deque[str], text: str) -> None:
        """Добавить текст в окно, вытеснив самый старый при переполнении."""
        counts = self._text_counts[chat_id]
        if len(window) == window.maxlen:
            evicted = window[0]
            counts[evicted] -= 1
            if counts[evicted] <= 0:
                del counts[evicted]
        window.append(text)
        counts[text] += 1

    async def _get_recent_texts(self, chat_id: int) -> list[str]:
        return await self._db.markov.get_recent_normalized(
            chat_id,
            self._text_cache_max_messages,
        )

    async def _build_text_window(self, chat_id: int) -> None:
        recent_texts = await self._get_recent_texts(chat_id)
        normalized = [normalize_for_verbatim(text) for text in recent_texts]
        self._text_window[chat_id] = deque(
            normalized, maxlen=self._text_cache_max_messages
        )
        self._text_counts[chat_id] = Counter(self._text_window[chat_id])

    async def get_order4_shadow_index(
        self, chat_id: int
    ) -> Mapping[tuple[str, str, str, str], Mapping[str, int]]:
        """Оконная оценка поддержки order-4 для теневого селектора (M2R-020).

        Документная база — те же удержанные сообщения, что у остальных
        кэшей; casefold с обеих сторон. После первого построения индекс
        пополняется новыми сообщениями (см. ``_absorb_message``): строго
        говоря, это «окно на момент построения плюс всё выученное после»,
        что по-прежнему не превышает полную историю — оценка остаётся
        нижней границей.
        """
        self._touch(chat_id)
        cached = self._shadow_order4.get(chat_id)
        if cached is not None:
            return cached
        index: dict[tuple[str, str, str, str], dict[str, int]] = {}
        for text in await self._get_recent_texts(chat_id):
            _fold_shadow_order4(index, tokenize(text))
        self._shadow_order4[chat_id] = index
        return index
