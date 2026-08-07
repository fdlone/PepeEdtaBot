from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Set

from app.core.candidate_scorer import build_token_idf, verbatim_ngram_windows
from app.core.intonation import IntonationProfile, build_intonation_profile
from app.core.markov import MarkovGenerator, tokenize
from app.core.slot_mutation import (
    ENDING_MATCH_LEN,
    MIN_MUTABLE_WORD_LEN,
    frequencies_by_ending,
)
from app.core.text import sanitize_text
from app.infrastructure.database import Database

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
    ) -> None:
        self._db = db
        self._generator = generator
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
        # Тот же словарь, разложенный по двухбуквенному окончанию: подбор
        # замены смотрит только корзину своего окончания, а не весь словарь.
        # Держится в синхроне с плоским словарём в одном месте — _absorb_message.
        self._frequencies_by_ending: dict[int, dict[str, dict[str, int]]] = {}
        # Монотонное время последнего обращения к кэшам чата — для вытеснения.
        self._last_touch: dict[int, float] = {}
        self._cleanup_tick = 0

    async def get_token_volume(self, chat_id: int) -> int:
        return await self._db.get_chat_token_volume(chat_id)

    async def run_due_maintenance(self) -> bool:
        """Прокручивает отложенное обслуживание базы, если подошёл срок.

        Планировщика в проекте нет, и learn-путь — единственное место, куда
        регулярно приходит управление. Обслуживание вызывается отдельным шагом,
        а не изнутри записи сообщения: его сбой не должен стоить выученного
        текста. Само обслуживание сбоев не пробрасывает (см.
        ``Database.decay_flavor_stats_if_due``).
        """
        return await self._db.decay_flavor_stats_if_due()

    async def record_message(
        self, chat_id: int, raw_text: str, tokens: list[str]
    ) -> int:
        """Атомарно записывает сообщение и обновляет модель.

        Возвращает текущий объём токенов модели для чата.
        """
        token_volume = await self._db.save_message_and_update_model(
            chat_id=chat_id,
            raw_text=raw_text,
            tokens=tokens,
        )
        self._generator.invalidate_chat_cache(chat_id)
        self._absorb_message(chat_id, raw_text, tokens)
        return token_volume

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
        self, chat_id: int, *, min_count: int, recency_share: float
    ) -> list[tuple[str, ...]]:
        """Currently-hot n-grams for unprompted-reply seeding (L1)."""
        return await self._db.chat_hot_ngrams.get_hot(
            chat_id, min_count=min_count, recency_share=recency_share
        )

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
        return await self._db.messages.get_recent_normalized(
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
