from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.core.candidate_scorer import VERBATIM_NGRAM_SIZE
from app.core.markov import MarkovGenerator, build_windows, content_tokens, tokenize
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


def content_ngram_windows(
    text: str, size: int = VERBATIM_NGRAM_SIZE
) -> list[tuple[str, ...]]:
    """Casefolded content-token ``size``-grams of one message."""
    content = [token.casefold() for token in content_tokens(tokenize(text))]
    return build_windows(content, size)


class LearningService:
    """Сохранение сообщения, обновление цепи Маркова и проверка дословного повтора."""

    def __init__(
        self,
        db: Database,
        generator: MarkovGenerator,
        *,
        text_cache_max_messages: int = 1000,
    ) -> None:
        self._db = db
        self._generator = generator
        self._text_cache_max_messages = text_cache_max_messages
        # Кэш нормализованных текстов: chat_id → set строк в нижнем регистре.
        # Строится при первой проверке из последних N сообщений чата и
        # сбрасывается при каждом новом сообщении.
        self._text_cache: dict[int, set[str]] = {}
        # Индекс контент-4-грамм корпуса для штрафа за цитирование: строится
        # из той же выборки последних N сообщений и сбрасывается вместе с
        # текстовым кэшем.
        self._ngram_index: dict[int, frozenset[tuple[str, ...]]] = {}

    async def get_token_volume(self, chat_id: int) -> int:
        return await self._db.get_chat_token_volume(chat_id)

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
        self._invalidate_text_cache(chat_id)
        return token_volume

    async def record_emojis(self, chat_id: int, counts: Mapping[str, int]) -> None:
        """Fold a message's emoji frequencies into the chat's emoji stats (M3)."""
        await self._db.record_chat_emojis(chat_id, counts)

    async def get_emoji_stats(self, chat_id: int) -> dict[str, int]:
        """Per-chat emoji frequencies for the emoji-append channel (M3)."""
        return await self._db.get_chat_emoji_stats(chat_id)

    async def record_hot_ngrams(
        self, chat_id: int, ngrams: Iterable[tuple[str, ...]]
    ) -> None:
        """Fold a learned message's content n-grams into the hot-ngram window (L1)."""
        await self._db.record_chat_hot_ngrams(chat_id, ngrams)

    async def get_hot_ngrams(
        self, chat_id: int, *, min_count: int, recency_share: float
    ) -> list[tuple[str, ...]]:
        """Currently-hot n-grams for unprompted-reply seeding (L1)."""
        return await self._db.get_hot_chat_ngrams(
            chat_id, min_count=min_count, recency_share=recency_share
        )

    async def is_verbatim_copy(self, chat_id: int, text: str) -> bool:
        """True если текст дословно совпадает с одним из последних обучающих сообщений.

        Сравнение игнорирует хвостовую пунктуацию с обеих сторон, чтобы
        добавленная финализацией точка не пропускала цитату через гейт.
        """
        normalized = normalize_for_verbatim(text)
        if not normalized:
            return False
        if chat_id not in self._text_cache:
            await self._build_text_cache(chat_id)
        return normalized in self._text_cache[chat_id]

    async def get_verbatim_ngram_index(
        self, chat_id: int
    ) -> frozenset[tuple[str, ...]]:
        """Content 4-grams of the chat's recent training messages.

        Used by the response generator to penalize candidates whose windows all
        come verbatim from the corpus (quote detection, not a hard gate).
        """
        cached = self._ngram_index.get(chat_id)
        if cached is not None:
            return cached
        recent_texts = await self._get_recent_texts(chat_id)
        index = frozenset(
            window
            for text in recent_texts
            for window in content_ngram_windows(text)
        )
        self._ngram_index[chat_id] = index
        return index

    # --- приватные методы ---

    async def _get_recent_texts(self, chat_id: int) -> list[str]:
        return await self._db.get_recent_normalized_messages(
            chat_id,
            self._text_cache_max_messages,
        )

    async def _build_text_cache(self, chat_id: int) -> None:
        recent_texts = await self._get_recent_texts(chat_id)
        self._text_cache[chat_id] = {
            normalize_for_verbatim(t) for t in recent_texts
        }

    def _invalidate_text_cache(self, chat_id: int) -> None:
        self._text_cache.pop(chat_id, None)
        self._ngram_index.pop(chat_id, None)
