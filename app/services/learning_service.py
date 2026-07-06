from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.core.markov import MarkovGenerator
from app.core.text import sanitize_text
from app.infrastructure.database import Database


class LearningService:
    """Сохранение сообщения, обновление цепи Маркова и проверка дословного повтора."""

    def __init__(
        self,
        db: Database,
        generator: MarkovGenerator,
        *,
        text_cache_max_messages: int = 500,
    ) -> None:
        self._db = db
        self._generator = generator
        self._text_cache_max_messages = text_cache_max_messages
        # Кэш нормализованных текстов: chat_id → set строк в нижнем регистре.
        # Строится при первой проверке из последних N сообщений чата и
        # сбрасывается при каждом новом сообщении.
        self._text_cache: dict[int, set[str]] = {}

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
        """True если текст дословно совпадает с одним из последних обучающих сообщений."""
        normalized = sanitize_text(text).lower()
        if not normalized:
            return False
        if chat_id not in self._text_cache:
            await self._build_text_cache(chat_id)
        return normalized in self._text_cache[chat_id]

    # --- приватные методы ---

    async def _build_text_cache(self, chat_id: int) -> None:
        recent_texts = await self._db.get_recent_normalized_messages(
            chat_id,
            self._text_cache_max_messages,
        )
        self._text_cache[chat_id] = {t.lower() for t in recent_texts}

    def _invalidate_text_cache(self, chat_id: int) -> None:
        self._text_cache.pop(chat_id, None)
