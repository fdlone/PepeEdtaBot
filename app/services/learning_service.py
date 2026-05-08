from __future__ import annotations

from db import Database
from markov import MarkovGenerator


class LearningService:
    """Сохранение сообщения, обновление цепи Маркова и инвалидация кеша генератора."""

    def __init__(self, db: Database, generator: MarkovGenerator) -> None:
        self._db = db
        self._generator = generator

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
        return token_volume

    async def is_duplicate(self, chat_id: int, text: str) -> bool:
        """True если normalized_text уже есть в messages для данного чата."""
        return await self._db.message_exists(chat_id, text)
