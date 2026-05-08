from __future__ import annotations

import random

from db import Database
from markov import MarkovGenerator, tokenize
from text_utils import sanitize_text


class LearningService:
    """Сохранение сообщения, обновление цепи Маркова и эвристика вариативности."""

    def __init__(self, db: Database, generator: MarkovGenerator) -> None:
        self._db = db
        self._generator = generator
        # Кэш префиксов: (chat_id, normalize_lower) → set кортежей токенов длиной 3–5.
        # Строится при первой проверке, сбрасывается при каждом новом сообщении.
        self._prefix_cache: dict[tuple[int, bool], set[tuple[str, ...]]] = {}

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
        self._invalidate_prefix_cache(chat_id)
        return token_volume

    async def looks_too_close_to_training_sample(
        self, chat_id: int, text: str, normalize_lower: bool
    ) -> bool:
        """True если текст слишком похож на уже виденное сообщение.

        Для коротких текстов (< 3 токенов) — точное SQL-совпадение.
        Для остальных — эвристика: проверяется случайный префикс длиной 3, 4 или
        5 токенов против in-memory кэша. Случайная длина намеренно добавляет
        вариативность: одно и то же сообщение может пройти фильтр на одной
        попытке и быть отклонено на другой. Это не строгая privacy-защита от
        копирования, а способ снизить ощущение механического повтора.
        """
        tokens = tokenize(sanitize_text(text), normalize_lower=normalize_lower)

        if len(tokens) < 3:
            return await self._db.message_exists(chat_id, text)

        cache_key = (chat_id, normalize_lower)
        if cache_key not in self._prefix_cache:
            await self._build_prefix_cache(chat_id, normalize_lower)

        prefix_len = random.randint(3, min(5, len(tokens)))
        return tuple(tokens[:prefix_len]) in self._prefix_cache[cache_key]

    # --- приватные методы ---

    async def _build_prefix_cache(
        self, chat_id: int, normalize_lower: bool
    ) -> None:
        all_texts = await self._db.get_all_normalized_messages(chat_id)
        prefixes: set[tuple[str, ...]] = set()
        for text in all_texts:
            toks = tokenize(text, normalize_lower=normalize_lower)
            for length in (3, 4, 5):
                if len(toks) >= length:
                    prefixes.add(tuple(toks[:length]))
        self._prefix_cache[(chat_id, normalize_lower)] = prefixes

    def _invalidate_prefix_cache(self, chat_id: int) -> None:
        keys = [k for k in self._prefix_cache if k[0] == chat_id]
        for k in keys:
            del self._prefix_cache[k]
