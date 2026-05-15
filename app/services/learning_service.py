from __future__ import annotations

from db import Database
from markov import MarkovGenerator, tokenize
from text_utils import sanitize_text


class LearningService:
    """Сохранение сообщения, обновление цепи Маркова и эвристика вариативности."""

    def __init__(
        self,
        db: Database,
        generator: MarkovGenerator,
        *,
        prefix_cache_max_messages: int = 2000,
    ) -> None:
        self._db = db
        self._generator = generator
        self._prefix_cache_max_messages = prefix_cache_max_messages
        # Кэш префиксов: (chat_id, normalize_lower) → set кортежей токенов длиной 3–5.
        # Строится при первой проверке из последних N сообщений чата и
        # сбрасывается при каждом новом сообщении.
        self._prefix_cache: dict[tuple[int, bool], set[tuple[str, ...]]] = {}

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
        self._invalidate_prefix_cache(chat_id)
        return token_volume

    async def matches_training_prefix(
        self, chat_id: int, text: str, normalize_lower: bool
    ) -> bool:
        """True если начало текста совпадает с началом какого-то обучающего сообщения.

        Проверяется случайно выбранный префикс длиной 3, 4 или 5 токенов против
        in-memory кэша префиксов последних сохранённых сообщений чата. Случайная
        длина намеренно даёт вариативность: один и тот же кандидат может пройти
        фильтр на одной попытке и быть отклонён на другой.

        Это **второстепенный novelty-фильтр**: основные защиты от копий и
        зацикливаний находятся в :meth:`markov.MarkovGenerator.generate_text`
        (`is_low_diversity_reply`, `is_context_heavy_reply`,
        `trim_repetitive_tail`, exact-match `message_exists`).

        Тексты короче 3 токенов фильтр пропускает (нечего сравнивать); такие
        кандидаты уже отсекаются exact-match'ем внутри генератора.
        """
        tokens = tokenize(sanitize_text(text), normalize_lower=normalize_lower)
        if len(tokens) < 3:
            return False

        cache_key = (chat_id, normalize_lower)
        if cache_key not in self._prefix_cache:
            await self._build_prefix_cache(chat_id, normalize_lower)

        prefix_len = min(5, len(tokens))
        return tuple(tokens[:prefix_len]) in self._prefix_cache[cache_key]

    # --- приватные методы ---

    async def _build_prefix_cache(
        self, chat_id: int, normalize_lower: bool
    ) -> None:
        recent_texts = await self._db.get_recent_normalized_messages(
            chat_id,
            self._prefix_cache_max_messages,
        )
        prefixes: set[tuple[str, ...]] = set()
        for text in recent_texts:
            toks = tokenize(text, normalize_lower=normalize_lower)
            for length in (3, 4, 5):
                if len(toks) >= length:
                    prefixes.add(tuple(toks[:length]))
        self._prefix_cache[(chat_id, normalize_lower)] = prefixes

    def _invalidate_prefix_cache(self, chat_id: int) -> None:
        keys = [k for k in self._prefix_cache if k[0] == chat_id]
        for k in keys:
            del self._prefix_cache[k]
