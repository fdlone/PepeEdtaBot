from __future__ import annotations

from collections.abc import Mapping

from app.repositories.base_repo import DecayableCountsRepo

# Вынесен в константу, чтобы тест читал тот же текст, что уходит в базу.
# Первая редакция теста сравнивала порядок ключей и была декоративной: план
# идёт по автоиндексу (chat_id, emoji), поэтому снятие `ORDER BY` его не
# роняло. Проверено мутацией 2026-08-26 — тест не упал. Тот же приём в
# `markov_repo.REVERSE_TRANSITIONS_SQL` мутацию ловит.
EMOJI_STATS_SQL = (
    "SELECT emoji, cnt FROM chat_emoji_stats WHERE chat_id = ? ORDER BY emoji"
)


class ChatEmojiStatsRepo(DecayableCountsRepo):
    """Per-chat emoji frequency for the M3 emoji channel.

    Keyed by raw ``chat_id`` to match the Markov model tables. Emojis are a
    per-chat aggregate (no author), so this stays within the existing privacy
    contour and is wiped together with the model in ``clear_chat``.
    """

    async def bump(self, chat_id: int, counts: Mapping[str, int]) -> None:
        """Add ``counts`` to the chat's emoji frequencies (no-op if empty)."""
        positive = {emoji: n for emoji, n in counts.items() if n > 0}
        if not positive:
            return
        await self._execute_many(
            """
            INSERT INTO chat_emoji_stats(chat_id, emoji, cnt)
            VALUES (?, ?, ?)
            ON CONFLICT(chat_id, emoji) DO UPDATE SET
                cnt = cnt + excluded.cnt,
                updated_at = datetime('now')
            """,
            [(chat_id, emoji, n) for emoji, n in positive.items()],
        )

    async def get_stats(self, chat_id: int) -> dict[str, int]:
        """Return {emoji: count} for a chat (empty if none learned)."""
        # ORDER BY — контракт-пин, а не починка, по образцу
        # `get_word_frequencies` и `get_hot` (пакет W1-C). Результат питает
        # розыгрыш: `sample_emoji` строит population/weights в порядке этого
        # словаря и передаёт их в `rng.choices`, а тот при фиксированном зерне
        # зависит от порядка популяции. Сегодня свойство держится случайно —
        # план запроса идёт по автоиндексу (chat_id, emoji), — но таблица
        # rowid'ная, и `decay_stale` переписывает строки, так что смена плана
        # на скан вернула бы порядок rowid. Оба прошлых экземпляра этой
        # ловушки (§5 CLAUDE.md) прожили до внешней разведки при зелёном
        # гарде: `generation_hash` эмодзи-канал не проверяет вовсе, он его
        # выключает (`tools/generation_hash.py`).
        rows = await self._fetch_all(EMOJI_STATS_SQL, (chat_id,))
        return {str(row[0]): int(row[1] or 0) for row in rows}

    async def decay_stale(self, cutoff_iso: str) -> int:
        """Halve counts of rows not bumped since ``cutoff_iso`` so dead memes fade."""
        return await self._decay_stale("chat_emoji_stats", cutoff_iso)
