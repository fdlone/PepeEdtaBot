from __future__ import annotations

from collections import Counter

from app.core.hot_ngrams import is_content_ngram
from app.repositories.base_repo import BaseRepo


class ChatPhraseNgramsRepo(BaseRepo):
    """All-time content phrases of a chat, derived from the order-2 chain.

    Not an accumulator: the truth lives in ``transitions`` and this table is a
    rebuilt projection of it (change ``derive-phrase-index``, design Р2). A
    second accumulator of the same fact would drift from the chain silently,
    and the project has already paid for that once (map §3.6).

    Bigrams are stored with ``w3 = ''``, matching ``chat_hot_ngrams`` — but the
    count here is all-time and no decay touches it, matching
    ``chat_verbatim_ngrams``. Keyed by raw ``chat_id`` like the model tables, so
    ``clear_chat`` wipes it together with them.

    Nothing reads phrases for generation yet: the phrase route (M3R-210) is its
    own change with its own gate.
    """

    async def get_phrases(
        self, chat_id: int, *, min_count: int
    ) -> list[tuple[tuple[str, ...], int]]:
        """Phrases of the chat with support of at least ``min_count``.

        The threshold is the caller's business, not the index's: which support
        is meaningful is a route gate's decision.

        ``ORDER BY`` names every key column, not just the count. This result is
        headed for a route that will draw from it, and a partial ordering makes
        the draw undefined the moment two phrases tie — the exact defect that
        outlived a green hash guard once already (`CLAUDE.md` §5, §7).
        """
        rows = await self._fetch_all(
            """
            SELECT w1, w2, w3, cnt
            FROM chat_phrase_ngrams
            WHERE chat_id = ? AND cnt >= ?
            ORDER BY cnt DESC, w1, w2, w3
            """,
            (chat_id, min_count),
        )
        return [
            ((str(w1), str(w2), str(w3)) if w3 else (str(w1), str(w2)), int(cnt))
            for w1, w2, w3, cnt in rows
        ]

    async def rebuild_chat(self, chat_id: int) -> int:
        """Recompute the chat's phrases from ``transitions``; return row count.

        Full replacement rather than a diff, in one transaction: idempotence
        then holds by construction — the result depends only on the current
        chain — instead of resting on a comparison with a previous state the
        index would have to remember.

        A trigram is a chain row as stored; a bigram is that row group summed
        over ``w3``, the same aggregation ``ChatHotNgramsRepo.get_hot`` uses for
        its bigram arm. Reads no knobs at all, which is what keeps the daily
        pass clear of the per-chat settings trap (`CLAUDE.md` §5).
        """
        async with self._transaction() as db:
            cursor = await db.execute(
                "SELECT w1, w2, w3, cnt FROM transitions WHERE chat_id = ?",
                (chat_id,),
            )
            counts: Counter[tuple[str, ...]] = Counter()
            for w1, w2, w3, cnt in await cursor.fetchall():
                if is_content_ngram((w1, w2, w3)):
                    counts[(w1, w2, w3)] += int(cnt)
                if is_content_ngram((w1, w2)):
                    counts[(w1, w2)] += int(cnt)
            await db.execute(
                "DELETE FROM chat_phrase_ngrams WHERE chat_id = ?", (chat_id,)
            )
            if counts:
                await db.executemany(
                    "INSERT INTO chat_phrase_ngrams(chat_id, w1, w2, w3, cnt)"
                    " VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            chat_id,
                            phrase[0],
                            phrase[1],
                            phrase[2] if len(phrase) == 3 else "",
                            count,
                        )
                        for phrase, count in counts.items()
                    ],
                )
            return len(counts)
