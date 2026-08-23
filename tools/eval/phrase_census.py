"""Census of content phrases already accumulated in the chain (M3R-000/M3R-210).

The phrase route's gate is a count: at least 1000 content phrases with a count
of 3 or more. The roadmap sourced that count from a census of the retention
window (132 phrases) and concluded the route waits on months of accumulation.
That reading looked at the wrong table.

``transitions`` stores the order-2 chain — ``(w1, w2) -> w3`` with an all-time
count. Every row is a contiguous trigram of some past message; summing a row
group over ``w3`` gives the bigram. The table is only ever incremented and is
wiped by ``/clear`` alone: message retention does not touch it. So the content
phrase index is not something to accumulate — it is a filter over what the
chain already holds, and this script measures how much that is.

The filter is imported from ``app.core.hot_ngrams`` rather than restated, so
the number cannot drift from the extraction the index will actually use.

Read-only: opens the database via a URI in ``mode=ro`` and never writes.

    python -m tools.eval.phrase_census --db db_prod_copy/markov.db
    python -m tools.eval.phrase_census --db db_prod_copy/markov.db --min-count 2 3 5
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

from app.core.hot_ngrams import MIN_CONTENT_TOKEN_LEN
from app.core.lexicon import STOPWORDS


def _is_wordlike(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def _is_content(token: str) -> bool:
    return (
        len(token) >= MIN_CONTENT_TOKEN_LEN
        and token.casefold() not in STOPWORDS
        and _is_wordlike(token)
    )


def _qualifies(ngram: tuple[str, ...]) -> bool:
    """The ``extract_content_ngrams`` predicate, applied to a stored n-gram.

    Punctuation disqualifies the whole n-gram, which is also what keeps this
    equivalent to extraction from the message stream: a phrase that crossed a
    punctuation mark cannot pass here either.
    """
    return all(_is_wordlike(token) for token in ngram) and any(
        _is_content(token) for token in ngram
    )


def _counts(
    db_path: Path, chat_id: int | None
) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]]]:
    """All-time counts of content trigrams and bigrams for one chat."""
    uri = f"file:{db_path}?mode=ro"
    # closing(), not the bare context manager: sqlite3's own is transaction
    # scoped and leaves the handle open, which on Windows keeps the file locked
    # for whoever reads it next.
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        if chat_id is None:
            row = conn.execute(
                "SELECT chat_id, SUM(cnt) FROM transitions "
                "GROUP BY chat_id ORDER BY 2 DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise SystemExit("no transitions in this database")
            chat_id = int(row[0])
        trigrams: Counter[tuple[str, ...]] = Counter()
        bigrams: Counter[tuple[str, ...]] = Counter()
        for w1, w2, w3, cnt in conn.execute(
            "SELECT w1, w2, w3, cnt FROM transitions WHERE chat_id = ?",
            (chat_id,),
        ):
            if _qualifies((w1, w2, w3)):
                trigrams[(w1, w2, w3)] += int(cnt)
            # The bigram's all-time count is the row group's sum over w3 — the
            # same aggregation ChatHotNgramsRepo.get_hot uses for its bigram arm.
            if _qualifies((w1, w2)):
                bigrams[(w1, w2)] += int(cnt)
    return trigrams, bigrams


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="db_prod_copy/markov.db")
    parser.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="default: the chat with the largest order-2 model",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        nargs="*",
        default=[2, 3],
        help="support thresholds to report (default: 2 and 3, the gate's number)",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"no such database: {db_path}")
    trigrams, bigrams = _counts(db_path, args.chat_id)

    # Chat id is deliberately not printed: this output is pasted into reports,
    # and a raw chat_id in a repository artifact is what
    # tests/test_no_real_chat_ids exists to prevent.
    print(f"database: {db_path}")
    print(f"distinct content trigrams: {len(trigrams)}")
    print(f"distinct content bigrams:  {len(bigrams)}")
    for min_count in sorted(set(args.min_count)):
        tri = sum(1 for count in trigrams.values() if count >= min_count)
        bi = sum(1 for count in bigrams.values() if count >= min_count)
        print(
            f"  cnt >= {min_count}: trigrams {tri}, bigrams {bi}, "
            f"phrases {tri + bi}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
