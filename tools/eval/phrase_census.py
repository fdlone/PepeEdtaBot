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
the number cannot drift from the extraction the index actually uses.

A raw phrase count is not enough to choose a support threshold, which is what
the count gets used for. Two numbers next to it say what the threshold really
buys:

* **self-standing phrases** — a bigram that is a slice of a trigram which also
  passed is not new material; it competes for the same pool slot with a near
  duplicate, which the escape metric scores as the same trajectory, not another
  one;
* **attachment points** — a phrase can only be spliced where its opening is an
  existing order-2 state of the chain, so this bounds the share of inputs a
  phrase route could touch at all. That bound is coverage, and the protocol
  requires coverage in every gate.

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

from app.core.hot_ngrams import is_content_ngram


def _counts(
    db_path: Path, chat_id: int | None
) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]], set[tuple[str, str]]]:
    """All-time content trigram and bigram counts, plus the chain's states.

    The states are every ``(w1, w2)`` the chain holds, punctuation included:
    a splice point is a state of the chain, and whether it happens to be a
    content pair has nothing to do with it.
    """
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
        states: set[tuple[str, str]] = set()
        for w1, w2, w3, cnt in conn.execute(
            "SELECT w1, w2, w3, cnt FROM transitions WHERE chat_id = ?",
            (chat_id,),
        ):
            states.add((w1, w2))
            if is_content_ngram((w1, w2, w3)):
                trigrams[(w1, w2, w3)] += int(cnt)
            # The bigram's all-time count is the row group's sum over w3 — the
            # same aggregation ChatHotNgramsRepo.get_hot uses for its bigram arm.
            if is_content_ngram((w1, w2)):
                bigrams[(w1, w2)] += int(cnt)
    return trigrams, bigrams, states


def _self_standing(
    trigrams: set[tuple[str, ...]], bigrams: set[tuple[str, ...]]
) -> int:
    """Phrases that are not a slice of a trigram which also passed.

    A bigram inside a passing trigram adds a candidate, not a trajectory: the
    two share every edge the shorter one has.
    """
    redundant = {
        bigram
        for bigram in bigrams
        if any(tri[:2] == bigram or tri[1:] == bigram for tri in trigrams)
    }
    return len(trigrams) + len(bigrams) - len(redundant)


def _attachment_points(
    trigrams: set[tuple[str, ...]],
    bigrams: set[tuple[str, ...]],
    states: set[tuple[str, str]],
) -> int:
    """Distinct chain states a passing phrase could be spliced at.

    An upper bound on coverage, and deliberately generous: it asks only whether
    the opening exists in the chain, not whether the scorer would keep the
    result.
    """
    openings = {phrase[:2] for phrase in trigrams} | {
        (phrase[0], phrase[1]) for phrase in bigrams
    }
    return len(openings & states)


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
    trigrams, bigrams, states = _counts(db_path, args.chat_id)

    # Chat id is deliberately not printed: this output is pasted into reports,
    # and a raw chat_id in a repository artifact is what
    # tests/test_no_real_chat_ids exists to prevent.
    print(f"database: {db_path}")
    print(f"distinct content trigrams: {len(trigrams)}")
    print(f"distinct content bigrams:  {len(bigrams)}")
    print(f"order-2 states in the chain: {len(states)}")

    # Where the mass actually sits. Printed because a threshold chosen without
    # the shape of the distribution is chosen blind.
    histogram: Counter[int] = Counter()
    for count in (*trigrams.values(), *bigrams.values()):
        histogram[min(count, 10)] += 1
    print("support histogram (10 = 10 or more):")
    for support in sorted(histogram):
        print(f"  {support:>3}: {histogram[support]}")

    for min_count in sorted(set(args.min_count)):
        tri = {p for p, c in trigrams.items() if c >= min_count}
        bi = {p for p, c in bigrams.items() if c >= min_count}
        attachable = _attachment_points(tri, bi, states)
        print(
            f"  cnt >= {min_count}: trigrams {len(tri)}, bigrams {len(bi)}, "
            f"phrases {len(tri) + len(bi)}, "
            f"self-standing {_self_standing(tri, bi)}, "
            f"attachment points {attachable} "
            f"({attachable / len(states):.1%} of states)"
            if states
            else f"  cnt >= {min_count}: phrases {len(tri) + len(bi)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
