"""Census of order-3 states and a simulation of Phase 9 interpolation.

The Phase 9 spec (docs/v2/M2R_PHASE9_INTERPOLATION.md §1, §2) is argued from six
numbers: how many order-3 states are deterministic, how few sit in the middle
entropy zone where sampling knobs have any leverage, and what interpolating with
the order-2 projection does to both. Those numbers were produced once, by hand,
on a copy of prod. This script makes them reproducible on any copy with one
command, which is the difference between a motivation and an anecdote.

Read-only: opens the database via a URI in ``mode=ro`` and never writes.

    python -m tools.eval.state_census --db db_prod_copy/markov.db
    python -m tools.eval.state_census --db db_prod_copy/markov.db --beta 0.15 0.3 0.5

Zones follow the spec: normalized entropy H_norm < 0.2 is "deterministic-ish"
(a single continuation gives exactly 0), [0.2, 0.8) is the middle zone where
temperature and blending have leverage at all, and >= 0.8 is near-uniform.
Weighting by visits as well as by states is deliberate — 97.9% of states being
deterministic and 94.6% of VISITS landing on them are different claims, and the
spec makes both.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_MIDDLE_LOW = 0.2
_MIDDLE_HIGH = 0.8


def _normalized_entropy(counts: list[float]) -> float:
    """Shannon entropy of the pool, normalized by log of its size.

    A single continuation returns 0.0 — no choice, hence no entropy — which is
    what puts 97.9% of this corpus's states in the deterministic zone.
    """
    if len(counts) <= 1:
        return 0.0
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    bits = -math.fsum(
        (c / total) * math.log2(c / total) for c in counts if c > 0
    )
    return bits / math.log2(len(counts))


def _zone(h_norm: float) -> str:
    if h_norm < _MIDDLE_LOW:
        return "deterministic"
    if h_norm < _MIDDLE_HIGH:
        return "middle"
    return "near-uniform"


def _load(db_path: Path, chat_id: int | None) -> tuple[
    dict[tuple[str, str, str], dict[str, int]],
    dict[tuple[str, str], dict[str, int]],
    int,
]:
    """Order-3 pools, order-2 pools and the resolved chat id."""
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        if chat_id is None:
            row = conn.execute(
                "SELECT chat_id, SUM(cnt) FROM transitions3 "
                "GROUP BY chat_id ORDER BY 2 DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise SystemExit("no order-3 transitions in this database")
            chat_id = int(row[0])
        pools3: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
        for w1, w2, w3, w4, cnt in conn.execute(
            "SELECT w1, w2, w3, w4, cnt FROM transitions3 WHERE chat_id = ?",
            (chat_id,),
        ):
            pools3[(w1, w2, w3)][w4] = int(cnt)
        pools2: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
        for w1, w2, w3, cnt in conn.execute(
            "SELECT w1, w2, w3, cnt FROM transitions WHERE chat_id = ?",
            (chat_id,),
        ):
            pools2[(w1, w2)][w3] = int(cnt)
    return dict(pools3), dict(pools2), chat_id


def _census(pools: dict[tuple[str, ...], tuple[list[float], int]]) -> dict[str, object]:
    """Zone shares over states and over visits.

    Pools carry (weights, visits) rather than counts: entropy must be computed on
    the DISTRIBUTION. Rescaling a merged distribution back to integer counts is
    what makes the simulation lie — on this corpus most states have one or two
    visits, so rounding collapses every merged pool to a row of ones, i.e. to
    uniform, and the middle zone the phase is about disappears into the
    near-uniform one.
    """
    by_zone_states: dict[str, int] = defaultdict(int)
    by_zone_visits: dict[str, int] = defaultdict(int)
    single = 0
    single_visits = 0
    total_visits = 0
    for counts, visits in pools.values():
        total_visits += visits
        zone = _zone(_normalized_entropy(counts))
        by_zone_states[zone] += 1
        by_zone_visits[zone] += visits
        if len(counts) == 1:
            single += 1
            single_visits += visits
    states = len(pools) or 1
    visits = total_visits or 1
    return {
        "states": len(pools),
        "visits": total_visits,
        "single_share": single / states,
        "single_visit_share": single_visits / visits,
        "zones": {
            zone: {
                "states": by_zone_states.get(zone, 0),
                "state_share": by_zone_states.get(zone, 0) / states,
                "visit_share": by_zone_visits.get(zone, 0) / visits,
            }
            for zone in ("deterministic", "middle", "near-uniform")
        },
    }


def _as_pools(
    raw: dict[tuple[str, ...], dict[str, int]],
) -> dict[tuple[str, ...], tuple[list[float], int]]:
    """Stored counts as (weights, visits) — the census input shape."""
    return {
        state: ([float(c) for c in pool.values()], sum(pool.values()))
        for state, pool in raw.items()
    }


def _interpolated(
    pools3: dict[tuple[str, str, str], dict[str, int]],
    pools2: dict[tuple[str, str], dict[str, int]],
    beta: float,
) -> dict[tuple[str, ...], tuple[list[float], int]]:
    """Merged distribution per state, kept as weights, never rescaled to counts.

    Visits stay the order-3 pool's own total so the visit weighting is comparable
    with the untouched census: the question is which zone a state lands in, and
    interpolation does not change how often the walk arrives there.
    """
    merged_pools: dict[tuple[str, ...], tuple[list[float], int]] = {}
    for state, pool3 in pools3.items():
        visits = sum(pool3.values())
        pool2 = pools2.get((state[1], state[2]), {})
        total3 = float(visits) or 1.0
        if not pool2 or set(pool2) <= set(pool3):
            merged_pools[state] = ([float(c) for c in pool3.values()], visits)
            continue
        total2 = float(sum(pool2.values())) or 1.0
        merged: dict[str, float] = {
            token: (1.0 - beta) * (count / total3) for token, count in pool3.items()
        }
        for token, count in pool2.items():
            merged[token] = merged.get(token, 0.0) + beta * (count / total2)
        merged_pools[state] = (list(merged.values()), visits)
    return merged_pools


def _print(label: str, census: dict[str, object]) -> None:
    zones = census["zones"]
    assert isinstance(zones, dict)
    print(f"\n{label}")
    print(f"  states: {census['states']}   visits: {census['visits']}")
    print(
        f"  exactly one continuation: {census['single_share']:.3f} of states, "
        f"{census['single_visit_share']:.3f} of visits"
    )
    for zone in ("deterministic", "middle", "near-uniform"):
        data = zones[zone]
        print(
            f"  {zone:<14} states {data['state_share']:.3f} "
            f"({data['states']})   visits {data['visit_share']:.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="db_prod_copy/markov.db")
    parser.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="default: the chat with the largest order-3 model",
    )
    parser.add_argument(
        "--beta",
        type=float,
        nargs="*",
        default=[0.3],
        help="interpolation weights to simulate (default: 0.3, the spec's number)",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"no such database: {db_path}")
    pools3, pools2, chat_id = _load(db_path, args.chat_id)
    # Chat id is deliberately not printed: this output is pasted into reports,
    # and a raw chat_id in a repository artifact is what tests/test_no_real_chat_ids
    # exists to prevent.
    print(f"database: {db_path}   order-3 states: {len(pools3)}")
    _print("order-3 as stored", _census(_as_pools(dict(pools3))))
    _print("order-2 (for comparison)", _census(_as_pools(dict(pools2))))
    for beta in args.beta:
        if not 0.0 <= beta <= 1.0:
            raise SystemExit(f"beta out of range: {beta}")
        _print(
            f"order-3 interpolated with order-2 projection, beta = {beta}",
            _census(_interpolated(pools3, pools2, beta)),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
