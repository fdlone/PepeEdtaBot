"""Solo connectedness rating round for the Phase 9 gate (M3R-020, doc 05 §5).

Two commands with a human between them:

``prepare``
    Runs the matrix on a copy of the snapshot, samples replies per arm, mixes
    in decoys (a real reply of the same run with its tokens shuffled) and
    hidden repeats, and writes ``rating_list.txt`` + ``rating_key.json`` into
    ``rating_rounds/<label>/``. The list is source-blind; without the key it
    cannot be scored, with the key it stops being blind.

``score``
    Reads the key and the owner's answers and writes an aggregate of numbers
    only: connected share per arm, self-agreement on the repeats, share of
    decoys caught, and a validity verdict with its reason. The aggregate is the
    file that may enter the repository — the list and the answers hold verbatim
    chat-derived text and stay with the owner (``rating_rounds/`` is gitignored,
    like the meme round).

Validity thresholds are read from the pre-registered ``phase9_interp`` block,
not duplicated here: the same numbers gate the phase, and two copies would let
the round and the verdict disagree about what a valid round is.

Usage:
    python -m tools.solo_rating_round prepare --db db_prod_copy/markov.db \\
        --label 2026-09-01 --context-mode ctx
    python -m tools.solo_rating_round score --label 2026-09-01 \\
        --answers rating_rounds/2026-09-01/answers.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import log_masking  # noqa: E402
from app.core.failure_taxonomy import FailureClass  # noqa: E402
from app.core.markov import tokenize  # noqa: E402
from tools.eval.config import (  # noqa: E402
    MATRIX_PATH,
    THRESHOLDS_PATH,
    load_matrix,
    load_thresholds,
)
from tools.eval.prompts import PROMPTS_PATH, load_prompts  # noqa: E402
from tools.eval.run import run_matrix  # noqa: E402
from tools.eval_prod import pick_chat_id  # noqa: E402

# Share of positions that are decoys. NOT a gate threshold — what the gate
# reads is how many of them were caught (`decoy_detected_share`), and that bar
# lives in the pre-registered block. This number only has to be large enough to
# resolve the bar and small enough not to eat the sample.
DECOY_SHARE = 0.10

CONNECTED_MIN_SCORE = 2  # doc 05 §5: 1-3 scale, "connected" is 2 or better

HEADER = """\
Раунд оценки связности (docs/SOLO_RATING_ROUND.md).

На каждую позицию — одна оценка связности:
  3 — связный ответ, читается как реплика
  2 — небезупречно, но связно
  1 — бессвязно

Для оценки 1 можно назвать класс отказа через дробь, например `4:1/F3_generic`.
Классы: {classes}.

Ответ одной строкой, например:  1:3 2:1/F1_irrelevant 3:2 ...

"""


def _round_entries(
    replies_by_arm: dict[str, list[str]],
    *,
    rated_min: int,
    repeat_share: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Positions of one round, before shuffling, plus arms short of the minimum.

    Each entry is ``{"text": ..., "arm": ..., "decoy": bool, "item": id}``;
    repeats share an ``item`` with the position they duplicate.
    """
    rng = random.Random(seed)
    entries: list[dict[str, Any]] = []
    below_minimum: list[str] = []
    for arm in sorted(replies_by_arm):
        pool = list(replies_by_arm[arm])
        rng.shuffle(pool)
        sample = pool[:rated_min]
        if len(sample) < rated_min:
            below_minimum.append(arm)
        for index, text in enumerate(sample):
            entries.append(
                {"text": text, "arm": arm, "decoy": False, "item": f"{arm}-{index}"}
            )
        # Decoys are built from replies of THIS arm's own run: a decoy has to
        # differ from a real reply in connectedness alone, so it keeps the
        # vocabulary and the length and loses only the order (design D2).
        # Не меньше одного на арм: раунд без декоев нечем провалидировать, и
        # доля, которая на короткой выборке округляется в ноль, тихо превратила
        # бы скрытый контроль в его отсутствие.
        decoys = max(1, round(len(sample) * DECOY_SHARE)) if sample else 0
        for index, text in enumerate(rng.sample(sample, min(decoys, len(sample)))):
            tokens = tokenize(text)
            if len(tokens) < 3:
                continue  # too short to destroy: shuffling would return itself
            shuffled = tokens[:]
            # Ограниченное число попыток: у ответа из повторяющихся токенов
            # («да да да») перестановки, отличной от исходной, может не быть
            # вовсе, и бесконечный цикл здесь стоил бы раунда.
            for _ in range(10):
                rng.shuffle(shuffled)
                if shuffled != tokens:
                    break
            else:
                continue
            entries.append(
                {
                    "text": " ".join(shuffled),
                    "arm": arm,
                    "decoy": True,
                    "item": f"{arm}-decoy-{index}",
                }
            )

    repeats = round(len(entries) * repeat_share)
    for entry in rng.sample(entries, min(repeats, len(entries))):
        entries.append(dict(entry))
    return entries, below_minimum


def build_round(
    replies_by_arm: dict[str, list[str]],
    *,
    rated_min: int,
    repeat_share: float,
    seed: int,
    context_mode: str = "ctx",
) -> tuple[str, dict[str, Any]]:
    """The blind list and its key. Pure: no DB, no files, no clock."""
    entries, below_minimum = _round_entries(
        replies_by_arm, rated_min=rated_min, repeat_share=repeat_share, seed=seed
    )
    rng = random.Random(seed + 1)
    rng.shuffle(entries)

    classes = ", ".join(item.value for item in FailureClass)
    listing = HEADER.format(classes=classes) + "\n".join(
        f"{position}. {entry['text']}" for position, entry in enumerate(entries, 1)
    )
    key = {
        "seed": seed,
        "context_mode": context_mode,
        "rated_min": rated_min,
        # Arms whose sample fell short are named in the key, not left to be
        # noticed at scoring time: a round that cannot satisfy the minimum is
        # worth knowing about BEFORE the owner spends an evening on it.
        "below_minimum": below_minimum,
        "counts": {
            "positions": len(entries),
            "decoys": sum(1 for entry in entries if entry["decoy"]),
            "repeat_pairs": len(entries) - len({entry["item"] for entry in entries}),
        },
        "positions": {
            str(position): {
                "arm": entry["arm"],
                "decoy": entry["decoy"],
                "item": entry["item"],
            }
            for position, entry in enumerate(entries, 1)
        },
    }
    return listing + "\n", key


ANSWER_RE = re.compile(r"(\d+)\s*:\s*([123])(?:\s*/\s*(\S+))?")


def parse_answers(text: str) -> dict[int, tuple[int, str | None]]:
    """``1:3 2:1/F3_generic`` -> ``{1: (3, None), 2: (1, "F3_generic")}``.

    An unknown failure class is rejected rather than tolerated: the point of
    the taxonomy is that the round and the telemetry name the same phenomenon
    the same way, and a typo silently becoming a new category defeats it.
    """
    answers: dict[int, tuple[int, str | None]] = {}
    known = {item.value for item in FailureClass}
    for position, score, failure_class in ANSWER_RE.findall(text):
        if failure_class and failure_class not in known:
            raise ValueError(
                f"position {position}: unknown failure class {failure_class!r}; "
                f"expected one of {', '.join(sorted(known))}"
            )
        answers[int(position)] = (int(score), failure_class or None)
    return answers


def score_round(
    key: dict[str, Any],
    answers: dict[int, tuple[int, str | None]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate of one round: numbers, and why it is (not) valid.

    Validity and quality are kept apart on purpose. An unstable rater or a
    rubric that cannot tell a shuffled reply from a real one makes the round
    uncountable — `insufficient data`, never a failed phase.
    """
    config = thresholds.get("phase9_interp", {})
    rated_min = int(config.get("manual_rated_min", 30))
    repeat_share_min = float(config.get("manual_repeat_share_min", 0.20))
    self_consistency_min = float(config.get("manual_self_consistency_min", 0.80))
    decoy_detected_min = float(config.get("manual_decoy_detected_min", 0.80))

    positions: dict[str, dict[str, Any]] = key["positions"]
    arms: dict[str, dict[str, int]] = {}
    by_item: dict[str, list[int]] = {}
    item_arm: dict[str, str] = {}
    decoys = decoys_detected = 0
    failure_classes: dict[str, int] = {}

    for position, meta in positions.items():
        answer = answers.get(int(position))
        if answer is None:
            continue
        score, failure_class = answer
        by_item.setdefault(meta["item"], []).append(score)
        if failure_class:
            failure_classes[failure_class] = failure_classes.get(failure_class, 0) + 1
        if meta["decoy"]:
            decoys += 1
            decoys_detected += score < CONNECTED_MIN_SCORE
            continue
        item_arm[meta["item"]] = meta["arm"]

    # Доля связных считается по РАЗЛИЧНЫМ ответам, а не по позициям: повтор
    # заведён, чтобы измерить оценщика, и взвешивать показанный дважды ответ
    # вдвое значило бы дать ему двойной голос в вердикте фазы. Из пары берётся
    # первая оценка — вторая уже потрачена на само-согласие.
    for item, arm in item_arm.items():
        counters = arms.setdefault(arm, {"rated": 0, "connected": 0})
        counters["rated"] += 1
        counters["connected"] += by_item[item][0] >= CONNECTED_MIN_SCORE

    repeat_pairs = [scores for scores in by_item.values() if len(scores) > 1]
    agreements = [
        int(scores[0] == scores[1]) for scores in repeat_pairs if len(scores) == 2
    ]
    rated_total = len(by_item)
    self_agreement = sum(agreements) / len(agreements) if agreements else None
    decoy_detected_share = decoys_detected / decoys if decoys else None
    repeat_share = len(repeat_pairs) / rated_total if rated_total else 0.0

    invalid_reasons: list[str] = []
    if self_agreement is None:
        invalid_reasons.append("no repeated positions were rated")
    elif self_agreement < self_consistency_min:
        invalid_reasons.append(
            f"self-agreement {self_agreement:.2f} below {self_consistency_min:.2f} "
            "(the rater is not stable)"
        )
    if repeat_share < repeat_share_min:
        invalid_reasons.append(
            f"repeats {repeat_share:.0%} of the sample, below {repeat_share_min:.0%}"
        )
    if decoy_detected_share is None:
        invalid_reasons.append("no decoys were rated")
    elif decoy_detected_share < decoy_detected_min:
        invalid_reasons.append(
            f"decoys caught {decoy_detected_share:.0%}, below "
            f"{decoy_detected_min:.0%} (the rubric does not separate connected "
            "from shuffled)"
        )
    short = sorted(arm for arm, counters in arms.items() if counters["rated"] < rated_min)
    if short:
        invalid_reasons.append(
            f"fewer than {rated_min} replies rated for: {', '.join(short)}"
        )

    return {
        "context_mode": key.get("context_mode", "ctx"),
        "seed": key.get("seed"),
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "self_agreement": self_agreement,
        "repeat_pairs": len(repeat_pairs),
        "repeat_share": repeat_share,
        "decoys": decoys,
        "decoys_detected": decoys_detected,
        "decoy_detected_share": decoy_detected_share,
        "arms": {
            arm: {
                "rated": counters["rated"],
                "connected": counters["connected"],
                "connected_share": (
                    counters["connected"] / counters["rated"] if counters["rated"] else None
                ),
            }
            for arm, counters in sorted(arms.items())
        },
        "failure_classes": dict(sorted(failure_classes.items())),
    }


async def prepare(args: argparse.Namespace, out_dir: Path) -> None:
    log_masking.init_masking("solo-rating-round")
    thresholds = load_thresholds(Path(args.thresholds))
    config = thresholds.get("phase9_interp", {})
    db_source = Path(args.db)
    runs, _skipped = await run_matrix(
        db_source=db_source,
        chat_id=pick_chat_id(db_source, args.chat_id),
        configs=load_matrix(Path(args.matrix)),
        prompt_set=load_prompts(Path(args.prompts)),
        seeds=[args.seed],
        generations=args.generations,
        context_mode=args.context_mode,
    )
    replies_by_arm = {
        run.config_id: [record.reply_text for record in run.records if record.success]
        for run in runs.values()
        if run.shared_with is None
    }
    listing, key = build_round(
        replies_by_arm,
        rated_min=int(config.get("manual_rated_min", 30)),
        repeat_share=float(config.get("manual_repeat_share_min", 0.20)),
        seed=args.seed,
        context_mode=args.context_mode,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "rating_list.txt"
    key_path = out_dir / "rating_key.json"
    list_path.write_text(listing, encoding="utf-8")
    key_path.write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Counts and paths only: every reply here is chat-derived text.
    print(f"positions: {key['counts']['positions']}")
    print(
        f"decoys: {key['counts']['decoys']}, "
        f"repeat pairs: {key['counts']['repeat_pairs']}"
    )
    if key["below_minimum"]:
        print(f"below the per-arm minimum: {', '.join(key['below_minimum'])}")
    print(f"list: {list_path}")
    print(f"key : {key_path}")


def score(args: argparse.Namespace, out_dir: Path) -> None:
    thresholds = load_thresholds(Path(args.thresholds))
    key = json.loads((out_dir / "rating_key.json").read_text(encoding="utf-8"))
    answers = parse_answers(Path(args.answers).read_text(encoding="utf-8"))
    aggregate = score_round(key, answers, thresholds)
    out_path = out_dir / "solo_rating.json"
    out_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"valid: {aggregate['valid']}")
    for reason in aggregate["invalid_reasons"]:
        print(f"  - {reason}")
    for arm, counters in aggregate["arms"].items():
        share = counters["connected_share"]
        print(
            f"{arm}: rated {counters['rated']}, connected "
            f"{counters['connected']}" + (f" ({share:.0%})" if share is not None else "")
        )
    print(f"aggregate: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--db", type=str, default="db_prod_copy/markov.db")
    parser.add_argument("--chat-id", type=int, default=None)
    parser.add_argument("--matrix", type=str, default=str(MATRIX_PATH))
    parser.add_argument("--prompts", type=str, default=str(PROMPTS_PATH))
    parser.add_argument("--thresholds", type=str, default=str(THRESHOLDS_PATH))
    parser.add_argument("--context-mode", type=str, default="ctx", choices=("ctx", "noctx"))
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answers", type=str, default=None)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "rating_rounds" / args.label
    if args.command == "prepare":
        asyncio.run(prepare(args, out_dir))
        return
    if not args.answers:
        parser.error("score requires --answers")
    score(args, out_dir)


if __name__ == "__main__":
    main()
