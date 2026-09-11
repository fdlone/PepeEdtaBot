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
import math
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

Каждую позицию оценивайте как незнакомый текст, даже если кажется, что такой
уже был: повтор в списке — намеренный контроль, и одинаковые тексты обязаны
получить одинаковую оценку. «Уже видел» — не оценка.
Список разбит на сессии: одна сессия — один присест, следующую — в другой день.

"""

SESSION_HEADER = "=== Сессия {session} из {sessions} — оценивать отдельно, с перерывом ==="

# Positions per session before repeats. A session is one sitting: the
# 2026-09-11 rounds showed one rater does not hold a 75-position list —
# the later presentation of an identical text was rated lower in 20 of 21
# disagreeing pairs. Not a gate threshold: the validity bars are unchanged
# and read over the whole round; sessions change where the repeats sit.
DEFAULT_SESSION_SIZE = 25

FORM_TEMPLATE_PATH = PROJECT_ROOT / "tools" / "rating_form_template.html"


def _round_entries(
    replies_by_arm: dict[str, list[str]],
    *,
    rated_min: int,
    repeat_share: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Positions of one round, before shuffling, plus arms short of the minimum.

    Each entry is ``{"text": ..., "arm": ..., "arms": [...], "decoy": bool,
    "item": id}``; repeats share an ``item`` with the position they duplicate.

    Одинаковый текст из нескольких рук — одна запись: связность есть свойство
    текста, и второе предъявление той же строки под другой рукой измеряло бы
    дрейф оценщика, а не связность (в `l1-route-v4` таких групп было 29, на
    них само-согласие 8/33). ``arms`` перечисляет все руки, породившие текст,
    ``arm``/``item`` — первую из них, чтобы старые ключи читались как прежде.
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
            _add_merging_identical(
                entries,
                {"text": text, "arm": arm, "arms": [arm], "decoy": False, "item": f"{arm}-{index}"},
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
            _add_merging_identical(
                entries,
                {
                    "text": " ".join(shuffled),
                    "arm": arm,
                    "arms": [arm],
                    "decoy": True,
                    "item": f"{arm}-decoy-{index}",
                },
            )

    return entries, below_minimum


def _split_into_sessions(
    entries: list[dict[str, Any]], *, session_size: int, rng: random.Random
) -> list[list[dict[str, Any]]]:
    """Distinct entries -> sessions of about ``session_size``, each with a decoy.

    Decoys are dealt round-robin so no session is left without its control;
    the rest is shuffled and cut evenly. Sessions are sittings, not arms: a
    session mixes every arm the same way the whole list did.
    """
    decoys = [entry for entry in entries if entry["decoy"]]
    reals = [entry for entry in entries if not entry["decoy"]]
    # Never more sessions than decoys: a sitting without its decoy control
    # would be a sitting nobody can validate, so a short round gets fewer,
    # longer sessions instead.
    sessions_count = max(1, min(math.ceil(len(entries) / session_size), len(decoys) or 1))
    rng.shuffle(decoys)
    rng.shuffle(reals)
    sessions: list[list[dict[str, Any]]] = [[] for _ in range(sessions_count)]
    for index, entry in enumerate(decoys):
        sessions[index % sessions_count].append(entry)
    # Fill the shortest session first so sizes stay even after the decoys.
    for entry in reals:
        min(sessions, key=len).append(entry)
    return [session for session in sessions if session]


def _add_repeats(
    session: list[dict[str, Any]], *, repeat_share: float, rng: random.Random
) -> list[dict[str, Any]]:
    """Append the session's own repeats, rounded UP to the share.

    Вверх, а не к ближайшему: валидность сверяет ДОЛЮ повторов с тем же
    порогом, поэтому округление вниз собирает раунд, невалидный при любом
    качестве оценки (26/132 = 19.70% при пороге 20%). Промах систематический —
    при 2 и 4 руках он был, при 3 и 5 его не было. Повторы берутся внутри
    сессии: контроль само-согласия обязан помещаться в один присест.
    """
    repeats = math.ceil(len(session) * repeat_share)
    return session + [dict(entry) for entry in rng.sample(session, min(repeats, len(session)))]


def _add_merging_identical(entries: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    """Append ``entry`` unless an entry with the same normalized text exists.

    Equality is over ``tokenize`` output, so case and punctuation do not split
    what the rater would read as the same reply. A decoy never merges with a
    real reply: a decoy differs from every real text by construction.
    """
    normalized = tuple(tokenize(entry["text"]))
    for other in entries:
        if other["decoy"] == entry["decoy"] and tuple(tokenize(other["text"])) == normalized:
            if entry["arm"] not in other["arms"]:
                other["arms"].append(entry["arm"])
            return
    entries.append(entry)


def _shuffle_spreading_repeats(
    entries: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    """Перемешать так, чтобы два предъявления одной позиции не встали рядом.

    Рядом стоящий дубль узнаётся, и само-согласие тогда меряет память оценщика,
    а не устойчивость его суждения (design D3).

    Дубли вставляются по одному в перемешанный список различных элементов, в
    слот, не соседний со своим близнецом. Вставка одного элемента не может
    свести вместе два других, поэтому инвариант держится по построению:
    запрещённых слотов ровно два из ``len + 1``, подходящий есть всегда.
    Перетасовка до успеха дала бы то же в среднем, но без верхней границы по
    времени — а простого ``shuffle`` не хватает: соседи выпадали примерно в
    трети раундов.
    """
    ordered: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        (duplicates if entry["item"] in seen else ordered).append(entry)
        seen.add(entry["item"])

    rng.shuffle(ordered)
    rng.shuffle(duplicates)
    for entry in duplicates:
        twin = next(
            index for index, other in enumerate(ordered) if other["item"] == entry["item"]
        )
        # Вставка в слот k ставит элемент между k-1 и k, поэтому соседним с
        # близнецом он оказывается ровно при k == twin и k == twin + 1.
        slots = [k for k in range(len(ordered) + 1) if k not in (twin, twin + 1)]
        ordered.insert(rng.choice(slots), entry)
    return ordered


def build_round(
    replies_by_arm: dict[str, list[str]],
    *,
    rated_min: int,
    repeat_share: float,
    seed: int,
    context_mode: str = "ctx",
    session_size: int = DEFAULT_SESSION_SIZE,
) -> tuple[str, dict[str, Any]]:
    """The blind list and its key. Pure: no DB, no files, no clock.

    The list is cut into sessions of about ``session_size`` distinct positions
    (design D1 of round-sessions): repeats and decoys live inside a session,
    positions are numbered through the whole list, and the key names the
    session of every position so the aggregate can show the controls per
    sitting. Validity is still read over the whole round.
    """
    distinct, below_minimum = _round_entries(
        replies_by_arm, rated_min=rated_min, repeat_share=repeat_share, seed=seed
    )
    rng = random.Random(seed + 1)
    sessions = [
        _shuffle_spreading_repeats(
            _add_repeats(session, repeat_share=repeat_share, rng=rng), rng
        )
        for session in _split_into_sessions(distinct, session_size=session_size, rng=rng)
    ]
    entries: list[dict[str, Any]] = []
    session_of: list[int] = []
    for number, session in enumerate(sessions, 1):
        entries.extend(session)
        session_of.extend([number] * len(session))

    classes = ", ".join(item.value for item in FailureClass)
    lines: list[str] = []
    for position, (entry, session) in enumerate(zip(entries, session_of), 1):
        if position == 1 or session_of[position - 2] != session:
            lines.append("")
            lines.append(SESSION_HEADER.format(session=session, sessions=len(sessions)))
        lines.append(f"{position}. {entry['text']}")
    listing = HEADER.format(classes=classes) + "\n".join(lines).lstrip("\n")
    key = {
        "seed": seed,
        "context_mode": context_mode,
        "rated_min": rated_min,
        "session_size": session_size,
        "sessions": len(sessions),
        # Arms whose sample fell short are named in the key, not left to be
        # noticed at scoring time: a round that cannot satisfy the minimum is
        # worth knowing about BEFORE the owner spends an evening on it.
        "below_minimum": below_minimum,
        "counts": {
            "positions": len(entries),
            "decoys": sum(1 for entry in entries if entry["decoy"]),
            "repeat_pairs": len(entries) - len({entry["item"] for entry in entries}),
            "per_session": [len(session) for session in sessions],
        },
        "positions": {
            str(position): {
                "arm": entry["arm"],
                "arms": list(entry["arms"]),
                "decoy": entry["decoy"],
                "item": entry["item"],
                "session": session,
            }
            for position, (entry, session) in enumerate(zip(entries, session_of), 1)
        },
    }
    return listing + "\n", key


def render_form(template: str, *, label: str, items: list[dict[str, Any]]) -> str:
    """The rating form for one session: the template with its two placeholders
    filled. ``items`` are ``{"n": global position, "text": ...}``; the form
    stores progress under ``label``, so each session gets its own label."""
    assert "__ITEMS__" in template and "__LABEL__" in template
    return template.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)).replace(
        "__LABEL__", label
    )


def session_items(listing: str, key: dict[str, Any], session: int) -> list[dict[str, Any]]:
    """Positions of one session as form items, read back from the list."""
    texts = {
        int(line.split(". ", 1)[0]): line.split(". ", 1)[1]
        for line in listing.splitlines()
        if ". " in line and line.split(". ", 1)[0].isdigit()
    }
    return [
        {"n": int(position), "text": texts[int(position)]}
        for position, meta in key["positions"].items()
        if meta.get("session", 1) == session
    ]


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
    item_arms: dict[str, list[str]] = {}
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
        # ``arms`` — все руки, породившие этот текст (склейка одинаковых);
        # ключи до склейки несут только ``arm``.
        item_arms[meta["item"]] = list(meta.get("arms") or [meta["arm"]])

    # Доля связных считается по РАЗЛИЧНЫМ ответам, а не по позициям: повтор
    # заведён, чтобы измерить оценщика, и взвешивать показанный дважды ответ
    # вдвое значило бы дать ему двойной голос в вердикте фазы. Из пары берётся
    # первая оценка — вторая уже потрачена на само-согласие.
    for item, item_arm_list in item_arms.items():
        for arm in item_arm_list:
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

    # Per-session controls are diagnostics, not validity: the bars are
    # pre-registered over the whole round. They show WHICH sitting drifted.
    sessions: dict[str, dict[str, Any]] = {}
    for position, meta in positions.items():
        answer = answers.get(int(position))
        session = str(meta.get("session", 1))
        bucket = sessions.setdefault(
            session, {"rated": 0, "decoys": 0, "decoys_detected": 0, "pairs": [], "_seen": {}}
        )
        if answer is None:
            continue
        bucket["rated"] += 1
        if meta["decoy"]:
            bucket["decoys"] += 1
            bucket["decoys_detected"] += answer[0] < CONNECTED_MIN_SCORE
        seen = bucket["_seen"]
        if meta["item"] in seen:
            bucket["pairs"].append(int(seen[meta["item"]] == answer[0]))
        else:
            seen[meta["item"]] = answer[0]
    per_session = {
        session: {
            "rated": bucket["rated"],
            "self_agreement": (
                sum(bucket["pairs"]) / len(bucket["pairs"]) if bucket["pairs"] else None
            ),
            "decoys_detected": f"{bucket['decoys_detected']}/{bucket['decoys']}",
        }
        for session, bucket in sorted(sessions.items(), key=lambda item: int(item[0]))
    }

    return {
        "context_mode": key.get("context_mode", "ctx"),
        "seed": key.get("seed"),
        "valid": not invalid_reasons,
        "sessions": per_session,
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


def unusable_reasons(key: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Почему собранный раунд не посчитается даже у безупречного оценщика.

    Вопрос задаётся самой валидации: ей подставляются ответы, в которых все
    настоящие позиции связны, а все декои опознаны, и всё, что она после этого
    называет, — свойство **списка**, а не оценщика. Второго списка условий при
    этом не заводится: если завтра в валидность добавится условие о составе,
    предупреждение подхватит его само — по той же причине, по которой пороги не
    дублируются в инструменте.
    """
    answers = {
        int(position): (1 if meta["decoy"] else 3, None)
        for position, meta in key["positions"].items()
    }
    return list(score_round(key, answers, thresholds)["invalid_reasons"])


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
        session_size=args.session_size,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / "rating_list.txt"
    key_path = out_dir / "rating_key.json"
    list_path.write_text(listing, encoding="utf-8")
    key_path.write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    template = FORM_TEMPLATE_PATH.read_text(encoding="utf-8")
    for session in range(1, key["sessions"] + 1):
        form = render_form(
            template,
            label=f"{args.label} · сессия {session}/{key['sessions']}",
            items=session_items(listing, key, session),
        )
        (out_dir / f"form_s{session}.html").write_text(form, encoding="utf-8")
    # Counts and paths only: every reply here is chat-derived text.
    print(
        f"positions: {key['counts']['positions']} in {key['sessions']} session(s) "
        f"{key['counts']['per_session']}"
    )
    print(
        f"decoys: {key['counts']['decoys']}, "
        f"repeat pairs: {key['counts']['repeat_pairs']}"
    )
    if key["below_minimum"]:
        print(f"below the per-arm minimum: {', '.join(key['below_minimum'])}")
    # Раунд, который не сможет быть посчитан, стоит увидеть ДО того, как на него
    # потрачен вечер — и не только по выборке ниже минимума: округление числа
    # повторов вниз однажды собрало раунд, невалидный при любом качестве оценки,
    # и это выяснилось после того, как его оценили два человека.
    for reason in unusable_reasons(key, thresholds):
        print(f"cannot be scored: {reason}")
    print(f"list: {list_path}")
    print(f"key : {key_path}")
    print(f"forms: {out_dir / 'form_s<N>.html'} — one per session, answers merge by position")


def score(args: argparse.Namespace, out_dir: Path) -> None:
    thresholds = load_thresholds(Path(args.thresholds))
    key = json.loads((out_dir / "rating_key.json").read_text(encoding="utf-8"))
    answers: dict[int, tuple[int, str | None]] = {}
    paths = [args.answers] if isinstance(args.answers, str) else list(args.answers)
    for path in paths:
        answers.update(parse_answers(Path(path).read_text(encoding="utf-8")))
    aggregate = score_round(key, answers, thresholds)
    out_path = out_dir / "solo_rating.json"
    out_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"valid: {aggregate['valid']}")
    for reason in aggregate["invalid_reasons"]:
        print(f"  - {reason}")
    for session, controls in aggregate.get("sessions", {}).items():
        agreement = controls["self_agreement"]
        print(
            f"session {session}: rated {controls['rated']}, self-agreement "
            + ("n/a" if agreement is None else f"{agreement:.2f}")
            + f", decoys {controls['decoys_detected']}"
        )
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
    parser.add_argument(
        "--answers",
        type=str,
        action="append",
        default=None,
        help="answers file; repeat the flag for one file per session",
    )
    parser.add_argument(
        "--session-size",
        type=int,
        default=DEFAULT_SESSION_SIZE,
        help="distinct positions per sitting before repeats (default 25)",
    )
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
