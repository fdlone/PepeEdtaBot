"""Behavioural probe of the reply-generation mechanics (M3R-050).

Answers, on a prod-copy snapshot and without touching ``app/``:

* which route each generation actually took (start source, order, jumps,
  early exits, silent degradations) and how often;
* what the scorer's components really do to the pool, per candidate origin;
* how the pipeline behaves in the regimes the eval protocol never exercises —
  a reply with NO context (the bot's unprompted, non-reply path) and a chat
  whose anti-repeat memory is warm (``recent_replies`` non-empty).

Everything is captured through hooks that already exist (``gen_trace_log.*``)
plus wrappers around private methods; no behaviour is changed. The wrappers are
read-only: they call the original and record what it returned.

Usage
-----
    python tools/sweeps/genmap_2026-08-13.py run  --arm ctx   --out ctx.json
    python tools/sweeps/genmap_2026-08-13.py run  --arm noctx --out noctx.json
    python tools/sweeps/genmap_2026-08-13.py report ctx.json noctx.json

Arms
----
``ctx``     context = the prompt (what tools/eval/run.py does for every run)
``noctx``   context = []          (prod: unprompted reply to a non-reply message)
``live``    ``noctx`` + warm ``recent_replies``/``recent_short_replies``
``livectx`` ``ctx``   + warm anti-repeat memory
``seeded``  ``ctx``   + ``markov_seeded_candidate_ratio=0.3``
``hotseed`` ``noctx`` + an L1 hot-ngram seed on every generation

``reply_flavor_strength`` and ``emoji_append_chance`` are zeroed in every arm so
the arms differ only where the arm says they do; both layers act after
selection and cannot change the pool.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import log_masking  # noqa: E402
from app.config.registry import RUNTIME_FIELDS  # noqa: E402
from app.core import gen_trace_log  # noqa: E402
from app.core.markov import MarkovGenerator, tokenize  # noqa: E402
from app.core.response_generator import (  # noqa: E402
    CANDIDATE_TARGET,
    GENERATION_ATTEMPT_BUDGET,
    GenerationRequest,
    ResponseGenerator,
    normalize_reply_for_repeat,
)
from app.core.text import sanitize_text  # noqa: E402
from app.infrastructure.database import Database  # noqa: E402
from tools.eval.prompts import load_prompts  # noqa: E402
from tools.eval.run import _populate_token_df  # noqa: E402
from tools.eval_prod import (  # noqa: E402
    _ProdVerbatimChecker,
    _TraceCapturingGenerator,
    copy_database,
    load_messages,
    pick_chat_id,
)

SCORE_FIELDS = (
    "completion_quality",
    "natural_length",
    "context_relevance",
    "collocation_delta",
    "repetition_penalty",
    "recent_penalty",
    "verbatim_penalty",
)

ARMS: dict[str, dict[str, Any]] = {
    "ctx": {"context": True, "live_memory": False, "hot_seed": False, "set": {}},
    "noctx": {"context": False, "live_memory": False, "hot_seed": False, "set": {}},
    "live": {"context": False, "live_memory": True, "hot_seed": False, "set": {}},
    "livectx": {"context": True, "live_memory": True, "hot_seed": False, "set": {}},
    "seeded": {
        "context": True,
        "live_memory": False,
        "hot_seed": False,
        "set": {"markov_seeded_candidate_ratio": 0.3},
    },
    "hotseed": {
        "context": False,
        "live_memory": False,
        "hot_seed": True,
        "set": {},
    },
}


class Probe:
    """Per-generation capture of every hook and wrapper."""

    def __init__(self) -> None:
        self.generations: list[dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self.pool: list[dict[str, Any]] = []
        self.selected: str | None = None
        self.accepted: dict[str, dict[str, Any]] = {}
        self.rejected: list[dict[str, str]] = []
        self.duplicates = 0
        self.failed = 0
        self.extended = 0
        self.extend_calls = 0
        self.mutation_calls = 0
        self.mutated: set[str] = set()
        self.seeded: set[str] = set()
        self.attempts: list[dict[str, Any]] = []
        self.anchor_deferred = 0
        self.anchor_spliced = 0

    # --- gen_trace_log replacements -------------------------------------
    def on_accepted(
        self, _chat_id, attempt, *, context_used, index, trace, score, text
    ) -> None:
        self.accepted[text] = {
            "attempt": attempt,
            "context_used": context_used,
            "start_source": getattr(trace, "start_source", "?"),
            "order_used": getattr(trace, "markov_order_used", 0),
            "jumps": getattr(trace, "jump_count", -1),
            "branching": float(getattr(trace, "mean_branching", 0.0) or 0.0),
        }

    def on_rejected(self, _chat_id, attempt, *, context_used, reason, text) -> None:
        self.rejected.append({"attempt": attempt, "reason": reason})

    def on_duplicate(self, _chat_id, attempt, *, context_used) -> None:
        self.duplicates += 1

    def on_failed(self, _chat_id, attempt, *, context_used) -> None:
        self.failed += 1

    def on_extended(self, _chat_id, attempt, *, original, extended) -> None:
        self.extended += 1

    def on_mutated(self, _chat_id, attempt, *, original, mutated, index, score) -> None:
        self.mutated.add(mutated)

    def on_selection(
        self,
        _chat_id,
        candidates,
        *,
        temperature,
        margin,
        selection_margin_used,
        selected=None,
    ) -> None:
        self.pool = [
            {
                "text": c.text,
                **{name: getattr(c.score, name) for name in SCORE_FIELDS},
                "total": c.score.total,
            }
            for c in (candidates or ())
        ]
        self.selected = None if selected is None else selected.text

    def install(self) -> dict[str, Any]:
        names = (
            "log_attempt_accepted",
            "log_attempt_rejected",
            "log_attempt_duplicate",
            "log_attempt_failed",
            "log_attempt_extended",
            "log_attempt_mutated",
            "log_selection",
        )
        saved = {name: getattr(gen_trace_log, name) for name in names}
        gen_trace_log.log_attempt_accepted = self.on_accepted
        gen_trace_log.log_attempt_rejected = self.on_rejected
        gen_trace_log.log_attempt_duplicate = self.on_duplicate
        gen_trace_log.log_attempt_failed = self.on_failed
        gen_trace_log.log_attempt_extended = self.on_extended
        gen_trace_log.log_attempt_mutated = self.on_mutated
        gen_trace_log.log_selection = self.on_selection
        gen_trace_log.configure(True)  # hooks are no-ops unless "enabled"
        return saved

    @staticmethod
    def restore(saved: dict[str, Any]) -> None:
        for name, fn in saved.items():
            setattr(gen_trace_log, name, fn)
        gen_trace_log.configure(False)

    def finish(
        self, prompt: str, category: str, latency_ms: float, reply: str | None
    ) -> None:
        for entry in self.pool:
            text = entry["text"]
            meta = self.accepted.get(text, {})
            entry.update(
                origin=(
                    "seeded"
                    if text in self.seeded
                    else "mutated"
                    if text in self.mutated
                    else "organic"
                ),
                start_source=meta.get("start_source", "?"),
                order_used=meta.get("order_used", 0),
                jumps=meta.get("jumps", -1),
                branching=meta.get("branching", 0.0),
                attempt=meta.get("attempt", -1),
                won=(text == self.selected),
            )
        self.generations.append(
            {
                "prompt": prompt,
                "category": category,
                "latency_ms": latency_ms,
                "reply": reply,
                "pool": self.pool,
                "rejected": self.rejected,
                "duplicates": self.duplicates,
                "failed": self.failed,
                "extended": self.extended,
                "extend_calls": self.extend_calls,
                "mutation_calls": self.mutation_calls,
                "mutations_accepted": len(self.mutated),
                "attempts": self.attempts,
                "anchor_deferred": self.anchor_deferred,
                "anchor_spliced": self.anchor_spliced,
            }
        )
        self.reset()


def _install_wrappers(probe: Probe) -> list[tuple[type, str, Any]]:
    """Wrap private methods so the branches gen_trace_log cannot see are counted."""
    saved: list[tuple[type, str, Any]] = []

    original_once = MarkovGenerator._generate_text_once

    async def traced_once(self, *args, **kwargs):
        attempt = await original_once(self, *args, **kwargs)
        probe.attempts.append(
            {
                "start_source": attempt.start_source,
                "order_used": attempt.markov_order_used,
                "jumps": attempt.jump_count,
                "rejection": attempt.rejection_reason,
                "tokens": attempt.token_count,
                "branching": attempt.mean_branching,
                "context_used": bool(kwargs.get("context_tokens")),
                "hidden_fallback": attempt.hidden_context_fallbacks,
                "lead_punct": attempt.leading_punctuation_stripped,
            }
        )
        return attempt

    original_loop = MarkovGenerator._run_generation_loop

    async def traced_loop(self, *args, **kwargs):
        result = await original_loop(self, *args, **kwargs)
        if kwargs.get("anchor_state") is not None:
            probe.anchor_deferred += 1
            if result[3]:
                probe.anchor_spliced += 1
        return result

    original_extend = ResponseGenerator._extend_verbatim_candidate

    async def traced_extend(self, *args, **kwargs):
        probe.extend_calls += 1
        return await original_extend(self, *args, **kwargs)

    original_mutate = ResponseGenerator._mutated_variant

    async def traced_mutate(self, *args, **kwargs):
        probe.mutation_calls += 1
        return await original_mutate(self, *args, **kwargs)

    original_seeded = ResponseGenerator._append_seeded_candidates

    async def traced_seeded(self, *args, **kwargs):
        result = await original_seeded(self, *args, **kwargs)
        probe.seeded = set(result)
        return result

    for owner, name, replacement, original in (
        (MarkovGenerator, "_generate_text_once", traced_once, original_once),
        (MarkovGenerator, "_run_generation_loop", traced_loop, original_loop),
        (
            ResponseGenerator,
            "_extend_verbatim_candidate",
            traced_extend,
            original_extend,
        ),
        (ResponseGenerator, "_mutated_variant", traced_mutate, original_mutate),
        (
            ResponseGenerator,
            "_append_seeded_candidates",
            traced_seeded,
            original_seeded,
        ),
    ):
        saved.append((owner, name, original))
        setattr(owner, name, replacement)
    return saved


def _restore_wrappers(saved: list[tuple[type, str, Any]]) -> None:
    for owner, name, original in saved:
        setattr(owner, name, original)


async def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    arm = ARMS[args.arm]
    log_masking.init_masking("genmap-probe")
    prompt_set = load_prompts()
    db_copy, temp_dir = copy_database(Path(args.db))
    probe = Probe()
    saved_hooks = probe.install()
    saved_wrappers = _install_wrappers(probe)
    try:
        chat_id = pick_chat_id(db_copy, args.chat_id)
        messages = load_messages(db_copy, chat_id)
        db = Database(str(db_copy))
        await db.init()
        try:
            overrides: dict[str, Any] = {
                "reply_flavor_strength": 0.0,
                "emoji_append_chance": 0.0,
                **arm["set"],
            }
            for item in args.set or ():
                key, _, raw = item.partition("=")
                overrides[key] = json.loads(raw)
            if overrides.get("markov_seeded_candidate_ratio", 0.0) > 0.0:
                await _populate_token_df(db, chat_id, messages)

            alltime = frozenset(
                tuple(row) for row in await db.get_verbatim_ngrams(chat_id)
            )
            checker = _ProdVerbatimChecker(messages, ngram_index=alltime, db=db)
            generator = _TraceCapturingGenerator(db.markov)
            runtime_state = SimpleNamespace(
                **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS}
            )
            runtime_state.recent_short_replies = {}
            runtime_state.recent_replies = {}
            for key, value in overrides.items():
                if not hasattr(runtime_state, key):
                    raise KeyError(f"unknown knob {key!r}")
                setattr(runtime_state, key, value)

            rg = ResponseGenerator(
                generator=generator,
                learning_service=checker,
                runtime_state=runtime_state,
            )
            hot_ngrams: list[tuple[str, ...]] = []
            if arm["hot_seed"]:
                hot_ngrams = await checker.get_hot_ngrams(
                    chat_id,
                    min_count=runtime_state.hot_ngram_min_count,
                    recency_share=runtime_state.hot_ngram_recency_share,
                )

            per_category = max(1, args.generations // len(prompt_set.categories))
            index = 0
            for category, prompts in prompt_set.categories.items():
                selector = random.Random(args.seed)
                order = list(range(len(prompts)))
                selector.shuffle(order)
                for i in range(per_category):
                    prompt = prompts[order[i % len(order)]]
                    rng = random.Random(args.seed * 100_000 + index)
                    index += 1
                    generator.reset_generation()
                    seed_tokens = (
                        list(rng.choice(hot_ngrams)) if hot_ngrams else None
                    )
                    started = time.perf_counter()
                    result = await rg.generate_with_result(
                        GenerationRequest(
                            chat_id=chat_id,
                            context_tokens=(
                                tokenize(prompt) if arm["context"] else []
                            ),
                            seed=seed_tokens,
                            current_message_normalized=sanitize_text(prompt).lower(),
                            now=args.moment,
                        ),
                        rng=rng,
                        candidate_target=args.target or CANDIDATE_TARGET,
                    )
                    probe.finish(
                        prompt,
                        category,
                        (time.perf_counter() - started) * 1000,
                        result.text,
                    )
                    if arm["live_memory"] and result.text:
                        _remember(runtime_state, chat_id, result.text)
        finally:
            await db.close()
    finally:
        _restore_wrappers(saved_wrappers)
        Probe.restore(saved_hooks)
        temp_dir.cleanup()

    return {
        "arm": args.arm,
        "db": str(args.db),
        "seed": args.seed,
        "overrides": overrides,
        "prompt_set_version": prompt_set.version,
        "generations": probe.generations,
    }


def _remember(runtime_state: Any, chat_id: int, reply_text: str) -> None:
    """Mirror what ReplyPipeline does after a reply is sent (anti-repeat memory)."""
    normalized = normalize_reply_for_repeat(reply_text)
    if normalized:
        recent = runtime_state.recent_replies.setdefault(chat_id, deque(maxlen=20))
        recent.append(normalized)
    tokens = tokenize(reply_text, normalize_lower=runtime_state.normalize_lower)
    content = [t for t in tokens if t not in {".", ",", "!", "?", ";", ":"}]
    if 0 < len(content) <= 3:
        short = runtime_state.recent_short_replies.setdefault(
            chat_id, deque(maxlen=5)
        )
        short.append(sanitize_text(reply_text).lower())


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def report(paths: list[Path]) -> None:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    print(f"prompt set: {runs[0]['prompt_set_version']}   db: {runs[0]['db']}\n")

    print("== A. Итог генерации ==")
    header = f"{'метрика':<34}" + "".join(f"{r['arm'] + '/' + str(r['seed']):>14}" for r in runs)
    print(header)
    rows: list[tuple[str, list[str]]] = []

    def add(name: str, fn) -> None:
        rows.append((name, [fn(r) for r in runs]))

    # Попытки основного цикла: каждая заканчивается ровно одним событием —
    # failed / rejected / duplicate / accepted. Мутанты и seeded попыток не
    # тратят, а вызовы обхода из verbatim-дописки идут мимо цикла, поэтому
    # считать по числу проходок ``_generate_text_once`` нельзя.
    def attempts_of(g: dict[str, Any]) -> int:
        organic = sum(1 for c in g["pool"] if c["origin"] == "organic")
        return g["failed"] + g["duplicates"] + len(g["rejected"]) + organic

    def per_gen(fn, digits: int = 3):
        """Среднее по генерациям арма, отформатированное для колонки."""
        return lambda r: format(_mean([fn(g) for g in r["generations"]]), f".{digits}f")

    add("генераций", lambda r: str(len(r["generations"])))
    add("пустой ответ", per_gen(lambda g: 0.0 if g["reply"] else 1.0))
    add("размер пула (среднее)", per_gen(lambda g: len(g["pool"]), 2))
    add("пул == 5", per_gen(lambda g: 1.0 if len(g["pool"]) == 5 else 0.0))
    add("попыток основного цикла", per_gen(attempts_of, 2))
    add(
        "заходов за попытку 5 (контекст выкл.)",
        per_gen(lambda g: 1.0 if attempts_of(g) > 5 else 0.0),
    )
    add("проходок цепи всего", per_gen(lambda g: len(g["attempts"]), 2))
    add("попыток без текста", per_gen(lambda g: g["failed"], 2))
    add("дубликатов", per_gen(lambda g: g["duplicates"], 2))
    add("реджектов гейтами", per_gen(lambda g: len(g["rejected"])))
    add("verbatim-дописок", per_gen(lambda g: g["extended"]))
    add("мутаций предложено", per_gen(lambda g: g["mutation_calls"]))
    add("мутаций принято", per_gen(lambda g: g["mutations_accepted"]))
    add("якорь отложен", per_gen(lambda g: g["anchor_deferred"]))
    add("якорь вклеен", per_gen(lambda g: g["anchor_spliced"]))
    add("латентность мс", per_gen(lambda g: g["latency_ms"], 0))
    for name, values in rows:
        print(f"{name:<34}" + "".join(f"{v:>14}" for v in values))

    print("\n== B. Реджекты цепи (на попытку) ==")
    kinds: set[str] = set()
    per_run: list[Counter[str]] = []
    for r in runs:
        counter: Counter[str] = Counter()
        total = 0
        for g in r["generations"]:
            for a in g["attempts"]:
                total += 1
                if a["rejection"]:
                    counter[a["rejection"]] += 1
        counter["__total__"] = total
        per_run.append(counter)
        kinds |= {k for k in counter if k != "__total__"}
    print(f"{'причина':<34}" + "".join(f"{r['arm']:>14}" for r in runs))
    for kind in sorted(kinds):
        print(
            f"{kind:<34}"
            + "".join(
                f"{c[kind] / max(1, c['__total__']):>14.3f}" for c in per_run
            )
        )

    print("\n== C. start_source принятых кандидатов ==")
    sources = ("global", "context", "context_spliced", "hidden_context", "seed", "?")
    print(f"{'source':<20}" + "".join(f"{r['arm'] + ' доля/win':>22}" for r in runs))
    for source in sources:
        cells = []
        for r in runs:
            pool = [c for g in r["generations"] for c in g["pool"]]
            org = [c for c in pool if c["origin"] == "organic"]
            hits = [c for c in org if c["start_source"] == source]
            share = len(hits) / len(org) if org else 0.0
            win = _mean([1.0 if c["won"] else 0.0 for c in hits])
            cells.append(f"{share:.3f}/{win:.3f}")
        print(f"{source:<20}" + "".join(f"{c:>22}" for c in cells))

    print("\n== D. Компоненты скорера: среднее | размах в пуле | доля пулов-решателей ==")
    for r in runs:
        print(f"-- {r['arm']} (seed {r['seed']})")
        pools = [g["pool"] for g in r["generations"] if len(g["pool"]) > 1]
        spreads: dict[str, list[float]] = defaultdict(list)
        decider: Counter[str] = Counter()
        means: dict[str, list[float]] = defaultdict(list)
        for pool in pools:
            best_name, best_spread = None, -1.0
            for name in SCORE_FIELDS:
                values = [c[name] for c in pool]
                spread = max(values) - min(values)
                spreads[name].append(spread)
                means[name].extend(values)
                if spread > best_spread:
                    best_name, best_spread = name, spread
            if best_name is not None and best_spread > 0.0:
                decider[best_name] += 1
        for name in SCORE_FIELDS:
            print(
                f"   {name:<22}{_mean(means[name]):>9.3f}"
                f"{_mean(spreads[name]):>9.3f}"
                f"{decider[name] / max(1, len(pools)):>9.3f}"
            )

    print("\n== E. Происхождение кандидатов ==")
    for r in runs:
        pool = [c for g in r["generations"] for c in g["pool"]]
        if not pool:
            continue
        print(f"-- {r['arm']}")
        for origin in ("organic", "mutated", "seeded"):
            group = [c for c in pool if c["origin"] == origin]
            if not group:
                continue
            print(
                f"   {origin:<10} n={len(group):<6} доля={len(group) / len(pool):.3f} "
                f"total={_mean([c['total'] for c in group]):+.3f} "
                f"win={_mean([1.0 if c['won'] else 0.0 for c in group]):.3f} "
                f"completion={_mean([c['completion_quality'] for c in group]):+.3f} "
                f"verbatim={_mean([c['verbatim_penalty'] for c in group]):.3f}"
            )

    print("\n== F. Отбор ==")
    for r in runs:
        pools = [g["pool"] for g in r["generations"] if g["pool"]]
        eligible = []
        top_wins = []
        for pool in pools:
            best = max(c["total"] for c in pool)
            eligible.append(sum(1 for c in pool if c["total"] >= best - 0.3))
            winner = next((c for c in pool if c["won"]), None)
            if winner is not None:
                top_wins.append(1.0 if winner["total"] == best else 0.0)
        print(
            f"   {r['arm']:<10} кандидатов в окне 0.3: {_mean(eligible):.2f}   "
            f"побеждает лучший: {_mean(top_wins):.3f}"
        )

    print("\n== G. Концовки победителей ==")
    for r in runs:
        winners = [
            c for g in r["generations"] for c in g["pool"] if c["won"]
        ]
        if not winners:
            continue
        dot = sum(1 for c in winners if c["text"].rstrip().endswith((".", "!", "?")))
        print(
            f"   {r['arm']:<10} терминальная пунктуация: {dot / len(winners):.3f} "
            f"(из них '.' срежет flavor в 25%: {0.25 * dot / len(winners):.3f} ответов)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--db", default="db_prod_copy/markov.db")
    run_p.add_argument("--chat-id", type=int, default=None)
    run_p.add_argument("--arm", choices=sorted(ARMS), required=True)
    run_p.add_argument("--generations", type=int, default=300)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--target", type=int, default=0)
    run_p.add_argument("--moment", type=int, default=None)
    run_p.add_argument("--set", action="append", default=[])
    run_p.add_argument("--out", required=True)

    report_p = sub.add_parser("report")
    report_p.add_argument("paths", nargs="+", type=Path)

    args = parser.parse_args()
    if args.cmd == "report":
        report(args.paths)
        return
    payload = asyncio.run(run_arm(args))
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"wrote {args.out}: arm={args.arm} "
        f"generations={len(payload['generations'])} "
        f"budget={GENERATION_ATTEMPT_BUDGET}"
    )


if __name__ == "__main__":
    main()
