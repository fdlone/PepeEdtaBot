"""Synthetic selection-path evaluation for the post-Phase-3.2 baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import log_masking  # noqa: E402
from app.core.candidate_scorer import meaningful_tokens  # noqa: E402
from app.core.markov import (  # noqa: E402
    MarkovGenerator,
    content_tokens,
    extract_best_seed,
    tokenize,
)
from app.core.response_generator import (  # noqa: E402
    CANDIDATE_TARGET,
    GENERATION_ATTEMPT_BUDGET,
    GenerationRequest,
    ResponseGenerator,
)
from app.infrastructure.database import Database  # noqa: E402

DEFAULT_SEED = 20260620
DEFAULT_GENERATIONS = 100
SYNTHETIC_CHAT_ID = 1
CORPUS_PATH = Path(__file__).with_name("fixtures") / "synthetic_generation_corpus.txt"


class _NoVerbatimCopies:
    async def is_verbatim_copy(self, chat_id: int, text: str) -> bool:
        return False


def load_synthetic_corpus() -> list[str]:
    return [
        line.strip()
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    return [
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    ]


def distinct_ratio(outputs: list[list[str]], size: int) -> float:
    ngrams = [
        ngram
        for output in outputs
        for ngram in build_ngrams(output, size)
    ]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def repeated_ngram_ratio(outputs: list[list[str]], size: int) -> float:
    repeated = 0
    total = 0
    for tokens in outputs:
        counts = Counter(build_ngrams(tokens, size))
        total += sum(counts.values())
        repeated += sum(count - 1 for count in counts.values() if count > 1)
    return repeated / total if total else 0.0


def context_token_overlap(
    output_tokens: list[str],
    context_tokens: list[str],
) -> float:
    output = set(meaningful_tokens(output_tokens))
    context = set(meaningful_tokens(context_tokens))
    if not output:
        return 0.0
    return len(output & context) / len(output)


async def evaluate_generation(
    *,
    seed: int = DEFAULT_SEED,
    generations: int = DEFAULT_GENERATIONS,
    candidate_target: int = CANDIDATE_TARGET,
) -> dict[str, int | float]:
    if generations <= 0:
        raise ValueError("generations must be positive")
    effective_candidate_target = max(
        1,
        min(candidate_target, GENERATION_ATTEMPT_BUDGET),
    )

    corpus = load_synthetic_corpus()
    rng = random.Random(seed)
    outputs: list[list[str]] = []
    context_overlaps: list[float] = []
    candidates_scored: list[int] = []
    latencies_ms: list[float] = []
    log_masking.init_masking("synthetic-generation-evaluation")

    with tempfile.TemporaryDirectory(prefix="pepe_generation_eval_") as temp_dir:
        db = Database(str(Path(temp_dir) / "synthetic.sqlite"))
        await db.init()
        try:
            for message in corpus:
                tokens = tokenize(message, normalize_lower=True)
                await db.save_message_and_update_model(
                    chat_id=SYNTHETIC_CHAT_ID,
                    raw_text=message,
                    tokens=tokens,
                )

            generator = MarkovGenerator(db)
            runtime_state = SimpleNamespace(
                randomness_strength=2.0,
                max_reply_chars=280,
                max_reply_tokens=45,
                reply_context_bias=1.8,
                reply_context_start_bias=2.2,
                repetition_penalty_strength=1.0,
                markov_order=3,
                enable_backoff=True,
                backoff_min_order=1,
                normalize_lower=True,
                recent_short_replies={},
            )
            response_generator = ResponseGenerator(
                generator=generator,
                learning_service=_NoVerbatimCopies(),
                runtime_state=runtime_state,
            )
            for index in range(generations):
                context_tokens = tokenize(
                    corpus[index % len(corpus)],
                    normalize_lower=True,
                )
                started_at = time.perf_counter()
                selection = await response_generator.generate_with_result(
                    GenerationRequest(
                        chat_id=SYNTHETIC_CHAT_ID,
                        context_tokens=context_tokens,
                        seed=extract_best_seed(context_tokens, 3),
                        current_message_normalized="__synthetic_evaluation_input__",
                    ),
                    rng=rng,
                    candidate_target=effective_candidate_target,
                )
                latencies_ms.append((time.perf_counter() - started_at) * 1000)
                output = content_tokens(
                    tokenize(selection.text or "", normalize_lower=True)
                )
                outputs.append(output)
                context_overlaps.append(
                    context_token_overlap(output, context_tokens)
                )
                candidates_scored.append(selection.candidates_scored)
        finally:
            await db.close()

    lengths = [len(output) for output in outputs]
    empty_count = sum(1 for output in outputs if not output)
    return {
        "baseline_phase": 3.2,
        "seed": seed,
        "generations": generations,
        "corpus_messages": len(corpus),
        "candidate_target": effective_candidate_target,
        "empty_result_rate": empty_count / generations,
        "distinct_1": distinct_ratio(outputs, 1),
        "distinct_2": distinct_ratio(outputs, 2),
        "repeated_bigram_ratio": repeated_ngram_ratio(outputs, 2),
        "repeated_trigram_ratio": repeated_ngram_ratio(outputs, 3),
        "avg_length_tokens": mean(lengths),
        "median_length_tokens": median(lengths),
        "context_token_overlap": mean(context_overlaps),
        "avg_candidates_scored": mean(candidates_scored),
        "avg_generation_latency_ms": mean(latencies_ms),
        "median_generation_latency_ms": median(latencies_ms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Markov generation on a synthetic corpus."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--candidate-target", type=int, default=CANDIDATE_TARGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(
        evaluate_generation(
            seed=args.seed,
            generations=args.generations,
            candidate_target=args.candidate_target,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
