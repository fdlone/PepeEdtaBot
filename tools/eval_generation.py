"""Synthetic generation evaluation for the post-Phase-2 baseline."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.markov import MarkovGenerator, content_tokens, tokenize  # noqa: E402
from app.infrastructure.database import Database  # noqa: E402

DEFAULT_SEED = 20260620
DEFAULT_GENERATIONS = 100
SYNTHETIC_CHAT_ID = 1
CORPUS_PATH = Path(__file__).with_name("fixtures") / "synthetic_generation_corpus.txt"


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


async def evaluate_generation(
    *,
    seed: int = DEFAULT_SEED,
    generations: int = DEFAULT_GENERATIONS,
) -> dict[str, int | float]:
    if generations <= 0:
        raise ValueError("generations must be positive")

    corpus = load_synthetic_corpus()
    rng = random.Random(seed)
    outputs: list[list[str]] = []
    attempts: list[int] = []
    jump_counts: list[int] = []
    latencies_ms: list[float] = []

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
            for _ in range(generations):
                started_at = time.perf_counter()
                text, trace = await generator.generate_text_with_trace(
                    chat_id=SYNTHETIC_CHAT_ID,
                    max_chars=280,
                    max_tokens=45,
                    randomness_strength=2.0,
                    repetition_penalty_strength=1.0,
                    markov_order=3,
                    enable_backoff=True,
                    backoff_min_order=1,
                    rng=rng,
                )
                latencies_ms.append((time.perf_counter() - started_at) * 1000)
                outputs.append(content_tokens(tokenize(text, normalize_lower=True)))
                attempts.append(trace.attempts_used)
                jump_counts.append(trace.jump_count)
        finally:
            await db.close()

    lengths = [len(output) for output in outputs]
    empty_count = sum(1 for output in outputs if not output)
    return {
        "baseline_phase": 2,
        "seed": seed,
        "generations": generations,
        "corpus_messages": len(corpus),
        "empty_result_rate": empty_count / generations,
        "distinct_1": distinct_ratio(outputs, 1),
        "distinct_2": distinct_ratio(outputs, 2),
        "repeated_bigram_ratio": repeated_ngram_ratio(outputs, 2),
        "repeated_trigram_ratio": repeated_ngram_ratio(outputs, 3),
        "avg_length_tokens": mean(lengths),
        "median_length_tokens": median(lengths),
        "avg_attempts": mean(attempts),
        "avg_jump_count": mean(jump_counts),
        "avg_generation_latency_ms": mean(latencies_ms),
        "median_generation_latency_ms": median(latencies_ms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Markov generation on a synthetic corpus."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(
        evaluate_generation(seed=args.seed, generations=args.generations)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
