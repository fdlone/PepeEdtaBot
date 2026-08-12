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
from statistics import mean, median, pstdev
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import log_masking  # noqa: E402
from app.core.candidate_scorer import meaningful_tokens  # noqa: E402
from app.core.markov import (  # noqa: E402
    PUNCT_SET,
    MarkovGenerator,
    content_tokens,
    tokenize,
)
from app.core.markov_port import MarkovReadPort  # noqa: E402
from app.core.response_generator import (  # noqa: E402
    CANDIDATE_TARGET,
    GENERATION_ATTEMPT_BUDGET,
    GenerationRequest,
    ResponseGenerator,
    remember_recent_reply,
)
from app.infrastructure.database import Database  # noqa: E402

DEFAULT_SEED = 20260620
DEFAULT_GENERATIONS = 100
SYNTHETIC_CHAT_ID = 1
CORPUS_PATH = Path(__file__).with_name("fixtures") / "synthetic_generation_corpus.txt"
CASE_CONTEXT_PATH = (
    Path(__file__).with_name("fixtures") / "synthetic_generation_case_context.txt"
)


class _NoVerbatimCopies:
    async def is_verbatim_copy(self, chat_id: int, text: str) -> bool:
        return False

    async def get_emoji_stats(self, chat_id: int) -> dict[str, int]:
        # Emoji channel is off in eval; the generator never calls this.
        return {}

    async def get_verbatim_ngram_index(
        self, chat_id: int
    ) -> frozenset[tuple[str, ...]]:
        # Verbatim penalty is off in the synthetic eval (strength 0.0); the
        # generator never calls this.
        return frozenset()

    async def get_context_idf(self, chat_id: int) -> dict[str, float]:
        # Empty IDF makes idf_context_relevance fall back to the length-normalized
        # formula, which is what the committed baselines below encode.
        return {}


def load_synthetic_corpus(*, normalize_lower: bool) -> list[str]:
    corpus = [
        line.strip()
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not normalize_lower:
        corpus.extend(
            line.strip()
            for line in CASE_CONTEXT_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return corpus


class _InstrumentedMarkovGenerator(MarkovGenerator):
    def __init__(self, db: MarkovReadPort) -> None:
        super().__init__(db)
        self.leading_punctuation_stripped = 0
        self.context_exact_matches = 0
        self.context_casefold_matches = 0
        self.hidden_context_fallbacks = 0

    async def generate_text_with_trace(
        self, *args: Any, **kwargs: Any
    ) -> tuple[str, Any]:
        text, trace = await super().generate_text_with_trace(*args, **kwargs)
        self.leading_punctuation_stripped += trace.leading_punctuation_stripped
        self.context_exact_matches += trace.context_exact_matches
        self.context_casefold_matches += trace.context_casefold_matches
        self.hidden_context_fallbacks += trace.hidden_context_fallbacks
        return text, trace


def build_ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    return [
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    ]


def distinct_ratio(outputs: list[list[str]], size: int) -> float:
    """Unique ``size``-grams over all ``size``-grams, pooled across outputs.

    NOT comparable across runs of different length. This is a type/token
    ratio: pooling more text can only repeat n-grams already seen, so the
    value falls as ``generations`` grows even when the model is unchanged
    (2026-07-21: the same config scored 0.50 at 200 generations and 0.41 at
    400, which reads as a regression and is not one). Compare distinct_* only
    between arms of the SAME sweep pass at equal generations; ``eval_prod``
    reports ``distinct_basis_tokens`` so the denominator can be checked.
    """
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


def starts_with_context_run(
    output_tokens: list[str],
    context_tokens: list[str],
    min_run: int = 3,
) -> bool:
    """True if the reply's leading content tokens reproduce a contiguous context run.

    Direct indicator of literal context echo at the start of the reply — the
    defect that phase 4.1 increment C (hidden context state) targets.
    """
    context = meaningful_tokens(context_tokens)
    run = min(min_run, len(output_tokens))
    if run < 2 or len(context) < run:
        return False
    prefix = output_tokens[:run]
    return any(
        context[start : start + run] == prefix
        for start in range(len(context) - run + 1)
    )


async def evaluate_generation(
    *,
    seed: int = DEFAULT_SEED,
    generations: int = DEFAULT_GENERATIONS,
    candidate_target: int = CANDIDATE_TARGET,
    normalize_lower: bool = True,
    fuzzy_context_casefold: bool = False,
    recent_reply_penalty_strength: float = 0.5,
    length_mode_weights: tuple[float, float, float] = (0.25, 0.55, 0.2),
) -> dict[str, int | float]:
    if generations <= 0:
        raise ValueError("generations must be positive")
    effective_candidate_target = max(
        1,
        min(candidate_target, GENERATION_ATTEMPT_BUDGET),
    )

    corpus = load_synthetic_corpus(normalize_lower=normalize_lower)
    rng = random.Random(seed)
    outputs: list[list[str]] = []
    context_overlaps: list[float] = []
    context_prefix_copies: list[bool] = []
    candidates_scored: list[int] = []
    latencies_ms: list[float] = []
    leading_punctuation_replies = 0
    log_masking.init_masking("synthetic-generation-evaluation")

    with tempfile.TemporaryDirectory(prefix="pepe_generation_eval_") as temp_dir:
        db = Database(str(Path(temp_dir) / "synthetic.sqlite"))
        await db.init()
        try:
            for message in corpus:
                tokens = tokenize(message, normalize_lower=normalize_lower)
                await db.save_message_and_update_model(
                    chat_id=SYNTHETIC_CHAT_ID,
                    raw_text=message,
                    tokens=tokens,
                )

            generator = _InstrumentedMarkovGenerator(db.markov)
            runtime_state = SimpleNamespace(
                randomness_strength=2.0,
                candidate_selection_temperature=1.3,
                reply_flavor_strength=1.0,
                max_reply_chars=280,
                max_reply_tokens=45,
                reply_context_bias=1.8,
                reply_context_start_bias=2.2,
                # 1.0 keeps the committed baselines byte-identical: the
                # affinity boost is measured on the prod-copy eval instead.
                context_start_affinity=1.0,
                repetition_penalty_strength=1.0,
                recent_reply_penalty_strength=recent_reply_penalty_strength,
                # Off in the synthetic eval: there is no prod corpus index here,
                # the baselines measure the word model alone.
                verbatim_penalty_strength=0.0,
                length_mode_weights=length_mode_weights,
                # 0.0 keeps the committed baselines byte-identical: the
                # synthetic context is a fixed placeholder string, so its length
                # carries no signal. Mirroring is measured on the prod-copy eval.
                intonation_profile_strength=0.0,
                length_context_adaptation=0.0,
                markov_order=3,
                # Neutral Phase 2 knobs keep the committed baselines
                # byte-identical: gain 0 is the 1.x sampler, and a degenerate
                # bound of 0 leaves the candidate target fixed.
                markov_entropy_temp_gain=0.0,
                markov_entropy_pivot=0.5,
                markov_entropy_temp_min=0.5,
                markov_entropy_temp_max=12.0,
                markov_branching_degenerate_max=0.0,
                markov_branching_candidate_floor=2,
                enable_backoff=True,
                # Keep the M3/M4 channels off in eval so the baselines measure the
                # word model alone (and no emoji-stats DB reads are attempted).
                markov_jump_probability=0.0,
                context_jump_boost=1.0,
                verbatim_extension_share=0.0,
                order_mix_probability=0.0,
                slot_mutation_probability=0.0,
                context_anchor_splice_probability=0.0,
                emoji_append_chance=0.0,
                normalize_lower=normalize_lower,
                fuzzy_context_casefold=fuzzy_context_casefold,
                auto_capitalize_replies=False,
                recent_short_replies={},
                recent_replies={},
            )
            response_generator = ResponseGenerator(
                generator=generator,
                learning_service=_NoVerbatimCopies(),
                runtime_state=runtime_state,
            )
            for index in range(generations):
                context_tokens = tokenize(
                    corpus[index % len(corpus)],
                    normalize_lower=normalize_lower,
                )
                if (
                    fuzzy_context_casefold
                    and not normalize_lower
                    and index % 10 == 0
                ):
                    context_tokens = [
                        token.swapcase() if token not in PUNCT_SET else token
                        for token in context_tokens
                    ]
                started_at = time.perf_counter()
                selection = await response_generator.generate_with_result(
                    GenerationRequest(
                        chat_id=SYNTHETIC_CHAT_ID,
                        context_tokens=context_tokens,
                        seed=None,
                        current_message_normalized="__synthetic_evaluation_input__",
                    ),
                    rng=rng,
                    candidate_target=effective_candidate_target,
                )
                latencies_ms.append((time.perf_counter() - started_at) * 1000)
                if selection.text:
                    # Mirror the handler: every sent reply feeds the full-reply
                    # anti-repeat, so the eval exercises the same rolling window.
                    remember_recent_reply(
                        runtime_state,  # type: ignore[arg-type]
                        SYNTHETIC_CHAT_ID,
                        selection.text,
                    )
                output = content_tokens(
                    tokenize(
                        selection.text or "",
                        normalize_lower=normalize_lower,
                    )
                )
                emitted_tokens = tokenize(selection.text or "")
                if emitted_tokens and emitted_tokens[0] in PUNCT_SET:
                    leading_punctuation_replies += 1
                outputs.append(output)
                context_overlaps.append(
                    context_token_overlap(output, context_tokens)
                )
                context_prefix_copies.append(
                    starts_with_context_run(output, context_tokens)
                )
                candidates_scored.append(selection.candidates_scored)
        finally:
            await db.close()

    lengths = [len(output) for output in outputs]
    empty_count = sum(1 for output in outputs if not output)
    context_resolution_attempts = (
        generator.context_exact_matches
        + generator.context_casefold_matches
        + generator.hidden_context_fallbacks
    )
    return {
        "baseline_phase": 4.1,
        "seed": seed,
        "generations": generations,
        "corpus_messages": len(corpus),
        "normalize_lower": normalize_lower,
        "fuzzy_context_casefold": fuzzy_context_casefold,
        "candidate_target": effective_candidate_target,
        "empty_result_rate": empty_count / generations,
        "distinct_1": distinct_ratio(outputs, 1),
        "distinct_2": distinct_ratio(outputs, 2),
        "repeated_bigram_ratio": repeated_ngram_ratio(outputs, 2),
        "repeated_trigram_ratio": repeated_ngram_ratio(outputs, 3),
        "avg_length_tokens": mean(lengths),
        "median_length_tokens": median(lengths),
        "stddev_length_tokens": pstdev(lengths),
        "context_token_overlap": mean(context_overlaps),
        "context_prefix_copy_rate": (
            sum(context_prefix_copies) / generations if generations else 0.0
        ),
        "leading_punctuation_rate": leading_punctuation_replies / generations,
        "leading_punctuation_stripped": generator.leading_punctuation_stripped,
        "context_exact_match_rate": (
            generator.context_exact_matches / context_resolution_attempts
            if context_resolution_attempts
            else 0.0
        ),
        "context_casefold_match_rate": (
            generator.context_casefold_matches / context_resolution_attempts
            if context_resolution_attempts
            else 0.0
        ),
        "hidden_context_fallback_to_global_rate": (
            generator.hidden_context_fallbacks / context_resolution_attempts
            if context_resolution_attempts
            else 0.0
        ),
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
    parser.add_argument(
        "--profile",
        choices=("normalize-lower", "case-preserved"),
        default="normalize-lower",
    )
    parser.add_argument("--fuzzy-context-casefold", action="store_true")
    parser.add_argument(
        "--recent-reply-penalty-strength", type=float, default=0.5
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(
        evaluate_generation(
            seed=args.seed,
            generations=args.generations,
            candidate_target=args.candidate_target,
            normalize_lower=args.profile == "normalize-lower",
            fuzzy_context_casefold=args.fuzzy_context_casefold,
            recent_reply_penalty_strength=args.recent_reply_penalty_strength,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
