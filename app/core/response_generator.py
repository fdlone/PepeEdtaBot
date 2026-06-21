from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from app.config.runtime_state import RuntimeState
from app.core.candidate_scorer import CandidateScore, score_candidate
from app.core.markov import (
    MarkovGenerator,
    escalated_randomness_strength,
    is_short_generated_reply,
    tokenize,
)
from app.core.text import capitalize_reply_sentences, sanitize_text
from app.log_masking import mask_chat_id

logger = logging.getLogger("chat_markov")

GENERATION_ATTEMPT_BUDGET = 10
GENERATION_ATTEMPTS_WITH_CONTEXT = 5
CANDIDATE_TARGET = 5

CandidateScorer = Callable[[str, list[str], list[str]], CandidateScore]


class VerbatimCopyChecker(Protocol):
    async def is_verbatim_copy(self, chat_id: int, text: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    chat_id: int
    context_tokens: list[str]
    seed: list[str] | None
    current_message_normalized: str


@dataclass(frozen=True, slots=True)
class ResponseGenerationResult:
    text: str | None
    candidates_scored: int


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    text: str
    score: CandidateScore


def was_recent_short_reply(
    runtime_state: RuntimeState,
    chat_id: int,
    candidate: str,
) -> bool:
    recent = runtime_state.recent_short_replies.get(chat_id)
    if not recent:
        return False
    return sanitize_text(candidate).lower() in recent


@dataclass(slots=True)
class ResponseGenerator:
    generator: MarkovGenerator
    learning_service: VerbatimCopyChecker
    runtime_state: RuntimeState
    scorer: CandidateScorer = score_candidate

    async def generate(
        self,
        request: GenerationRequest,
        rng: random.Random | None = None,
        candidate_target: int = CANDIDATE_TARGET,
    ) -> str | None:
        result = await self.generate_with_result(
            request,
            rng=rng,
            candidate_target=candidate_target,
        )
        return result.text

    async def generate_with_result(
        self,
        request: GenerationRequest,
        rng: random.Random | None = None,
        candidate_target: int = CANDIDATE_TARGET,
    ) -> ResponseGenerationResult:
        generation_rng = rng or random.Random()
        seed = request.seed
        target = max(1, min(candidate_target, GENERATION_ATTEMPT_BUDGET))
        candidates: list[_ScoredCandidate] = []
        seen_candidates: set[str] = set()

        for attempt in range(GENERATION_ATTEMPT_BUDGET):
            attempt_context_tokens = (
                request.context_tokens
                if attempt < GENERATION_ATTEMPTS_WITH_CONTEXT
                else None
            )
            attempt_randomness_strength = escalated_randomness_strength(
                self.runtime_state.randomness_strength,
                attempt,
                GENERATION_ATTEMPT_BUDGET,
            )
            candidate = await self.generator.generate_text(
                chat_id=request.chat_id,
                max_chars=self.runtime_state.max_reply_chars,
                max_tokens=self.runtime_state.max_reply_tokens,
                seed_tokens=seed,
                context_tokens=attempt_context_tokens,
                context_bias=self.runtime_state.reply_context_bias,
                context_start_bias=self.runtime_state.reply_context_start_bias,
                randomness_strength=attempt_randomness_strength,
                repetition_penalty_strength=self.runtime_state.repetition_penalty_strength,
                markov_order=self.runtime_state.markov_order,
                enable_backoff=self.runtime_state.enable_backoff,
                backoff_min_order=self.runtime_state.backoff_min_order,
                rng=generation_rng,
                attempt_budget=1,
            )
            if not candidate:
                logger.debug(
                    "Generation attempt failed: chat=%s attempt=%s context=%s seed_len=%s",
                    mask_chat_id(request.chat_id),
                    attempt + 1,
                    bool(attempt_context_tokens),
                    len(seed or []),
                )
            else:
                candidate_normalized = sanitize_text(candidate).lower()
                candidate_tokens = tokenize(
                    candidate,
                    normalize_lower=self.runtime_state.normalize_lower,
                )
                is_short_candidate = is_short_generated_reply(candidate_tokens)
                if candidate_normalized == request.current_message_normalized:
                    logger.debug(
                        "Generated text copied the current message, retrying: "
                        "chat=%s attempt=%s",
                        mask_chat_id(request.chat_id),
                        attempt + 1,
                    )
                elif is_short_candidate and was_recent_short_reply(
                    self.runtime_state,
                    request.chat_id,
                    candidate,
                ):
                    logger.debug(
                        "Generated short reply was used recently, retrying: "
                        "chat=%s attempt=%s",
                        mask_chat_id(request.chat_id),
                        attempt + 1,
                    )
                elif (
                    not is_short_candidate
                    and await self.learning_service.is_verbatim_copy(
                        request.chat_id,
                        candidate,
                    )
                ):
                    logger.debug(
                        "Generated text starts like a training sample, retrying: "
                        "chat=%s attempt=%s",
                        mask_chat_id(request.chat_id),
                        attempt + 1,
                    )
                else:
                    logger.debug(
                        "Reply generated: chat=%s attempt=%s tokens=%s context=%s",
                        mask_chat_id(request.chat_id),
                        attempt + 1,
                        len(candidate.split()),
                        bool(attempt_context_tokens),
                    )
                    if candidate not in seen_candidates:
                        seen_candidates.add(candidate)
                        candidates.append(
                            _ScoredCandidate(
                                text=candidate,
                                score=self.scorer(
                                    candidate,
                                    candidate_tokens,
                                    request.context_tokens,
                                ),
                            )
                        )
                        if len(candidates) >= target:
                            break

            seed = None

        if not candidates:
            return ResponseGenerationResult(text=None, candidates_scored=0)
        selected = max(candidates, key=lambda candidate: candidate.score.total)
        text = (
            capitalize_reply_sentences(selected.text)
            if self.runtime_state.auto_capitalize_replies
            else selected.text
        )
        return ResponseGenerationResult(
            text=text,
            candidates_scored=len(candidates),
        )
