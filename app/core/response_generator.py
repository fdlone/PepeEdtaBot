from __future__ import annotations

import logging
import math
import random
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from app.config.runtime_state import RuntimeState
from app.core.candidate_scorer import (
    CandidateScore,
    build_recent_reply_trigrams,
    coherence_penalty_for_order,
    recent_reply_overlap,
    sample_length_mode,
    score_candidate,
    verbatim_ngram_overlap,
)
from app.core.emoji import append_emoji_flavor, strip_trailing_emojis
from app.core.markov import (
    MarkovGenerator,
    escalated_randomness_strength,
    is_short_generated_reply,
    tokenize,
)
from app.core.mood import HEATED, NEUTRAL_MODIFIERS, MoodModifiers
from app.core.reply_flavor import apply_reply_flavor
from app.core.text import capitalize_reply_sentences, sanitize_text
from app.log_masking import mask_chat_id

logger = logging.getLogger("chat_markov")

GENERATION_ATTEMPT_BUDGET = 10
GENERATION_ATTEMPTS_WITH_CONTEXT = 5
CANDIDATE_TARGET = 5
# Softmax candidate selection only considers candidates whose score is within
# this margin of the best one; clearly weaker candidates never win the roll.
# 0.5 keeps context relevance influential (its score weight caps at 0.8) while
# still letting near-best candidates win: with margin 1.0 the synthetic eval's
# context_token_overlap collapsed from 0.38 to 0.13.
SELECTION_SCORE_MARGIN = 0.5
# How many recently sent replies are remembered per chat for the full-reply
# anti-repeat: exact re-sends are rejected, partial (trigram) overlap is
# penalized before softmax selection.
RECENT_REPLY_LIMIT = 20
# In "short" length mode the Markov walk itself is capped so candidates
# actually come out short instead of only being re-ranked by the scorer.
# 8 raw tokens chosen by eval sweep: cap 6 starved generation on the
# synthetic corpus (11% empty results), 8 shifts lengths with ~1% empties.
SHORT_MODE_MAX_TOKENS = 8

CandidateScorer = Callable[[str, list[str], list[str], str], CandidateScore]


class VerbatimCopyChecker(Protocol):
    async def is_verbatim_copy(self, chat_id: int, text: str) -> bool: ...
    async def get_emoji_stats(self, chat_id: int) -> Mapping[str, int]: ...
    async def get_verbatim_ngram_index(
        self, chat_id: int
    ) -> frozenset[tuple[str, ...]]: ...


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


def select_scored_candidate(
    candidates: list[_ScoredCandidate],
    temperature: float,
    rng: random.Random,
) -> _ScoredCandidate:
    """Pick a candidate via softmax over scores; temperature 0 means argmax.

    Sampling within SELECTION_SCORE_MARGIN of the best score keeps the
    long-tail (odd but valid) candidates reachable instead of always
    collapsing onto the single "safest" reply.
    """
    best = max(candidates, key=lambda candidate: candidate.score.total)
    if temperature <= 0.0 or len(candidates) == 1:
        return best
    eligible = [
        candidate
        for candidate in candidates
        if candidate.score.total >= best.score.total - SELECTION_SCORE_MARGIN
    ]
    if len(eligible) == 1:
        return eligible[0]
    weights = [
        math.exp((candidate.score.total - best.score.total) / temperature)
        for candidate in eligible
    ]
    return rng.choices(population=eligible, weights=weights, k=1)[0]


def was_recent_short_reply(
    runtime_state: RuntimeState,
    chat_id: int,
    candidate: str,
) -> bool:
    recent = runtime_state.recent_short_replies.get(chat_id)
    if not recent:
        return False
    return sanitize_text(candidate).lower() in recent


def normalize_reply_for_repeat(text: str) -> str:
    """Normalize a reply for full-text anti-repeat comparison.

    The trailing emoji flavor and punctuation cluster are stripped so a
    pre-flavor candidate matches its post-flavor sent form (the flavor pass only
    rewrites the ending punctuation and may append a sampled emoji, M3).
    """
    return strip_trailing_emojis(sanitize_text(text).lower()).rstrip(" .!?…")


def remember_recent_reply(
    runtime_state: RuntimeState,
    chat_id: int,
    reply_text: str,
) -> None:
    normalized = normalize_reply_for_repeat(reply_text)
    if not normalized:
        return
    recent = runtime_state.recent_replies.get(chat_id)
    if recent is None:
        recent = deque(maxlen=RECENT_REPLY_LIMIT)
        runtime_state.recent_replies[chat_id] = recent
    recent.append(normalized)


def was_recent_reply(
    runtime_state: RuntimeState,
    chat_id: int,
    candidate: str,
) -> bool:
    recent = runtime_state.recent_replies.get(chat_id)
    if not recent:
        return False
    return normalize_reply_for_repeat(candidate) in recent


@dataclass(slots=True)
class ResponseGenerator:
    generator: MarkovGenerator
    learning_service: VerbatimCopyChecker
    runtime_state: RuntimeState
    scorer: CandidateScorer = score_candidate
    # Per-chat mood modulation; None (or the neutral instance) leaves the
    # configured runtime knobs untouched.
    mood_modifiers: MoodModifiers | None = None
    # Per-chat mood label (M3): only used to boost the emoji-append chance when
    # the chat is heated. None leaves the base chance unchanged.
    mood: str | None = None

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
        modifiers = self.mood_modifiers or NEUTRAL_MODIFIERS
        seed = request.seed
        target = max(1, min(candidate_target, GENERATION_ATTEMPT_BUDGET))
        candidates: list[_ScoredCandidate] = []
        seen_candidates: set[str] = set()
        recent_penalty_strength = self.runtime_state.recent_reply_penalty_strength
        recent_trigrams = (
            build_recent_reply_trigrams(
                self.runtime_state.recent_replies.get(request.chat_id) or ()
            )
            if recent_penalty_strength > 0.0
            else set()
        )
        # Quote detection: candidates whose content 4-grams all exist in the
        # corpus are verbatim replays and lose score to recombined candidates.
        # The index read is skipped entirely when the penalty is off.
        verbatim_penalty_strength = self.runtime_state.verbatim_penalty_strength
        corpus_ngrams: frozenset[tuple[str, ...]] = frozenset()
        if verbatim_penalty_strength > 0.0:
            corpus_ngrams = await self.learning_service.get_verbatim_ngram_index(
                request.chat_id
            )
        base_weights = self.runtime_state.length_mode_weights
        mood_weights = (
            base_weights[0] * modifiers.length_weight_mult[0],
            base_weights[1] * modifiers.length_weight_mult[1],
            base_weights[2] * modifiers.length_weight_mult[2],
        )
        length_mode = sample_length_mode(mood_weights, generation_rng)
        effective_randomness = max(
            0.0, self.runtime_state.randomness_strength + modifiers.randomness_delta
        )
        max_tokens = self.runtime_state.max_reply_tokens
        if length_mode == "short":
            max_tokens = min(max_tokens, SHORT_MODE_MAX_TOKENS)

        for attempt in range(GENERATION_ATTEMPT_BUDGET):
            attempt_context_tokens = (
                request.context_tokens
                if attempt < GENERATION_ATTEMPTS_WITH_CONTEXT
                else None
            )
            attempt_randomness_strength = escalated_randomness_strength(
                effective_randomness,
                attempt,
                GENERATION_ATTEMPT_BUDGET,
            )
            candidate, candidate_trace = await self.generator.generate_text_with_trace(
                chat_id=request.chat_id,
                max_chars=self.runtime_state.max_reply_chars,
                max_tokens=max_tokens,
                seed_tokens=seed,
                context_tokens=attempt_context_tokens,
                context_bias=self.runtime_state.reply_context_bias,
                context_start_bias=self.runtime_state.reply_context_start_bias,
                randomness_strength=attempt_randomness_strength,
                repetition_penalty_strength=self.runtime_state.repetition_penalty_strength,
                markov_order=self.runtime_state.markov_order,
                enable_backoff=self.runtime_state.enable_backoff,
                backoff_min_order=self.runtime_state.backoff_min_order,
                fuzzy_context_casefold=self.runtime_state.fuzzy_context_casefold,
                fuzzy_context_prefix=self.runtime_state.fuzzy_context_prefix,
                context_emit_start=self.runtime_state.reply_context_emit_start,
                jump_probability=self.runtime_state.markov_jump_probability,
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
                elif was_recent_reply(
                    self.runtime_state,
                    request.chat_id,
                    candidate,
                ):
                    logger.debug(
                        "Generated reply repeats a recently sent one, retrying: "
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
                        score = self.scorer(
                            candidate,
                            candidate_tokens,
                            request.context_tokens,
                            length_mode,
                        )
                        if recent_trigrams:
                            score = replace(
                                score,
                                recent_penalty=recent_penalty_strength
                                * recent_reply_overlap(
                                    candidate_tokens, recent_trigrams
                                ),
                            )
                        if corpus_ngrams:
                            score = replace(
                                score,
                                verbatim_penalty=verbatim_penalty_strength
                                * verbatim_ngram_overlap(
                                    candidate_tokens, corpus_ngrams
                                ),
                            )
                        score = replace(
                            score,
                            coherence_penalty=coherence_penalty_for_order(
                                candidate_trace.markov_order_used
                            ),
                        )
                        candidates.append(
                            _ScoredCandidate(text=candidate, score=score)
                        )
                        if len(candidates) >= target:
                            break

            seed = None

        if not candidates:
            return ResponseGenerationResult(text=None, candidates_scored=0)
        selected = select_scored_candidate(
            candidates,
            self.runtime_state.candidate_selection_temperature,
            generation_rng,
        )
        text = (
            capitalize_reply_sentences(selected.text)
            if self.runtime_state.auto_capitalize_replies
            else selected.text
        )
        text = apply_reply_flavor(
            text,
            generation_rng,
            self.runtime_state.reply_flavor_strength * modifiers.flavor_strength_mult,
        )
        # M3 emoji channel: occasionally end on an emoji this chat actually uses.
        # The stats read is skipped entirely when the channel is off so the common
        # path adds no query.
        emoji_chance = self.runtime_state.emoji_append_chance
        if emoji_chance > 0.0:
            emoji_stats = await self.learning_service.get_emoji_stats(request.chat_id)
            if emoji_stats:
                text = append_emoji_flavor(
                    text,
                    emoji_stats,
                    generation_rng,
                    chance=emoji_chance,
                    heated=self.mood == HEATED,
                )
        return ResponseGenerationResult(
            text=text,
            candidates_scored=len(candidates),
        )
