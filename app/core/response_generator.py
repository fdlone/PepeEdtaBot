from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from app.config.runtime_state import RuntimeState, get_recent_deque
from app.core import gen_trace_log
from app.core.candidate_scorer import (
    CandidateScore,
    build_recent_reply_trigrams,
    coherence_penalty_for_order,
    idf_context_relevance,
    recent_reply_overlap,
    sample_length_mode,
    score_candidate,
    verbatim_ngram_overlap,
    verbatim_quote_severity,
)
from app.core.emoji import append_emoji_flavor, strip_trailing_emojis
from app.core.markov import (
    JUMP_CONNECTIVE_TOKENS,
    PUNCT_SET,
    MarkovGenerator,
    detokenize,
    escalated_randomness_strength,
    finalize_reply_ending,
    is_short_generated_reply,
    pick_jump_connective,
    tokenize,
    trim_splice_tail,
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
# 0.3 (2026-07-09): this margin, not the temperature, is what decides whether
# the best candidate wins -- eval_prod moved context_win_given_top 0.55 -> 0.89
# on the margin alone, while temperatures 1.3/0.7/0.3 stayed within 1pp of each
# other. Widening it hurts: at 1.0 the synthetic eval's context_token_overlap
# collapsed from 0.38 to 0.13.
SELECTION_SCORE_MARGIN = 0.3
# How many recently sent replies are remembered per chat for the full-reply
# anti-repeat: exact re-sends are rejected, partial (trigram) overlap is
# penalized before softmax selection.
RECENT_REPLY_LIMIT = 20
# In "short" length mode the Markov walk itself is capped so candidates
# actually come out short instead of only being re-ranked by the scorer.
# 8 raw tokens chosen by eval sweep: cap 6 starved generation on the
# synthetic corpus (11% empty results), 8 shifts lengths with ~1% empties.
SHORT_MODE_MAX_TOKENS = 8
# A candidate that IS a training message 1:1 is not discarded: it gets a
# connective + a fresh short walk appended, so the reply is "corpus base +
# отсебятина" instead of a bare quote. This is the walk budget of the tail.
VERBATIM_COPY_REASON = "Generated text is a training sample 1:1"
EXTENSION_MAX_TOKENS = 10

CandidateScorer = Callable[[str, list[str], list[str], str], CandidateScore]


class VerbatimCopyChecker(Protocol):
    async def is_verbatim_copy(self, chat_id: int, text: str) -> bool: ...
    async def get_emoji_stats(self, chat_id: int) -> Mapping[str, int]: ...
    async def get_verbatim_ngram_index(
        self, chat_id: int
    ) -> frozenset[tuple[str, ...]]: ...
    async def get_context_idf(self, chat_id: int) -> Mapping[str, float]: ...


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
    recent = get_recent_deque(
        runtime_state.recent_replies, chat_id, RECENT_REPLY_LIMIT
    )
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

    async def _candidate_reject_reason(
        self,
        request: GenerationRequest,
        candidate: str,
        candidate_normalized: str,
        is_short_candidate: bool,
    ) -> str | None:
        """Reason to discard a generated candidate, or None if it is usable.

        Ordered gates: echo of the current message, short-reply anti-repeat,
        full-reply anti-repeat, verbatim training-sample copy (long replies
        only — short ones are governed by the short-reply anti-repeat).
        """
        if candidate_normalized == request.current_message_normalized:
            return "Generated text copied the current message"
        if is_short_candidate and was_recent_short_reply(
            self.runtime_state, request.chat_id, candidate
        ):
            return "Generated short reply was used recently"
        if was_recent_reply(self.runtime_state, request.chat_id, candidate):
            return "Generated reply repeats a recently sent one"
        if not is_short_candidate and await self.learning_service.is_verbatim_copy(
            request.chat_id, candidate
        ):
            return VERBATIM_COPY_REASON
        return None

    async def _extend_verbatim_candidate(
        self,
        request: GenerationRequest,
        candidate: str,
        rng: random.Random,
    ) -> str | None:
        """Append a connective + fresh short walk to a 1:1 training-sample copy.

        The tail starts from a global (context-affine) start — never from a
        contextual anchor, which would just keep continuing the quote. Returns
        None when no usable continuation could be generated.
        """
        state = self.runtime_state
        continuation, _ = await self.generator.generate_text_with_trace(
            chat_id=request.chat_id,
            max_chars=state.max_reply_chars,
            max_tokens=EXTENSION_MAX_TOKENS,
            context_tokens=request.context_tokens,
            context_bias=state.reply_context_bias,
            context_start_bias=1.0,
            context_start_affinity=state.context_start_affinity,
            randomness_strength=state.randomness_strength,
            repetition_penalty_strength=state.repetition_penalty_strength,
            markov_order=state.markov_order,
            enable_backoff=state.enable_backoff,
            jump_probability=0.0,
            rng=rng,
            attempt_budget=2,
        )
        if not continuation:
            return None
        base_tokens = tokenize(candidate, normalize_lower=state.normalize_lower)
        while base_tokens and base_tokens[-1] in PUNCT_SET:
            base_tokens.pop()
        trim_splice_tail(base_tokens)
        if not base_tokens:
            return None
        tail_tokens = tokenize(continuation, normalize_lower=state.normalize_lower)
        # Same stutter guard as the M4 splice: never pick a connective that
        # contains the continuation's first word (", ну и" + "ну как..." read
        # as "ну и ну" in live samples).
        connective = pick_jump_connective(
            rng,
            exclude=[
                phrase
                for phrase in JUMP_CONNECTIVE_TOKENS
                if tail_tokens and tail_tokens[0].casefold() in phrase
            ],
        )
        combined = finalize_reply_ending(
            base_tokens + list(connective) + tail_tokens
        )
        extended = detokenize(combined, max_chars=state.max_reply_chars)
        if not extended or extended == candidate:
            return None
        return extended

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
        # Topic overlap is scored by how much of the context's *informative* mass
        # the candidate echoes, not by raw token count -- otherwise a short reply
        # sharing a pronoun beats a long one sharing a proper noun.
        context_idf: Mapping[str, float] = {}
        if request.context_tokens:
            context_idf = await self.learning_service.get_context_idf(
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

        gen_trace_log.log_request_header(
            request.chat_id,
            length_mode=length_mode,
            randomness=effective_randomness,
            temperature=self.runtime_state.candidate_selection_temperature,
            target=target,
            budget=GENERATION_ATTEMPT_BUDGET,
            attempts_with_context=GENERATION_ATTEMPTS_WITH_CONTEXT,
            recent_penalty_strength=recent_penalty_strength,
            verbatim_penalty_strength=verbatim_penalty_strength,
        )

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
                context_start_affinity=self.runtime_state.context_start_affinity,
                randomness_strength=attempt_randomness_strength,
                repetition_penalty_strength=self.runtime_state.repetition_penalty_strength,
                markov_order=self.runtime_state.markov_order,
                enable_backoff=self.runtime_state.enable_backoff,
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
                gen_trace_log.log_attempt_failed(
                    request.chat_id,
                    attempt + 1,
                    context_used=bool(attempt_context_tokens),
                )
            else:
                candidate_normalized = sanitize_text(candidate).lower()
                candidate_tokens = tokenize(
                    candidate,
                    normalize_lower=self.runtime_state.normalize_lower,
                )
                is_short_candidate = is_short_generated_reply(candidate_tokens)
                reject_reason = await self._candidate_reject_reason(
                    request,
                    candidate,
                    candidate_normalized,
                    is_short_candidate,
                )
                if reject_reason == VERBATIM_COPY_REASON:
                    # Corpus replies are welcome — bare quotes are not: extend
                    # the copy with отсебятина and re-run the gates on the
                    # combined text instead of discarding the attempt.
                    extended = await self._extend_verbatim_candidate(
                        request, candidate, generation_rng
                    )
                    if extended is not None:
                        gen_trace_log.log_attempt_extended(
                            request.chat_id,
                            attempt + 1,
                            original=candidate,
                            extended=extended,
                        )
                        candidate = extended
                        candidate_normalized = sanitize_text(candidate).lower()
                        candidate_tokens = tokenize(
                            candidate,
                            normalize_lower=self.runtime_state.normalize_lower,
                        )
                        is_short_candidate = is_short_generated_reply(
                            candidate_tokens
                        )
                        reject_reason = await self._candidate_reject_reason(
                            request,
                            candidate,
                            candidate_normalized,
                            is_short_candidate,
                        )
                if reject_reason is not None:
                    logger.debug(
                        "%s, retrying: chat=%s attempt=%s",
                        reject_reason,
                        mask_chat_id(request.chat_id),
                        attempt + 1,
                    )
                    gen_trace_log.log_attempt_rejected(
                        request.chat_id,
                        attempt + 1,
                        context_used=bool(attempt_context_tokens),
                        reason=reject_reason,
                        text=candidate,
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
                        coherence_penalty = coherence_penalty_for_order(
                            candidate_trace.markov_order_used
                        )
                        score = replace(
                            self.scorer(
                                candidate,
                                candidate_tokens,
                                request.context_tokens,
                                length_mode,
                            ),
                            context_relevance=idf_context_relevance(
                                candidate_tokens,
                                request.context_tokens,
                                context_idf,
                            ),
                            recent_penalty=recent_penalty_strength
                            * recent_reply_overlap(candidate_tokens, recent_trigrams),
                            verbatim_penalty=verbatim_penalty_strength
                            * verbatim_quote_severity(
                                verbatim_ngram_overlap(
                                    candidate_tokens, corpus_ngrams
                                )
                            ),
                            coherence_penalty=coherence_penalty,
                        )
                        candidates.append(
                            _ScoredCandidate(text=candidate, score=score)
                        )
                        gen_trace_log.log_attempt_accepted(
                            request.chat_id,
                            attempt + 1,
                            context_used=bool(attempt_context_tokens),
                            index=len(candidates),
                            trace=candidate_trace,
                            score=score,
                            text=candidate,
                        )
                        if len(candidates) >= target:
                            break
                    else:
                        gen_trace_log.log_attempt_duplicate(
                            request.chat_id,
                            attempt + 1,
                            context_used=bool(attempt_context_tokens),
                        )

            seed = None

        if not candidates:
            gen_trace_log.log_selection(
                request.chat_id,
                candidates,
                temperature=self.runtime_state.candidate_selection_temperature,
                margin=SELECTION_SCORE_MARGIN,
                selection_margin_used=True,
            )
            return ResponseGenerationResult(text=None, candidates_scored=0)
        selected = select_scored_candidate(
            candidates,
            self.runtime_state.candidate_selection_temperature,
            generation_rng,
        )
        gen_trace_log.log_selection(
            request.chat_id,
            candidates,
            temperature=self.runtime_state.candidate_selection_temperature,
            margin=SELECTION_SCORE_MARGIN,
            selection_margin_used=True,
            selected=selected,
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
