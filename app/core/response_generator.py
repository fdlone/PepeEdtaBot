from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from typing import Protocol

from app.config.runtime_state import RuntimeState
from app.core import gen_trace_log
from app.core.candidate_scorer import (
    CandidateScore,
    build_recent_reply_trigrams,
    context_length_weights,
    idf_context_relevance,
    recent_reply_overlap,
    sample_length_mode,
    score_candidate,
    verbatim_ngram_overlap,
    verbatim_quote_severity,
)
from app.core.collocations import collocation_effect
from app.core.emoji import append_emoji_flavor, strip_trailing_emojis
from app.core.generation_telemetry import CandidateRoute
from app.core.interpolation import OrderInterpolation
from app.core.intonation import IntonationProfile, blend_length_weights
from app.core.markov import (
    JUMP_CONNECTIVE_TOKENS,
    PUNCT_SET,
    EntropySampling,
    MarkovGenerator,
    content_tokens,
    detokenize,
    escalated_randomness_strength,
    finalize_candidate_tokens,
    finalize_reply_ending,
    is_short_generated_reply,
    pick_splice_connective,
    splice_marker_tokens,
    tokenize,
    trim_splice_tail,
)
from app.core.mood import (
    CALM,
    HEATED,
    LIVELY,
    NEUTRAL_MODIFIERS,
    SLEEPY,
    MoodModifiers,
)
from app.core.reply_flavor import apply_reply_flavor
from app.core.shadow_order import shadow_order4_stats
from app.core.slot_mutation import mutate_candidate_tokens
from app.core.temporal import TemporalBlend
from app.core.text import capitalize_reply_sentences, sanitize_text
from app.log_masking import mask_chat_id

logger = logging.getLogger("chat_markov")

GENERATION_ATTEMPT_BUDGET = 10
# Default of the runtime knob ``generation_attempts_with_context`` (M3R-110,
# 2026-09-01): the generator reads the knob, this constant is what the registry
# default equals (pinned by a test) and what tests reason about.
GENERATION_ATTEMPTS_WITH_CONTEXT = 5
CANDIDATE_TARGET = 5


def branching_aware_target(
    base_target: int,
    branching_samples: list[float],
    *,
    degenerate_max: float,
    floor: int,
) -> int:
    """Candidate target for a chain whose branching has been observed (M2R-110).

    On a near-degenerate chain, candidates 2..N are near-duplicates of the
    first — the scorer ends up choosing between copies of one walk — so the
    target drops to the floor and the generation stops early. Everything else
    keeps the configured target. The attempt budget is untouched either way, so
    a chain that keeps failing the gates still gets all of its tries.

    ``degenerate_max <= 0`` disables the rule (no pool has branching <= 0),
    restoring the fixed target this project shipped before Phase 2.
    """
    if degenerate_max <= 0.0 or not branching_samples:
        return base_target
    mean_branching = sum(branching_samples) / len(branching_samples)
    if mean_branching > degenerate_max:
        return base_target
    return max(1, min(base_target, floor))


def route_slot_budget(target: int, ratio: float) -> int:
    """Slots a route may take from INSIDE the pool budget (O10, ветвь 2).

    Маршруты клали кандидатов **сверх** заполненного пула, поэтому пул рос с
    каждым включённым маршрутом (замер 2026-09-01: C0 ровно 5, C4 в среднем
    6.40 при максимуме 7), а страховочный инвариант «ECB ≥ 4» сравнивал
    абсолютный счёт с плывущим знаменателем. Бюджет заставляет маршрут
    конкурировать за места: что взял маршрут, того не получит основной обход.

    Потолок в половину пула **побеждает** минимум «хотя бы один слот», и это
    осознанно: при пуле из одного слота конкурировать не за что, и маршрут не
    вправе выставить пул, в котором нет ни одного кандидата обхода.
    """
    if ratio <= 0.0 or target <= 0:
        return 0
    return min(max(1, round(target * ratio)), target // 2)
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
STALE_REPLY_REASON = "Generated reply repeats recent chat content"
EXTENSION_MAX_TOKENS = 10

CandidateScorer = Callable[[str, list[str], list[str], str], CandidateScore]

# Пустой контекст для базового скорера: его оценку релевантности конвейер
# всё равно заменяет вариантом с весами IDF (см. _build_score).
_NO_CONTEXT: list[str] = []


class VerbatimCopyChecker(Protocol):
    async def is_verbatim_copy(self, chat_id: int, text: str) -> bool: ...
    async def get_emoji_stats(self, chat_id: int) -> Mapping[str, int]: ...
    async def get_verbatim_ngram_index(
        self, chat_id: int
    ) -> AbstractSet[tuple[str, ...]]: ...
    async def get_context_idf(self, chat_id: int) -> Mapping[str, float]: ...
    async def get_intonation_profile(
        self, chat_id: int
    ) -> IntonationProfile | None: ...
    async def get_word_frequencies_by_ending(
        self, chat_id: int
    ) -> Mapping[str, Mapping[str, int]]: ...
    async def get_hot_ngrams(
        self,
        chat_id: int,
        *,
        min_count: int,
        recency_share: float,
        meme_ordering: bool = False,
    ) -> list[tuple[str, ...]]: ...
    async def get_order4_shadow_index(
        self, chat_id: int
    ) -> Mapping[tuple[str, str, str, str], Mapping[str, int]]: ...
    async def get_active_collocations(
        self, chat_id: int
    ) -> frozenset[tuple[str, str]]: ...


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    chat_id: int
    context_tokens: list[str]
    seed: list[str] | None
    current_message_normalized: str
    # M2R-210: the moment the temporal layer is evaluated at, as Unix seconds.
    # None means "now" — the bot's own path. The eval runner and the tests pass
    # a fixed value, which is what keeps a time-dependent weight reproducible
    # (design D3).
    now: int | None = None


@dataclass(frozen=True, slots=True)
class ResponseGenerationResult:
    text: str | None
    candidates_scored: int
    # M3R-143: маршрут победителя (значение CandidateRoute) — пер-ответный
    # сигнал шва дописки, который агрегаты M3R-103 дать не могут: молчаливая
    # связка «.» невидима текст-скану по построению (map §3.5). None, когда
    # ответа нет.
    winner_route: str | None = None


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    text: str
    score: CandidateScore
    # M3R-103: which mechanism built this candidate. Set where the candidate is
    # created, never inferred afterwards from its text — the whole point is to
    # attribute the pool without guessing, and the previous state of affairs
    # (seeded candidates indistinguishable from organic ones in the trace) is
    # exactly what forced an external harness to wrap the method to find out.
    route: str = CandidateRoute.VANILLA


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
    recent = runtime_state.recent_replies.setdefault(
        chat_id, deque(maxlen=RECENT_REPLY_LIMIT)
    )
    recent.append(normalized)


def repeats_recent_chat_content(
    runtime_state: RuntimeState,
    chat_id: int,
    candidate: str,
    candidate_normalized: str,
    current_message_normalized: str,
    is_short_candidate: bool,
) -> bool:
    """Single staleness gate: echo of the current message, a short reply from
    the recent-short window, or an exact repeat of a recently sent reply.

    Нормализации две, и они разные по смыслу: ``candidate_normalized`` — это
    санитизированный текст в нижнем регистре (та же форма, в которой хранятся
    короткие ответы), а ``normalize_reply_for_repeat`` дополнительно снимает
    хвостовые эмодзи и пунктуацию, чтобы кандидат совпал со своей же
    отправленной формой после вариатора концовки. Первая приходит готовой от
    вызывающего — раньше она считалась здесь второй раз.
    """
    if candidate_normalized == current_message_normalized:
        return True
    if is_short_candidate:
        recent_short = runtime_state.recent_short_replies.get(chat_id)
        if recent_short and candidate_normalized in recent_short:
            return True
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

    @property
    def entropy_sampling(self) -> EntropySampling:
        """M2R-100 settings for this chat, resolved from the runtime knobs.

        Applied to every walk that produces user-visible text, the verbatim
        extension included — a setting that governs "how the chain sounds"
        would otherwise stop at the sentence boundary where an extension begins.
        """
        state = self.runtime_state
        return EntropySampling(
            gain=state.markov_entropy_temp_gain,
            pivot=state.markov_entropy_pivot,
            temp_min=state.markov_entropy_temp_min,
            temp_max=state.markov_entropy_temp_max,
        )

    @property
    def interpolation(self) -> OrderInterpolation:
        """M2R-900 interpolation weight for this chat, from the runtime knob.

        Unlike the temporal blend above, beta does not follow mood: the phase
        measures one weight per arm, and tying it to mood would make the grid
        measure two things at once. The default is 0 — the neutral instance
        whose merge returns None before reading the order-2 projection.
        """
        return OrderInterpolation(
            beta=self.runtime_state.markov_interp_order2_weight
        )

    @property
    def temporal_blend(self) -> TemporalBlend:
        """M2R-210 blend settings for this chat, resolved from the runtime knobs.

        Alpha follows the chat's mood (TZ §8.3): a sleepy chat leans on what it
        has always said, a heated one on what it is saying right now. An unknown
        or absent mood falls back to the calm weight rather than to a guess.

        Every alpha defaults to 0, so this returns the neutral instance until
        someone deliberately raises a knob — and the neutral instance is the one
        that takes the early return inside the blend.
        """
        state = self.runtime_state
        alpha_by_mood = {
            SLEEPY: state.markov_alpha_sleepy,
            CALM: state.markov_alpha_calm,
            LIVELY: state.markov_alpha_lively,
            HEATED: state.markov_alpha_heated,
        }
        return TemporalBlend(
            alpha=alpha_by_mood.get(self.mood or CALM, state.markov_alpha_calm),
            half_life_days=state.markov_short_half_life_days,
            compression=state.markov_long_compression,
            beta=state.markov_long_compression_beta,
        )

    async def _candidate_reject_reason(
        self,
        request: GenerationRequest,
        candidate: str,
        candidate_normalized: str,
        is_short_candidate: bool,
    ) -> str | None:
        """Reason to discard a generated candidate, or None if it is usable.

        Two gates: staleness (echo of the current message or repeat of a
        recent reply) and verbatim training-sample copy (long replies only —
        short ones are governed by the short-reply anti-repeat).
        """
        if repeats_recent_chat_content(
            self.runtime_state,
            request.chat_id,
            candidate,
            candidate_normalized,
            request.current_message_normalized,
            is_short_candidate,
        ):
            return STALE_REPLY_REASON
        if not is_short_candidate and await self.learning_service.is_verbatim_copy(
            request.chat_id, candidate
        ):
            return VERBATIM_COPY_REASON
        return None

    async def _read_mutation_inputs(
        self, request: GenerationRequest
    ) -> tuple[Mapping[str, Mapping[str, int]], frozenset[str]]:
        """Frequency dictionary and protected hot-ngram words for slot mutations.

        Both reads happen once per request and only when the knob is on; the
        hot-ngram read is additionally skipped when the chat has no frequency
        data, because without it no mutation can be proposed anyway.
        """
        word_frequencies = (
            await self.learning_service.get_word_frequencies_by_ending(
                request.chat_id
            )
        )
        if not word_frequencies:
            return {}, frozenset()
        hot_ngrams = await self.learning_service.get_hot_ngrams(
            request.chat_id,
            min_count=self.runtime_state.hot_ngram_min_count,
            recency_share=self.runtime_state.hot_ngram_recency_share,
            meme_ordering=self.runtime_state.markov_hot_ngram_meme_ordering,
        )
        protected_tokens = frozenset(
            token.casefold() for ngram in hot_ngrams for token in ngram
        )
        return word_frequencies, protected_tokens

    def _plan_length(
        self,
        request: GenerationRequest,
        modifiers: MoodModifiers,
        intonation_profile: IntonationProfile | None,
        intonation_strength: float,
        rng: random.Random,
    ) -> tuple[str, int]:
        """Sample the length mode for this reply and the token ceiling it implies.

        Three influences stack onto the configured weights, in this order: the
        chat's observed length habits (P4 intonation), the mood multiplier, and
        length mirroring of the incoming message. Consumes exactly one draw
        from ``rng``.
        """
        base_weights = self.runtime_state.length_mode_weights
        if intonation_profile is not None:
            base_weights = blend_length_weights(
                base_weights,
                intonation_profile.length_weights,
                intonation_strength,
            )
        mood_weights = (
            base_weights[0] * modifiers.length_weight_mult[0],
            base_weights[1] * modifiers.length_weight_mult[1],
            base_weights[2] * modifiers.length_weight_mult[2],
        )
        # The message we are answering gets a say in how long the answer is:
        # people mirror each other's length, and the fixed weights did not.
        # Counted on the current message alone -- request.context_tokens also
        # carries the replied-to message, whose length is not what we mirror.
        incoming_tokens = len(
            content_tokens(tokenize(request.current_message_normalized))
        )
        conditioned_weights = context_length_weights(
            mood_weights,
            incoming_tokens,
            self.runtime_state.length_context_adaptation,
        )
        length_mode = sample_length_mode(conditioned_weights, rng, base_weights)
        max_tokens = self.runtime_state.max_reply_tokens
        if length_mode == "short":
            max_tokens = min(max_tokens, SHORT_MODE_MAX_TOKENS)
        return length_mode, max_tokens

    async def _evaluate_candidate(
        self, request: GenerationRequest, candidate: str
    ) -> tuple[list[str], bool, str | None]:
        """Tokenize ``candidate`` and run the rejection gates over it.

        Returns ``(tokens, is_short, reject_reason)``. Every extension in the
        attempt loop rewrites the candidate text and has to redo exactly this
        work on the new string, so it lives here instead of being spelled out
        at each of the three sites.
        """
        normalized = sanitize_text(candidate).lower()
        tokens = tokenize(
            candidate, normalize_lower=self.runtime_state.normalize_lower
        )
        is_short = is_short_generated_reply(tokens)
        reason = await self._candidate_reject_reason(
            request, candidate, normalized, is_short
        )
        return tokens, is_short, reason

    async def _extend_verbatim_candidate(
        self,
        request: GenerationRequest,
        candidate: str,
        rng: random.Random,
        now: int,
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
            order_mix_probability=state.order_mix_probability,
            entropy_sampling=self.entropy_sampling,
            temporal_blend=self.temporal_blend,
            interpolation=self.interpolation,
            now=now,
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
        connective = pick_splice_connective(
            rng,
            exclude=[
                phrase
                for phrase in JUMP_CONNECTIVE_TOKENS
                if tail_tokens and tail_tokens[0].casefold() in phrase
            ],
        )
        combined = finalize_reply_ending(
            base_tokens
            + splice_marker_tokens(base_tokens, connective)
            + tail_tokens
        )
        extended = detokenize(combined, max_chars=state.max_reply_chars)
        if not extended or extended == candidate:
            return None
        return extended

    def _collocation_delta(
        self, chat_id: int, tokens: list[str], active: frozenset[tuple[str, str]]
    ) -> float:
        """Net collocation effect on one candidate (M2R-320, ADR-016).

        Availability is answered from the transition pools the walks already
        cached — never a query per candidate per collocation (design D4). The
        applied and withheld counts go to telemetry: the configured weight is
        the intent, these counts are the effect.
        """
        if not active:
            return 0.0
        effect = collocation_effect(
            tokens,
            active,
            lambda state, token: self.generator.transition_was_available(
                chat_id, state, token
            ),
        )
        self.generator.telemetry.note_collocations(
            bonus_hits=effect.bonus_hits,
            penalty_hits=effect.penalty_hits,
            withheld=effect.withheld,
        )
        return effect.delta(
            self.runtime_state.markov_collocation_bonus,
            self.runtime_state.markov_collocation_break_penalty,
        )

    def _note_rejected(
        self, chat_id: int, route: str, reason: str, text: str
    ) -> None:
        """Record a discarded candidate in both the trace and the counters.

        Branches outside the main attempt loop discarded candidates silently
        until M3R-103, so «ветка ничего не дала» и «ветка дала, но всё
        отклонили» выглядели одинаково — нулём.
        """
        gen_trace_log.log_attempt_rejected(
            chat_id, 0, context_used=False, reason=reason, text=text, route=route
        )
        self.generator.telemetry.note_route_rejected(route, reason)

    async def _append_seeded_candidates(
        self,
        request: GenerationRequest,
        candidates: list[_ScoredCandidate],
        seen_candidates: set[str],
        *,
        length_mode: str,
        max_tokens: int,
        context_idf: Mapping[str, float],
        recent_trigrams: set[tuple[str, ...]],
        recent_penalty_strength: float,
        corpus_ngrams: AbstractSet[tuple[str, ...]],
        verbatim_penalty_strength: float,
        active_collocations: frozenset[tuple[str, str]],
        entropy_sampling: EntropySampling,
        temporal_blend: TemporalBlend,
        now: int,
        slots: int,
        rng: random.Random,
    ) -> set[str]:
        """Grow seeded candidates from the message's best anchors (M2R-410).

        ``slots`` — бюджет мест **внутри** пула, посчитанный вызывающим
        (``route_slot_budget``). Ветка политикой размера пула не владеет: когда
        маршрутов станет больше одного, делить бюджет придётся в одном месте,
        а не в каждой ветке по её формуле (O10, design D3).

        The seed ranking is computed once per reply (never per candidate). Each
        seeded slot takes the next distinct top seed above the minimum score and
        assembles a bidirectional candidate, which is then finished by the same
        pipeline as the main walk (``finalize_candidate_tokens``: tail trims plus
        the four form gates) and screened by the same staleness and verbatim-copy
        gates. Survivors join the pool with a full score and no priority.
        Returns the set of texts that are seeded, for telemetry.

        The "same gates" claim is load-bearing and was false until M3R-101: the
        branch went straight to ``detokenize``, so it skipped the tail pipeline
        and all four form gates while this docstring already claimed parity.
        """
        state = self.runtime_state
        message_tokens = tokenize(
            request.current_message_normalized,
            normalize_lower=state.normalize_lower,
        )
        ranked = await self.generator.rank_seeds(
            request.chat_id,
            message_tokens,
            min_support=state.markov_seed_min_support,
            branch_min=state.markov_seed_branch_min,
            branch_ideal=state.markov_seed_branch_ideal,
            branch_max=state.markov_seed_branch_max,
            min_token_len=state.markov_seed_min_token_len,
        )
        usable = [s for s in ranked if s.score >= state.markov_seed_min_score]
        if not usable:
            return set()
        # ``randomness_strength`` — шкала 0–3, а ``next_explore`` — вероятность
        # (markov.py сравнивает её с rng.random()): сырое значение >= 1.0 делало
        # исследование безусловным. Отображение то же, что в основном обходе
        # (см. ``_generate_text_once`` в markov.py).
        strength = max(0.0, min(3.0, state.randomness_strength))
        next_explore = min(0.98, 0.12 + 0.18 * strength)
        seeded_texts: set[str] = set()
        for seed_score in usable[:slots]:
            tokens = await self.generator.generate_seeded_candidate(
                request.chat_id,
                seed_score.token,
                max_tokens=max_tokens,
                head_share=state.markov_seed_head_share,
                next_explore=next_explore,
                next_power=1.0,
                repetition_penalty_strength=state.repetition_penalty_strength,
                entropy_sampling=entropy_sampling,
                temporal_blend=temporal_blend,
                now=now,
                rng=rng,
            )
            if not tokens:
                continue
            # Тот же хвостовой конвейер и те же четыре гейта формы, что у
            # основного обхода. До M3R-101 здесь стоял голый ``detokenize``, и
            # seeded-кандидат приходил в пул недоведённым: терминальная
            # пунктуация у 21% против 91% у органических, то есть он терял
            # ``CLEAN_END_BONUS`` и получал от скорера минус за собственную
            # сборку, а не за содержание (map §3.4, REVIEW §2).
            final = finalize_candidate_tokens(
                tokens,
                max_chars=state.max_reply_chars,
                context_tokens=request.context_tokens,
            )
            if final.rejection_reason is not None:
                self._note_rejected(
                    request.chat_id,
                    CandidateRoute.SEEDED,
                    final.rejection_reason,
                    detokenize(tokens, max_chars=state.max_reply_chars),
                )
                continue
            text = final.text
            if text in seen_candidates:
                continue
            candidate_tokens, is_short, reject = await self._evaluate_candidate(
                request, text
            )
            if reject is not None:
                self._note_rejected(
                    request.chat_id, CandidateRoute.SEEDED, reject, text
                )
                continue
            seen_candidates.add(text)
            seeded_texts.add(text)
            score = self._build_score(
                text,
                candidate_tokens,
                request.context_tokens,
                length_mode,
                context_idf=context_idf,
                recent_trigrams=recent_trigrams,
                recent_penalty_strength=recent_penalty_strength,
                corpus_ngrams=corpus_ngrams,
                verbatim_penalty_strength=verbatim_penalty_strength,
                chat_id=request.chat_id,
                active_collocations=active_collocations,
            )
            candidates.append(
                _ScoredCandidate(
                    text=text, score=score, route=CandidateRoute.SEEDED
                )
            )
        return seeded_texts

    def _build_score(
        self,
        text: str,
        tokens: list[str],
        context_tokens: list[str],
        length_mode: str,
        *,
        context_idf: Mapping[str, float],
        recent_trigrams: set[tuple[str, ...]],
        recent_penalty_strength: float,
        corpus_ngrams: AbstractSet[tuple[str, ...]],
        verbatim_penalty_strength: float,
        chat_id: int,
        active_collocations: frozenset[tuple[str, str]] = frozenset(),
    ) -> CandidateScore:
        """Full candidate score with the per-request penalty components.

        Контекст в базовый скорер не передаётся намеренно: его оценку
        релевантности конвейер всё равно заменяет вариантом с весами IDF (он
        учитывает редкость совпавших токенов и отсекает пересказ контекста),
        поэтому считать её дважды незачем.
        """
        return replace(
            self.scorer(text, tokens, _NO_CONTEXT, length_mode),
            context_relevance=idf_context_relevance(
                tokens, context_tokens, context_idf
            ),
            recent_penalty=recent_penalty_strength
            * recent_reply_overlap(tokens, recent_trigrams),
            # M3R-120: гард «одной признанной единицы» применяется здесь и
            # только здесь. Триггер verbatim-дописки ниже намеренно остаётся
            # на сырой доле: дописка существует, чтобы добавить отсебятину к
            # почти-цитате, и наличие признанной единицы этого не отменяет
            # (design D4).
            verbatim_penalty=verbatim_penalty_strength
            * verbatim_quote_severity(
                verbatim_ngram_overlap(
                    tokens,
                    corpus_ngrams,
                    exempt_recognized_unit=(
                        self.runtime_state.verbatim_recognized_unit
                    ),
                )
            ),
            collocation_delta=self._collocation_delta(
                chat_id, tokens, active_collocations
            ),
        )

    async def _mutated_variant(
        self,
        request: GenerationRequest,
        candidate_tokens: list[str],
        *,
        frequencies: Mapping[str, Mapping[str, int]],
        protected_tokens: frozenset[str],
        rng: random.Random,
    ) -> tuple[str, list[str]] | None:
        """A slot-mutated copy of the candidate that passed the reject gates.

        Returns None when no mutation applies or the mutated text fails the
        same staleness/verbatim gates the original went through.
        """
        mutated_tokens = mutate_candidate_tokens(
            candidate_tokens,
            frequencies=frequencies,
            protected_tokens=protected_tokens,
            context_tokens=request.context_tokens,
            rng=rng,
        )
        if mutated_tokens is None:
            return None
        mutated_text = detokenize(
            mutated_tokens, max_chars=self.runtime_state.max_reply_chars
        )
        if not mutated_text:
            return None
        # Токены выводятся из текста, а не берутся до него. `detokenize`
        # обрывает список на символьном пределе, а вернуть исходный список
        # значило бы скорить объект, которого в чате не будет: `natural_length`,
        # `repetition_penalty`, `idf_context_relevance` и анти-цитата читали бы
        # хвост, отброшенный при сборке. Механизм срабатывания не
        # гипотетический — замена может быть длиннее оригинала на 40%
        # (`MAX_LENGTH_DELTA_SHARE`), а прогулка сама останавливается ровно на
        # `max_chars`, поэтому кандидаты приходят впритык.
        #
        # Это был единственный маршрут с таким расхождением: vanilla и
        # extension получают токены через `_evaluate_candidate`, seeded — через
        # `finalize_candidate_tokens`, и оба выводят их из финального текста.
        mutated_tokens = tokenize(
            mutated_text, normalize_lower=self.runtime_state.normalize_lower
        )
        if not mutated_tokens:
            return None
        reject_reason = await self._candidate_reject_reason(
            request,
            mutated_text,
            sanitize_text(mutated_text).lower(),
            is_short_generated_reply(mutated_tokens),
        )
        if reject_reason is not None:
            return None
        return mutated_text, mutated_tokens

    async def generate_with_result(
        self,
        request: GenerationRequest,
        rng: random.Random | None = None,
        candidate_target: int = CANDIDATE_TARGET,
    ) -> ResponseGenerationResult:
        generation_rng = rng or random.Random()
        # M3R-140: the mode is noted before anything can fail, so a generation
        # that collects no candidates still counts. The share of the two modes
        # is what weights every PRE gate verdict, and it is unrecoverable
        # afterwards — nothing persists whether a reply was asked with context.
        self.generator.telemetry.note_context_mode(
            with_context=bool(request.context_tokens)
        )
        modifiers = self.mood_modifiers or NEUTRAL_MODIFIERS
        seed = request.seed
        target = max(1, min(candidate_target, GENERATION_ATTEMPT_BUDGET))
        # M2R-110: the target may shrink once the walk shows how much choice it
        # actually had. Starts at the configured value and is recomputed after
        # each accepted candidate.
        effective_target = target
        branching_samples: list[float] = []
        candidates: list[_ScoredCandidate] = []
        seen_candidates: set[str] = set()
        attempt_jump_counts: dict[str, int] = {}
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
        corpus_ngrams: AbstractSet[tuple[str, ...]] = frozenset()
        if verbatim_penalty_strength > 0.0:
            corpus_ngrams = await self.learning_service.get_verbatim_ngram_index(
                request.chat_id
            )
        # M2R-320: active collocations are read once per reply, never per
        # candidate — and not at all while both scoring weights sit at their
        # neutral zero, so installing the capability adds no query and changes
        # no reply (the generation_hash contract).
        active_collocations: frozenset[tuple[str, str]] = frozenset()
        if (
            self.runtime_state.markov_collocation_bonus > 0.0
            or self.runtime_state.markov_collocation_break_penalty > 0.0
        ):
            active_collocations = (
                await self.learning_service.get_active_collocations(
                    request.chat_id
                )
            )
        # Topic overlap is scored by how much of the context's *informative* mass
        # the candidate echoes, not by raw token count -- otherwise a short reply
        # sharing a pronoun beats a long one sharing a proper noun.
        context_idf: Mapping[str, float] = {}
        if request.context_tokens:
            context_idf = await self.learning_service.get_context_idf(
                request.chat_id
            )
        # P4 intonation: blend the configured mode weights toward the chat's
        # observed length habits before mood and mirroring apply on top. The
        # profile read is skipped entirely when the knob is off; None means
        # the chat is still below the profile's message floor.
        intonation_strength = self.runtime_state.intonation_profile_strength
        intonation_profile: IntonationProfile | None = None
        if intonation_strength > 0.0:
            intonation_profile = await self.learning_service.get_intonation_profile(
                request.chat_id
            )
        # Slot mutations: the frequency dictionary and the protected hot-ngram
        # words are fetched once per request, and only when the knob is on.
        slot_mutation_probability = self.runtime_state.slot_mutation_probability
        word_frequencies: Mapping[str, Mapping[str, int]] = {}
        protected_tokens: frozenset[str] = frozenset()
        if slot_mutation_probability > 0.0:
            word_frequencies, protected_tokens = await self._read_mutation_inputs(
                request
            )
        length_mode, max_tokens = self._plan_length(
            request,
            modifiers,
            intonation_profile,
            intonation_strength,
            generation_rng,
        )
        effective_randomness = max(
            0.0, self.runtime_state.randomness_strength + modifiers.randomness_delta
        )
        entropy_sampling = self.entropy_sampling
        temporal_blend = self.temporal_blend
        interpolation = self.interpolation
        # M2R-210 / design D3: one moment for the whole generation. Reading the
        # clock per step would make two runs of the same seed differ by however
        # long the run took, and the eval protocol's bit-for-bit requirement —
        # what makes every phase verdict auditable — would stop holding.
        now = request.now if request.now is not None else int(time.time())
        # M3R-110: a knob, not the constant — the eval matrix sweeps it and
        # /set can move the drop point live.
        attempts_with_context = self.runtime_state.generation_attempts_with_context

        gen_trace_log.log_request_header(
            request.chat_id,
            length_mode=length_mode,
            randomness=effective_randomness,
            temperature=self.runtime_state.candidate_selection_temperature,
            target=target,
            budget=GENERATION_ATTEMPT_BUDGET,
            attempts_with_context=attempts_with_context,
            recent_penalty_strength=recent_penalty_strength,
            verbatim_penalty_strength=verbatim_penalty_strength,
        )

        # O10 (route-slot-budget): маршруты берут слоты ИЗНУТРИ бюджета пула,
        # а не сверх него — иначе пул растёт с каждым включённым маршрутом
        # (замер: C0 ровно 5, C4 в среднем 6.40 при максимуме 7), и
        # пре-регистрированный порог ECB ≥ 4 сравнивает абсолютный счёт с
        # плывущим знаменателем.
        #
        # Маршрут идёт ПЕРВЫМ, и это то, что позволяет удержать бюджет, не
        # уменьшая пул: условие остановки цикла ниже уже считает общий размер
        # пула, поэтому обход просто добирает то, что маршрут не занял. Резерв
        # с маршрутом после цикла давал бы недозаполненный пул каждый раз,
        # когда маршрут ничего не произвёл (design D1).
        seeded_budget = route_slot_budget(
            target, self.runtime_state.markov_seeded_candidate_ratio
        )
        seeded_texts: set[str] = set()
        if seeded_budget > 0:
            seeded_texts = await self._append_seeded_candidates(
                request,
                candidates,
                seen_candidates,
                length_mode=length_mode,
                max_tokens=max_tokens,
                context_idf=context_idf,
                recent_trigrams=recent_trigrams,
                recent_penalty_strength=recent_penalty_strength,
                corpus_ngrams=corpus_ngrams,
                verbatim_penalty_strength=verbatim_penalty_strength,
                active_collocations=active_collocations,
                entropy_sampling=entropy_sampling,
                temporal_blend=temporal_blend,
                now=now,
                slots=seeded_budget,
                rng=generation_rng,
            )

        # M3R-141: a generation that starts with context and runs past the
        # with-context budget finishes on a different mechanism than the one it
        # was asked with — measured at 37% of ctx-answers (map §1.3) and, until
        # now, invisible: neither the trace nor telemetry said it happened.
        context_dropped = False
        for attempt in range(GENERATION_ATTEMPT_BUDGET):
            attempt_context_tokens = (
                request.context_tokens
                if attempt < attempts_with_context
                else None
            )
            if request.context_tokens and attempt_context_tokens is None:
                if not context_dropped:
                    gen_trace_log.log_context_dropped(
                        request.chat_id,
                        attempt + 1,
                        attempts_with_context=attempts_with_context,
                    )
                context_dropped = True
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
                jump_probability=self.runtime_state.markov_jump_probability,
                context_jump_boost=self.runtime_state.context_jump_boost,
                order_mix_probability=self.runtime_state.order_mix_probability,
                context_anchor_splice_probability=(
                    self.runtime_state.context_anchor_splice_probability
                ),
                entropy_sampling=entropy_sampling,
                temporal_blend=temporal_blend,
                interpolation=interpolation,
                now=now,
                rng=generation_rng,
                attempt_budget=1,
            )
            if candidate:
                # M2R-020: jump count per candidate text — the shadow selector
                # below skips replies whose token adjacency crosses a splice.
                # getattr-guarded: test doubles stub the trace with namespaces;
                # an unknown jump count (-1) simply skips the shadow measure.
                attempt_jump_counts[candidate] = getattr(
                    candidate_trace, "jump_count", -1
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
                (
                    candidate_tokens,
                    is_short_candidate,
                    reject_reason,
                ) = await self._evaluate_candidate(request, candidate)
                was_extended = False
                if reject_reason == VERBATIM_COPY_REASON:
                    # Corpus replies are welcome — bare quotes are not: extend
                    # the copy with отсебятина and re-run the gates on the
                    # combined text instead of discarding the attempt.
                    extended = await self._extend_verbatim_candidate(
                        request, candidate, generation_rng, now
                    )
                    if extended is not None:
                        gen_trace_log.log_attempt_extended(
                            request.chat_id,
                            attempt + 1,
                            original=candidate,
                            extended=extended,
                        )
                        was_extended = True
                        candidate = extended
                        (
                            candidate_tokens,
                            is_short_candidate,
                            reject_reason,
                        ) = await self._evaluate_candidate(request, candidate)
                if (
                    reject_reason is None
                    and not was_extended
                    and not is_short_candidate
                    and self.runtime_state.verbatim_extension_share > 0.0
                    and corpus_ngrams
                    and verbatim_ngram_overlap(candidate_tokens, corpus_ngrams)
                    >= self.runtime_state.verbatim_extension_share
                ):
                    # Near-quote (a training message with a word or two
                    # changed): passes the exact-copy gate but still reads as
                    # a replay. Splice отсебятина the same way full copies
                    # get; keep the original candidate when the combined text
                    # fails the gates (the original already passed them).
                    extended = await self._extend_verbatim_candidate(
                        request, candidate, generation_rng, now
                    )
                    if extended is not None:
                        extended_tokens, _, extended_reject = (
                            await self._evaluate_candidate(request, extended)
                        )
                        if extended_reject is None:
                            gen_trace_log.log_attempt_extended(
                                request.chat_id,
                                attempt + 1,
                                original=candidate,
                                extended=extended,
                            )
                            was_extended = True
                            candidate = extended
                            candidate_tokens = extended_tokens
                if reject_reason is not None:
                    logger.debug(
                        "%s, retrying: chat=%s attempt=%s",
                        reject_reason,
                        mask_chat_id(request.chat_id),
                        attempt + 1,
                    )
                    route = (
                        CandidateRoute.EXTENSION
                        if was_extended
                        else CandidateRoute.VANILLA
                    )
                    gen_trace_log.log_attempt_rejected(
                        request.chat_id,
                        attempt + 1,
                        context_used=bool(attempt_context_tokens),
                        reason=reject_reason,
                        text=candidate,
                        route=route,
                    )
                    self.generator.telemetry.note_route_rejected(
                        route, reject_reason
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
                        score = self._build_score(
                            candidate,
                            candidate_tokens,
                            request.context_tokens,
                            length_mode,
                            context_idf=context_idf,
                            recent_trigrams=recent_trigrams,
                            recent_penalty_strength=recent_penalty_strength,
                            corpus_ngrams=corpus_ngrams,
                            verbatim_penalty_strength=verbatim_penalty_strength,
                            chat_id=request.chat_id,
                            active_collocations=active_collocations,
                        )
                        candidates.append(
                            _ScoredCandidate(
                                text=candidate,
                                score=score,
                                route=(
                                    CandidateRoute.EXTENSION
                                    if was_extended
                                    else CandidateRoute.VANILLA
                                ),
                            )
                        )
                        # M2R-110: the accepted candidate's own mean branching
                        # is what tells us whether more attempts would produce
                        # anything different. getattr-guarded like the jump
                        # count above — test doubles stub the trace.
                        candidate_branching = float(
                            getattr(candidate_trace, "mean_branching", 0.0) or 0.0
                        )
                        if candidate_branching > 0.0:
                            branching_samples.append(candidate_branching)
                        effective_target = branching_aware_target(
                            target,
                            branching_samples,
                            degenerate_max=(
                                self.runtime_state.markov_branching_degenerate_max
                            ),
                            floor=self.runtime_state.markov_branching_candidate_floor,
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
                        if (
                            word_frequencies
                            and len(candidates) < effective_target
                            and generation_rng.random() < slot_mutation_probability
                        ):
                            variant = await self._mutated_variant(
                                request,
                                candidate_tokens,
                                frequencies=word_frequencies,
                                protected_tokens=protected_tokens,
                                rng=generation_rng,
                            )
                            if (
                                variant is not None
                                and variant[0] not in seen_candidates
                            ):
                                mutated_text, mutated_tokens = variant
                                seen_candidates.add(mutated_text)
                                mutated_score = self._build_score(
                                    mutated_text,
                                    mutated_tokens,
                                    request.context_tokens,
                                    length_mode,
                                    context_idf=context_idf,
                                    recent_trigrams=recent_trigrams,
                                    recent_penalty_strength=recent_penalty_strength,
                                    corpus_ngrams=corpus_ngrams,
                                    verbatim_penalty_strength=verbatim_penalty_strength,
                                    chat_id=request.chat_id,
                                    active_collocations=active_collocations,
                                )
                                candidates.append(
                                    _ScoredCandidate(
                                        text=mutated_text,
                                        score=mutated_score,
                                        route=CandidateRoute.MUTATED,
                                    )
                                )
                                gen_trace_log.log_attempt_mutated(
                                    request.chat_id,
                                    attempt + 1,
                                    original=candidate,
                                    mutated=mutated_text,
                                    index=len(candidates),
                                    score=mutated_score,
                                )
                        if len(candidates) >= effective_target:
                            break
                    else:
                        gen_trace_log.log_attempt_duplicate(
                            request.chat_id,
                            attempt + 1,
                            context_used=bool(attempt_context_tokens),
                        )

            seed = None

        if context_dropped:
            self.generator.telemetry.note_context_dropped()

        # M3R-103. ``attempted`` is deliberately not "every route in the enum":
        # a route whose knob is off must stay distinguishable from one that ran
        # and produced nothing, and only the knobs know which is which.
        # Computed before the empty-pool return for the same reason
        # ``note_context_mode`` is called first: a generation where every route
        # ran and every candidate was rejected is precisely the case the pair of
        # denominators exists to show. Counting it only on success inflates both
        # route_present/route_attempts and route_won/route_present, and it
        # inflates them towards "the route is useful" — the worst direction for
        # a promotion gate (M3R-220).
        attempted = {CandidateRoute.VANILLA, CandidateRoute.EXTENSION}
        # Ручка включена, но бюджет слотов нулевой (пул из одного места) —
        # маршрут НЕ запускался, и знаменатель «маршрут отработал» его считать
        # не должен: иначе «не было места» слилось бы с «место было, кандидата
        # нет» — ровно та пара, ради различения которой заведены счётчики.
        if seeded_budget > 0:
            attempted.add(CandidateRoute.SEEDED)
        if self.runtime_state.slot_mutation_probability > 0.0:
            attempted.add(CandidateRoute.MUTATED)

        if not candidates:
            self.generator.telemetry.note_routes(
                attempted=attempted,
                present=set(),
                winner=None,
            )
            # `note_seeded` — по тем же основаниям и в ту же сторону. Первая
            # редакция этой правки его пропустила, и знаменатель
            # `seeded_generations` остался считаться только на успешном пути:
            # `seeded_present_rate` и `seeded_win_rate_given_present` были
            # завышены ровно на долю провальных генераций. Именно по этим двум
            # числам в /stats принимается решение M2R-430.
            if seeded_budget > 0:
                self.generator.telemetry.note_seeded(present=False, won=False)
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
        if seeded_budget > 0:
            self.generator.telemetry.note_seeded(
                present=bool(seeded_texts),
                won=selected.text in seeded_texts,
            )
        self.generator.telemetry.note_routes(
            attempted=attempted,
            present={candidate.route for candidate in candidates},
            winner=selected.route,
        )
        gen_trace_log.log_selection(
            request.chat_id,
            candidates,
            temperature=self.runtime_state.candidate_selection_temperature,
            margin=SELECTION_SCORE_MARGIN,
            selection_margin_used=True,
            selected=selected,
        )
        # M2R-020 shadow order-4 selector: measurement only, after selection,
        # over the winner's raw token sequence. Skipped for replies with jumps
        # or texts not produced by a single walk (extensions/mutations) — their
        # adjacency crosses splice boundaries. getattr-guarded: test doubles
        # and older harnesses may not implement the protocol method yet.
        if (
            getattr(self.runtime_state, "markov_shadow_order4_enabled", False)
            and attempt_jump_counts.get(selected.text) == 0
        ):
            shadow_index_getter = getattr(
                self.learning_service, "get_order4_shadow_index", None
            )
            if shadow_index_getter is not None:
                shadow_index = await shadow_index_getter(request.chat_id)
                eligible, chosen = shadow_order4_stats(
                    tokenize(selected.text), shadow_index
                )
                self.generator.telemetry.note_shadow(
                    eligible=eligible, selected=chosen
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
            ending_profile=intonation_profile,
            profile_strength=intonation_strength,
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
            winner_route=selected.route,
        )
