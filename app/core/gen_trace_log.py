"""Verbose step-by-step logging of the candidate-selection pipeline.

Test-time instrumentation for understanding how a reply is chosen from up to
``CANDIDATE_TARGET`` candidates: it traces the route each generation attempt
took (Markov order, start source, context-match kind, jumps), the full score
breakdown of every accepted candidate, and the softmax weights of the final
pick. Everything is emitted on the ``chat_markov.gen`` logger at INFO so it can
be followed live without turning the whole app to DEBUG.

This module has no behavioural effect; deleting it (and its call sites in
``response_generator``) restores the silent pipeline.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from app.core.candidate_scorer import CandidateScore
from app.log_masking import mask_chat_id

if TYPE_CHECKING:
    from app.core.markov import GenerationTrace

gen_logger = logging.getLogger("chat_markov.gen")


class _HasScore(Protocol):
    text: str
    score: CandidateScore


def enabled() -> bool:
    """True when the gen-trace logger would actually emit at INFO."""
    return gen_logger.isEnabledFor(logging.INFO)


def log_request_header(
    chat_id: int,
    *,
    length_mode: str,
    randomness: float,
    temperature: float,
    target: int,
    budget: int,
    attempts_with_context: int,
    recent_penalty_strength: float,
    verbatim_penalty_strength: float,
) -> None:
    if not enabled():
        return
    gen_logger.info(
        "\n"
        "========== GENERATION chat=%s ==========\n"
        "  goal: collect up to %d candidates (attempt budget=%d, "
        "first %d attempts use context)\n"
        "  length_mode=%s  base_randomness=%.3f  selection_temperature=%.3f\n"
        "  penalties: recent=%.2f verbatim=%.2f",
        mask_chat_id(chat_id),
        target,
        budget,
        attempts_with_context,
        length_mode,
        randomness,
        temperature,
        recent_penalty_strength,
        verbatim_penalty_strength,
    )


def _route(trace: GenerationTrace) -> str:
    """One-line description of which branches the generator walked."""
    if trace.context_exact_matches:
        ctx = f"context=EXACTx{trace.context_exact_matches}"
    elif trace.context_casefold_matches:
        ctx = f"context=CASEFOLDx{trace.context_casefold_matches}"
    elif trace.context_prefix_matches:
        singleton = (
            f" (singletonx{trace.context_prefix_singleton_matches})"
            if trace.context_prefix_singleton_matches
            else ""
        )
        ctx = f"context=PREFIXx{trace.context_prefix_matches}{singleton}"
    elif trace.hidden_context_fallbacks:
        ctx = f"context=HIDDEN_FALLBACKx{trace.hidden_context_fallbacks}"
    else:
        ctx = "context=none"
    return (
        f"start_source={trace.start_source} "
        f"markov_order_used={trace.markov_order_used} "
        f"{ctx} jumps={trace.jump_count} "
        f"lead_punct_stripped={trace.leading_punctuation_stripped}"
    )


def log_attempt_failed(_chat_id: int, attempt: int, *, context_used: bool) -> None:
    if not enabled():
        return
    gen_logger.info(
        "  attempt %02d [context=%s] -> FAILED (generator produced nothing)",
        attempt,
        "on" if context_used else "off",
    )


def log_attempt_rejected(
    _chat_id: int, attempt: int, *, context_used: bool, reason: str, text: str
) -> None:
    if not enabled():
        return
    gen_logger.info(
        "  attempt %02d [context=%s] -> REJECTED (%s)\n"
        "        text: %r",
        attempt,
        "on" if context_used else "off",
        reason,
        text,
    )


def log_attempt_duplicate(_chat_id: int, attempt: int, *, context_used: bool) -> None:
    if not enabled():
        return
    gen_logger.info(
        "  attempt %02d [context=%s] -> DUPLICATE of an already-collected "
        "candidate (dropped)",
        attempt,
        "on" if context_used else "off",
    )


def log_attempt_accepted(
    _chat_id: int,
    attempt: int,
    *,
    context_used: bool,
    index: int,
    trace: GenerationTrace,
    score: CandidateScore,
    text: str,
) -> None:
    if not enabled():
        return
    gen_logger.info(
        "  attempt %02d [context=%s] -> ACCEPTED as candidate #%d\n"
        "        route: %s\n"
        "        text : %r\n"
        "        score: total=%+.3f = "
        "completion=%+.3f + diversity=%+.3f + natural_len=%+.3f "
        "+ context_rel=%+.3f\n"
        "               - repetition=%.3f - recent=%.3f - verbatim=%.3f "
        "- coherence=%.3f",
        attempt,
        "on" if context_used else "off",
        index,
        _route(trace),
        text,
        score.total,
        score.completion_quality,
        score.lexical_diversity,
        score.natural_length,
        score.context_relevance,
        score.repetition_penalty,
        score.recent_penalty,
        score.verbatim_penalty,
        score.coherence_penalty,
    )


def log_selection(
    _chat_id: int,
    candidates: Sequence[_HasScore],
    *,
    temperature: float,
    margin: float,
    selection_margin_used: bool,
    selected: _HasScore | None = None,
) -> None:
    """Log the final softmax pick over the collected candidates."""
    if not enabled():
        return
    if not candidates or selected is None:
        gen_logger.info("  SELECTION: no candidates collected -> no reply")
        return
    best_total = max(c.score.total for c in candidates)
    argmax = not selection_margin_used or temperature <= 0.0 or len(candidates) == 1

    lines = [
        f"  ---------- SELECTION over {len(candidates)} candidate(s) "
        f"----------",
        f"  best_total={best_total:+.3f}  temperature={temperature:.3f}  "
        f"margin={margin:.2f}  mode={'ARGMAX' if argmax else 'SOFTMAX'}",
    ]

    if argmax:
        for i, c in enumerate(candidates, 1):
            mark = "  <== CHOSEN" if c is selected else ""
            lines.append(
                f"    #{i} total={c.score.total:+.3f}{mark}  {c.text!r}"
            )
        gen_logger.info("\n".join(lines))
        return

    eligible = [
        c for c in candidates if c.score.total >= best_total - margin
    ]
    weights = [
        math.exp((c.score.total - best_total) / temperature) for c in eligible
    ]
    weight_sum = sum(weights) or 1.0
    eligible_ids = {id(c) for c in eligible}
    weight_by_id = {id(c): w for c, w in zip(eligible, weights, strict=True)}

    for i, c in enumerate(candidates, 1):
        mark = "  <== CHOSEN" if c is selected else ""
        if id(c) in eligible_ids:
            w = weight_by_id[id(c)]
            prob = 100.0 * w / weight_sum
            lines.append(
                f"    #{i} total={c.score.total:+.3f}  eligible  "
                f"weight={w:.4f}  p={prob:5.1f}%{mark}  {c.text!r}"
            )
        else:
            gap = best_total - c.score.total
            lines.append(
                f"    #{i} total={c.score.total:+.3f}  "
                f"excluded (>{margin:.2f} below best, gap={gap:.3f})  "
                f"{c.text!r}"
            )
    gen_logger.info("\n".join(lines))
