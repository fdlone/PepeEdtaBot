"""Chat intonation profile: the chat's own length and ending-punctuation habits.

Pure functions only (no I/O). The profile is computed on demand from the
chat's retained messages (the ``messages`` table window, already kept for the
verbatim gate) and cached by ``LearningService`` — nothing new is learned or
stored, and the retention window is what keeps the profile current.

Applied in two places, both scaled by ``intonation_profile_strength``
(0 disables): the reply length-mode weights are blended toward the chat's
observed short/medium/long shares, and the ending-flavor probabilities are
blended toward the chat's observed final-punctuation shares.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.core.candidate_scorer import LENGTH_MODE_BANDS
from app.core.markov import content_tokens, tokenize

# Below this many usable messages the profile is not applied at all: early
# chats would imprint a distribution made of noise.
MIN_PROFILE_MESSAGES = 200
# A mode share never blends below this floor, so a chat of one-liners cannot
# zero out long replies entirely — rarity is fine, extinction is not.
MIN_LENGTH_MODE_SHARE = 0.05

_SHORT_MAX = LENGTH_MODE_BANDS["short"][1]
_LONG_MIN = LENGTH_MODE_BANDS["long"][0]
_TERMINAL_CHARS = (".", "!", "?", "…")


@dataclass(frozen=True, slots=True)
class IntonationProfile:
    """Observed shares of the chat's message shapes.

    ``length_weights`` sums to 1.0 (short, medium, long). The ending shares
    are fractions of all profiled messages and do not have to sum to 1.0 —
    the remainder is the plain-period/question tail the flavor pass leaves
    alone.
    """

    length_weights: tuple[float, float, float]
    ending_none_share: float
    ending_ellipsis_share: float
    ending_exclamation_share: float


def build_intonation_profile(
    messages: Iterable[str],
    *,
    min_messages: int = MIN_PROFILE_MESSAGES,
) -> IntonationProfile | None:
    """Profile of the given normalized messages, or None below the floor."""
    length_counts = [0, 0, 0]
    none_endings = 0
    ellipsis_endings = 0
    exclamation_endings = 0
    total = 0
    for message in messages:
        stripped = message.strip()
        if not stripped:
            continue
        token_count = len(content_tokens(tokenize(stripped)))
        if token_count == 0:
            continue
        total += 1
        if token_count <= _SHORT_MAX:
            length_counts[0] += 1
        elif token_count >= _LONG_MIN:
            length_counts[2] += 1
        else:
            length_counts[1] += 1
        if stripped.endswith(("...", "…")):
            ellipsis_endings += 1
        elif stripped.endswith("!"):
            exclamation_endings += 1
        elif not stripped.endswith(_TERMINAL_CHARS):
            none_endings += 1
    if total < min_messages:
        return None
    floored = [
        max(count / total, MIN_LENGTH_MODE_SHARE) for count in length_counts
    ]
    norm = sum(floored)
    return IntonationProfile(
        length_weights=(
            floored[0] / norm,
            floored[1] / norm,
            floored[2] / norm,
        ),
        ending_none_share=none_endings / total,
        ending_ellipsis_share=ellipsis_endings / total,
        ending_exclamation_share=exclamation_endings / total,
    )


def blend_length_weights(
    base_weights: tuple[float, float, float],
    profile_weights: tuple[float, float, float],
    strength: float,
) -> tuple[float, float, float]:
    """Linear blend of the configured mode weights toward the chat profile.

    Both sides are normalized first so the blend is between distributions,
    not raw magnitudes; strength is clamped to [0, 1] and 0 returns the
    base weights untouched.
    """
    if strength <= 0.0:
        return base_weights
    scale = min(strength, 1.0)
    base_sum = sum(base_weights)
    if base_sum <= 0.0:
        return base_weights
    return (
        (1.0 - scale) * base_weights[0] / base_sum + scale * profile_weights[0],
        (1.0 - scale) * base_weights[1] / base_sum + scale * profile_weights[1],
        (1.0 - scale) * base_weights[2] / base_sum + scale * profile_weights[2],
    )
