"""Per-chat mood state (Stage 3 M1).

A hidden per-chat variable that makes the bot's *behaviour* drift with the
rhythm of the conversation. It is computed from cheap signals already flowing
through ``on_text_message`` — message rate, emphatic punctuation / caps, and how
often the bot is addressed — smoothed with EWMAs so it reacts to trends rather
than single messages.

This module is intentionally pure and side-effect free: it owns the signal math
and the classification, while ``RuntimeState`` owns storage and the learning
handler owns the wiring. Everything is deterministic given its inputs so the
state machine can be unit-tested with synthetic message timelines.
"""
from __future__ import annotations

from dataclasses import dataclass

# Mood states, ordered roughly from quiet to intense.
SLEEPY = "sleepy"
CALM = "calm"
LIVELY = "lively"
HEATED = "heated"

MOODS: tuple[str, ...] = (SLEEPY, CALM, LIVELY, HEATED)


@dataclass(frozen=True, slots=True)
class MoodConfig:
    """Thresholds and smoothing for mood classification (all registry-tunable)."""

    ewma_alpha: float
    lively_rate_per_min: float
    sleepy_rate_per_min: float
    heated_intensity: float
    max_rate_per_min: float


@dataclass(frozen=True, slots=True)
class ChatMoodState:
    """Smoothed per-chat signals plus the derived mood label."""

    mood: str
    rate_ewma: float
    intensity_ewma: float
    mention_ewma: float
    last_ts: float


@dataclass(frozen=True, slots=True)
class MoodModifiers:
    """How a mood nudges generation knobs, relative to neutral (all 1.0 / 0.0)."""

    reply_probability_mult: float
    randomness_delta: float
    length_weight_mult: tuple[float, float, float]
    flavor_strength_mult: float


NEUTRAL_MODIFIERS = MoodModifiers(
    reply_probability_mult=1.0,
    randomness_delta=0.0,
    length_weight_mult=(1.0, 1.0, 1.0),
    flavor_strength_mult=1.0,
)

# Per-mood defaults at modulation strength 1.0. ``calm`` is neutral by design;
# the others push the knobs in the direction the plan describes. Every deviation
# is scaled by ``mood_modulation_strength`` in ``modifiers_for_mood`` so the whole
# effect dials from 0 (off) through 1 (this table) upward.
_MOOD_MODIFIERS: dict[str, MoodModifiers] = {
    CALM: NEUTRAL_MODIFIERS,
    LIVELY: MoodModifiers(
        reply_probability_mult=1.5,
        randomness_delta=0.2,
        length_weight_mult=(0.8, 1.0, 1.2),
        flavor_strength_mult=1.1,
    ),
    HEATED: MoodModifiers(
        reply_probability_mult=1.3,
        randomness_delta=0.5,
        length_weight_mult=(1.1, 1.0, 0.9),
        flavor_strength_mult=1.4,
    ),
    SLEEPY: MoodModifiers(
        reply_probability_mult=0.5,
        randomness_delta=-0.3,
        length_weight_mult=(1.6, 1.0, 0.5),
        flavor_strength_mult=0.8,
    ),
}


def modifiers_for_mood(mood: str, strength: float) -> MoodModifiers:
    """Return the modifiers for ``mood``, with each deviation scaled by ``strength``.

    ``strength`` 0.0 collapses everything onto :data:`NEUTRAL_MODIFIERS` (mood is
    still tracked and logged but changes no behaviour); 1.0 yields the table
    above; higher values exaggerate the effect. Unknown moods are neutral.
    """
    base = _MOOD_MODIFIERS.get(mood, NEUTRAL_MODIFIERS)
    if strength == 1.0:
        return base

    # Linear scaling ``1 + strength * (mult - 1)`` drives any multiplier below 1.0
    # negative once ``strength`` exceeds ``1 / (1 - mult)`` (e.g. sleepy's 0.5 base
    # at strength 3.0 yields -0.5). A negative multiplier would put a negative
    # weight into ``random.choices`` (undefined pick) via ``sample_length_mode`` and
    # silently disable unprompted replies via ``reply_probability_mult``. Clamp each
    # scaled multiplier at 0.0 so any strength degrades gracefully — a mode whose
    # weight hits 0 is simply excluded, which is safe because registry validation
    # guarantees at least one positive base length weight. ``randomness_delta`` is a
    # signed additive nudge (already floored via ``max(0.0, ...)`` in
    # ``ResponseGenerator``), so it is left unclamped here.
    def _scale(mult: float) -> float:
        return max(0.0, 1.0 + strength * (mult - 1.0))

    return MoodModifiers(
        reply_probability_mult=_scale(base.reply_probability_mult),
        randomness_delta=strength * base.randomness_delta,
        length_weight_mult=(
            _scale(base.length_weight_mult[0]),
            _scale(base.length_weight_mult[1]),
            _scale(base.length_weight_mult[2]),
        ),
        flavor_strength_mult=_scale(base.flavor_strength_mult),
    )


def message_intensity(text: str) -> float:
    """Emphatic-signal score of one message in ``[0, 1]``.

    Blends the density of ``!``/``?`` (capped, so a wall of ``!!!`` cannot
    dominate forever) with the share of upper-case letters. Empty or
    punctuation-only messages score by their emphasis alone.
    """
    exclaims = text.count("!") + text.count("?")
    punct_signal = min(1.0, exclaims / 3.0)

    letters = [c for c in text if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    else:
        caps_ratio = 0.0

    return min(1.0, 0.5 * punct_signal + 0.5 * caps_ratio)


def classify_mood(
    *,
    rate_ewma: float,
    intensity_ewma: float,
    config: MoodConfig,
) -> str:
    """Map smoothed signals to a mood label.

    Intensity wins first (a heated argument reads as heated even at a modest
    pace); otherwise pace decides between sleepy / lively / calm.
    """
    if intensity_ewma >= config.heated_intensity:
        return HEATED
    if rate_ewma <= config.sleepy_rate_per_min:
        return SLEEPY
    if rate_ewma >= config.lively_rate_per_min:
        return LIVELY
    return CALM


def initial_mood_state(now: float, *, config: MoodConfig) -> ChatMoodState:
    """Seed state for a chat's first observed message.

    The rate EWMA starts at the calm midpoint between the sleepy and lively
    thresholds so a chat does not read as ``sleepy`` purely because it was just
    created.
    """
    baseline_rate = (config.sleepy_rate_per_min + config.lively_rate_per_min) / 2.0
    return ChatMoodState(
        mood=CALM,
        rate_ewma=baseline_rate,
        intensity_ewma=0.0,
        mention_ewma=0.0,
        last_ts=now,
    )


def update_mood_state(
    state: ChatMoodState | None,
    *,
    now: float,
    text: str,
    mentioned: bool,
    config: MoodConfig,
) -> ChatMoodState:
    """Fold one message into the chat's mood state and reclassify.

    ``now`` is a monotonic timestamp (seconds). The instantaneous rate is
    ``60 / dt`` messages per minute, clamped to ``max_rate_per_min`` so a burst
    of near-simultaneous messages cannot spike the EWMA to infinity.
    """
    if state is None:
        return initial_mood_state(now, config=config)

    alpha = config.ewma_alpha
    dt = now - state.last_ts
    if dt <= 0.0:
        inst_rate = config.max_rate_per_min
    else:
        inst_rate = min(config.max_rate_per_min, 60.0 / dt)

    rate_ewma = alpha * inst_rate + (1.0 - alpha) * state.rate_ewma
    intensity_ewma = (
        alpha * message_intensity(text) + (1.0 - alpha) * state.intensity_ewma
    )
    mention_ewma = alpha * (1.0 if mentioned else 0.0) + (1.0 - alpha) * state.mention_ewma

    mood = classify_mood(
        rate_ewma=rate_ewma,
        intensity_ewma=intensity_ewma,
        config=config,
    )
    return ChatMoodState(
        mood=mood,
        rate_ewma=rate_ewma,
        intensity_ewma=intensity_ewma,
        mention_ewma=mention_ewma,
        last_ts=now,
    )
