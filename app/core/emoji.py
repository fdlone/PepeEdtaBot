"""Emoji extraction and frequency sampling (Stage 3 M3).

The Markov word model deliberately drops emojis, so the bot never speaks the
chat's emoji vocabulary. This module owns the two pure pieces of the emoji
channel: pulling emojis out of raw message text (before tokenization discards
them) and sampling one by learned frequency. Storage lives in
``ChatEmojiStatsRepo`` and the wiring in the learning handler / response
generator, keeping this module side-effect free and unit-testable.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Mapping

# Core emoji code-point ranges. This intentionally covers the common pictographic
# blocks rather than the full, ever-growing Unicode emoji grammar: a range match
# stays dependency-free. On top of the base ranges, *sequences* are matched so
# the channel never stores or echoes a bare fragment: a base pictograph may carry
# variation selectors (U+FE0F) and skin-tone modifiers (U+1F3FB–1F3FF), and ZWJ
# (U+200D) glues bases into composed emojis (🏳️‍🌈, 🏴‍☠️, 👨‍👩‍👧) that are
# extracted whole. Paired regional indicators are folded into whole flags below
# rather than matched singly.
_EMOJI_BASE_PATTERN = (
    "["
    "\U0001f300-\U0001f3fa"  # symbols & pictographs (below skin-tone modifiers)
    "\U0001f400-\U0001f5ff"  # symbols & pictographs (above skin-tone modifiers)
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f700-\U0001f77f"  # alchemical
    "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
    "\U0001fa70-\U0001faff"  # symbols & pictographs extended-A
    "\U00002600-\U000026ff"  # miscellaneous symbols
    "\U00002700-\U000027bf"  # dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flag halves)
    "]"
)
# Modifiers that bind to the preceding base: variation selector + skin tones.
_EMOJI_MODIFIERS_PATTERN = "(?:\ufe0f|[\U0001f3fb-\U0001f3ff])*"
# One full emoji sequence: base(+modifiers), optionally chained with ZWJ (U+200D).
_EMOJI_SEQ_PATTERN = (
    f"(?:{_EMOJI_BASE_PATTERN}{_EMOJI_MODIFIERS_PATTERN}"
    f"(?:\u200d{_EMOJI_BASE_PATTERN}{_EMOJI_MODIFIERS_PATTERN})*)"
)
_EMOJI_SEQ_RE = re.compile(_EMOJI_SEQ_PATTERN, flags=re.UNICODE)

# Regional indicators only carry meaning in pairs (two halves make one flag), so
# a lone half is dropped instead of leaking into stats or a reply.
_REGIONAL_INDICATOR_RE = re.compile("[\U0001f1e6-\U0001f1ff]")

# Trailing run of emojis (plus the spaces/punctuation between and after them),
# used to make anti-repeat normalization ignore an appended emoji flavor.
# The run must contain at least one emoji: a bare punctuation/space tail
# ("привет...") is not an emoji flavor and must survive untouched.
_TRAILING_EMOJI_RE = re.compile(
    r"(?:[\s.!?…]*" + _EMOJI_SEQ_PATTERN + r")+[\s.!?…]*\Z",
    flags=re.UNICODE,
)

# Flattening exponent applied to raw counts when sampling (same idea as the
# Markov exploration flattening): < 1 lifts rarer emojis so a single dominant
# meme does not win every time. Fixed in code; the live knob is the append chance.
EMOJI_SAMPLE_POWER = 0.5


def _is_regional_indicator(ch: str) -> bool:
    return _REGIONAL_INDICATOR_RE.fullmatch(ch) is not None


def extract_emojis(text: str) -> list[str]:
    """Return every emoji in ``text``, in order, with repeats.

    Repeats are kept so the caller can fold them into a frequency Counter and a
    burst of the same emoji counts proportionally. Regional indicators are folded
    back into whole two-letter flags (a lone trailing half is dropped) so the
    channel never stores or echoes a flag fragment.
    """
    if not text:
        return []
    result: list[str] = []
    pending_indicator: str | None = None
    for ch in _EMOJI_SEQ_RE.findall(text):
        if _is_regional_indicator(ch):
            if pending_indicator is None:
                pending_indicator = ch
            else:
                result.append(pending_indicator + ch)
                pending_indicator = None
        else:
            pending_indicator = None
            result.append(ch)
    return result


def strip_trailing_emojis(text: str) -> str:
    """Strip a trailing run of emojis (and the punctuation/space around them).

    Used by anti-repeat normalization so an appended emoji flavor does not make a
    reply look different from the candidate it was built from.
    """
    return _TRAILING_EMOJI_RE.sub("", text)


def count_emojis(text: str) -> Counter[str]:
    """Frequency of each emoji in ``text`` (empty if none)."""
    return Counter(extract_emojis(text))


def sample_emoji(
    stats: Mapping[str, int],
    rng: random.Random,
    *,
    power: float = EMOJI_SAMPLE_POWER,
) -> str | None:
    """Pick one emoji weighted by ``count ** power``; ``None`` if no usable stats.

    Non-positive counts are ignored. Determinism is the caller's via ``rng``.
    """
    population: list[str] = []
    weights: list[float] = []
    for emoji, count in stats.items():
        if count <= 0:
            continue
        population.append(emoji)
        weights.append(float(count) ** power)
    if not population:
        return None
    return rng.choices(population, weights=weights, k=1)[0]


def append_emoji_flavor(
    text: str,
    stats: Mapping[str, int],
    rng: random.Random,
    *,
    chance: float,
    heated: bool = False,
    heated_boost: float = 1.5,
    power: float = EMOJI_SAMPLE_POWER,
) -> str:
    """Maybe append a frequency-sampled emoji to ``text`` (M3).

    Suppressed when the reply ends on a question (``?``) — a trailing emoji reads
    oddly after a genuine question. A ``heated`` mood scales ``chance`` up. The
    roll is consumed only when ``chance`` and ``stats`` make an append possible,
    keeping seeded callers stable when the feature is effectively off.
    """
    if chance <= 0.0 or not text or not stats:
        return text
    if text.rstrip().endswith("?"):
        return text
    effective_chance = min(1.0, chance * (heated_boost if heated else 1.0))
    if rng.random() >= effective_chance:
        return text
    emoji = sample_emoji(stats, rng, power=power)
    if emoji is None:
        return text
    return f"{text} {emoji}"
