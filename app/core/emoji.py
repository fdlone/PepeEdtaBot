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
# blocks rather than the full, ever-growing Unicode emoji grammar (ZWJ sequences,
# skin-tone modifiers, variation selectors): for frequency stats, counting the
# base pictographs is enough, and a simple range match stays dependency-free.
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f700-\U0001f77f"  # alchemical
    "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
    "\U0001fa70-\U0001faff"  # symbols & pictographs extended-A
    "\U00002600-\U000026ff"  # miscellaneous symbols
    "\U00002700-\U000027bf"  # dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flag halves)
    "]",
    flags=re.UNICODE,
)

# Flattening exponent applied to raw counts when sampling (same idea as the
# Markov exploration flattening): < 1 lifts rarer emojis so a single dominant
# meme does not win every time. Fixed in code; the live knob is the append chance.
EMOJI_SAMPLE_POWER = 0.5


def extract_emojis(text: str) -> list[str]:
    """Return every emoji code point in ``text``, in order, with repeats.

    Repeats are kept so the caller can fold them into a frequency Counter and a
    burst of the same emoji counts proportionally.
    """
    if not text:
        return []
    return _EMOJI_RE.findall(text)


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
