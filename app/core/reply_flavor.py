from __future__ import annotations

import random
import re

# Probabilities (at strength 1.0) of the mutually exclusive ending transforms.
# One roll decides which transform applies, so probabilities stay independent
# of each other and easy to reason about. Only the trailing punctuation
# cluster is ever touched; the words themselves are never modified.
DROP_FINAL_PERIOD_PROBABILITY = 0.25
ELLIPSIS_PROBABILITY = 0.07
EXCLAMATION_PROBABILITY = 0.05
DOUBLE_TERMINAL_PROBABILITY = 0.04


def apply_reply_flavor(
    text: str,
    rng: random.Random,
    strength: float = 1.0,
) -> str:
    """Vary the reply ending so the output stops being uniformly "word word.".

    ``strength`` scales all transform probabilities; 0 disables flavoring and
    returns the text unchanged. Applied after candidate selection and
    capitalization, before sending.
    """
    if strength <= 0.0 or not text:
        return text
    scale = min(max(0.0, strength), 2.0)
    stripped = text.rstrip()
    if not stripped:
        return text

    if stripped.endswith(".") and not stripped.endswith(".."):
        roll = rng.random()
        drop_below = DROP_FINAL_PERIOD_PROBABILITY * scale
        ellipsis_below = drop_below + ELLIPSIS_PROBABILITY * scale
        exclaim_below = ellipsis_below + EXCLAMATION_PROBABILITY * scale
        body = stripped[:-1].rstrip()
        if not body:
            return stripped
        if roll < drop_below:
            return body
        if roll < ellipsis_below:
            return body + "..."
        if roll < exclaim_below:
            return body + "!"
        return stripped

    if stripped.endswith(("?", "!")) and not stripped.endswith(("??", "!!")):
        if rng.random() < DOUBLE_TERMINAL_PROBABILITY * scale:
            return stripped + stripped[-1]

    return stripped


# --- L3 rare events & false starts -----------------------------------------
# Surface-only shape breaks for generated replies. The words of the reply are
# only ever uppercased or split, never rewritten; "verdict" replaces the reply
# with a one-word reaction from a fixed pool (rare enough to stay a delight).

RARE_EVENT_KINDS = ("verdict", "caps", "double")

VERDICT_WORDS = (
    "база",
    "жиза",
    "классика",
    "мощно",
    "сильно",
    "именно",
)

FALSE_START_FILLERS = (
    "ну как бы...",
    "щас",
    "эм",
    "короче",
    "погоди",
)

# A sentence boundary for the "double" split: terminal punctuation followed by
# whitespace. Abbreviation dots ("т.е. так") can also match — an odd split is
# acceptable for a ~0.17%-per-reply cosmetic event.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+(?=\S)")


def roll_rare_event(
    rng: random.Random,
    *,
    event_chance: float,
    false_start_chance: float,
) -> str | None:
    """Decide whether this reply becomes a rare event.

    The event roll (verdict/caps/double, uniform among them) is checked first,
    then the false-start roll; both gates are independent knobs and 0 disables
    the respective roll entirely.
    """
    if event_chance > 0.0 and rng.random() < event_chance:
        return rng.choice(RARE_EVENT_KINDS)
    if false_start_chance > 0.0 and rng.random() < false_start_chance:
        return "false_start"
    return None


def apply_rare_event(kind: str, text: str, rng: random.Random) -> list[str]:
    """Map a reply into the message sequence for the rolled event.

    Non-applicable input (no sentence boundary for "double", unknown kind,
    empty text) degrades to the unchanged single-message sequence.
    """
    if not text:
        return [text]
    if kind == "verdict":
        return [rng.choice(VERDICT_WORDS)]
    if kind == "caps":
        return [text.upper()]
    if kind == "double":
        parts = _SENTENCE_BOUNDARY_RE.split(text, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return [parts[0].strip(), parts[1].strip()]
        return [text]
    if kind == "false_start":
        return [rng.choice(FALSE_START_FILLERS), text]
    return [text]
