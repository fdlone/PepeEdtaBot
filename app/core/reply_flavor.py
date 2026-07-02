from __future__ import annotations

import random

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
