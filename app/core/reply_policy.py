from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.core.markov import tokenize

# M2 momentum blend weights. Message rate is the dominant driver; being addressed
# (mention EWMA) and active reply threading add on top. Fixed in code — the
# director's live-tunable knobs are the [min, max] band, the burst multipliers, and
# the per-hour cap. They sum to 1.0 so a fully saturated chat scores exactly 1.0.
_MOMENTUM_RATE_WEIGHT = 0.55
_MOMENTUM_MENTION_WEIGHT = 0.30
_MOMENTUM_CHAIN_WEIGHT = 0.15

# Default text aliases the bot answers to when used as plain words in chat
# (i.e. without a Telegram @-mention). Frozen so callers can't mutate it
# accidentally; fully overridable from .env via Settings.bot_text_aliases.
DEFAULT_BOT_TEXT_ALIASES: frozenset[str] = frozenset({"pepe", "пепе"})


def text_contains_bot_alias(text: str, bot_aliases: Iterable[str]) -> bool:
    aliases = {a.lower() for a in bot_aliases}
    for chunk in tokenize(text, normalize_lower=True):
        if chunk in aliases:
            return True
    return False


def is_mention_entity(entity_type: object) -> bool:
    value = getattr(entity_type, "value", entity_type)
    name = getattr(entity_type, "name", "")
    return value == "mention" or name == "MENTION"


def _reply_is_to_bot(reply_to_message: object, bot_id: int) -> bool:
    reply_from_user = getattr(reply_to_message, "from_user", None)
    return reply_from_user is not None and getattr(reply_from_user, "id", None) == bot_id


def bot_is_mentioned(
    message: object,
    bot_username: str,
    bot_id: int,
    bot_text_aliases: Iterable[str] = DEFAULT_BOT_TEXT_ALIASES,
) -> bool:
    reply_to_message = getattr(message, "reply_to_message", None)
    text = getattr(message, "text", None)

    if not text:
        return _reply_is_to_bot(reply_to_message, bot_id)

    username_mention = f"@{bot_username}".lower()
    if username_mention in text.lower():
        return True
    if text_contains_bot_alias(text, bot_text_aliases):
        return True

    entities = getattr(message, "entities", None)
    if entities:
        for ent in entities:
            if is_mention_entity(getattr(ent, "type", None)):
                mention = text[ent.offset : ent.offset + ent.length]
                if mention.lower() == username_mention:
                    return True

    return _reply_is_to_bot(reply_to_message, bot_id)


def has_enough_model_data(token_volume: int, min_tokens_for_model: int) -> bool:
    return token_volume >= min_tokens_for_model


def cooldown_allows_reply(
    now_ts: float, last_reply_ts: float, min_cooldown_sec: int
) -> bool:
    return now_ts - last_reply_ts >= min_cooldown_sec


def should_reply_to_message(
    *,
    mentioned: bool,
    cooldown_ok: bool,
    hourly_cap_ok: bool,
    reply_probability: float,
    random_value: float,
) -> bool:
    if mentioned:
        return True
    return cooldown_ok and hourly_cap_ok and random_value < reply_probability


def conversation_momentum(
    *,
    rate_ewma: float,
    mention_ewma: float,
    is_reply: bool,
    lively_rate_per_min: float,
) -> float:
    """Blend cheap conversation signals into a momentum score in ``[0, 1]`` (M2).

    ``rate_ewma`` is normalised against the ``lively`` rate threshold (a chat at
    or above lively pace saturates the rate term); ``mention_ewma`` is already a
    smoothed ``[0, 1]`` share of bot-addressing; ``is_reply`` approximates active
    thread depth with the cheap immediate-reply flag (true thread walking would
    cost a DB lookup per message). The weighted blend is the base the director
    maps into its probability band.
    """
    if lively_rate_per_min > 0.0:
        rate_norm = min(1.0, max(0.0, rate_ewma / lively_rate_per_min))
    else:
        rate_norm = 1.0 if rate_ewma > 0.0 else 0.0
    mention_norm = min(1.0, max(0.0, mention_ewma))
    chain = 1.0 if is_reply else 0.0
    score = (
        _MOMENTUM_RATE_WEIGHT * rate_norm
        + _MOMENTUM_MENTION_WEIGHT * mention_norm
        + _MOMENTUM_CHAIN_WEIGHT * chain
    )
    return min(1.0, max(0.0, score))


def burst_factor(
    *,
    seconds_since_reply: float,
    boost_window_sec: float,
    boost_mult: float,
    suppress_window_sec: float,
    suppress_mult: float,
) -> float:
    """Multiplier realising the join-then-withdraw rhythm after a reply (M2).

    For ``boost_window_sec`` after the last reply the probability is boosted
    (the bot stays in the exchange), then for the following
    ``suppress_window_sec`` it is damped (the bot backs off), returning to
    neutral afterwards. A chat with no prior reply (a huge or negative
    ``seconds_since_reply``) is neutral.
    """
    if seconds_since_reply < 0.0:
        return 1.0
    if seconds_since_reply <= boost_window_sec:
        return boost_mult
    if seconds_since_reply <= boost_window_sec + suppress_window_sec:
        return suppress_mult
    return 1.0


def effective_reply_probability(
    *,
    base_min: float,
    base_max: float,
    momentum: float,
    mood_mult: float,
    burst_mult: float,
) -> float:
    """Map momentum into the ``[base_min, base_max]`` band, then modulate (M2).

    The mood multiplier (M1) and the burst multiplier are applied on top of the
    momentum-derived base and the result is clamped to ``[0, 1]``.
    """
    base = base_min + momentum * (base_max - base_min)
    return min(1.0, max(0.0, base * mood_mult * burst_mult))


def within_hourly_cap(
    reply_times: Sequence[float],
    now: float,
    max_per_hour: int,
    *,
    window_sec: float = 3600.0,
) -> bool:
    """True if fewer than ``max_per_hour`` replies fall inside the last hour (M2).

    ``max_per_hour <= 0`` disables the cap. ``reply_times`` are monotonic-second
    timestamps of recent bot replies (kept trimmed by ``note_reply_sent``).
    """
    if max_per_hour <= 0:
        return True
    recent = sum(1 for ts in reply_times if now - ts < window_sec)
    return recent < max_per_hour
