from __future__ import annotations

from markov import tokenize


BOT_TEXT_ALIASES = {"pepe", "пепе"}


def text_contains_bot_alias(text: str, bot_aliases: set[str]) -> bool:
    for chunk in tokenize(text, normalize_lower=True):
        if chunk in bot_aliases:
            return True
    return False


def is_mention_entity(entity_type: object) -> bool:
    value = getattr(entity_type, "value", entity_type)
    name = getattr(entity_type, "name", "")
    return value == "mention" or name == "MENTION"


def bot_is_mentioned(message: object, bot_username: str, bot_id: int) -> bool:
    reply_to_message = getattr(message, "reply_to_message", None)
    text = getattr(message, "text", None)

    if not text:
        reply_from_user = getattr(reply_to_message, "from_user", None)
        return reply_from_user is not None and getattr(reply_from_user, "id", None) == bot_id

    username_mention = f"@{bot_username}".lower()
    if username_mention in text.lower():
        return True
    if text_contains_bot_alias(text, BOT_TEXT_ALIASES):
        return True

    entities = getattr(message, "entities", None)
    if entities:
        for ent in entities:
            if is_mention_entity(getattr(ent, "type", None)):
                mention = text[ent.offset : ent.offset + ent.length]
                if mention.lower() == username_mention:
                    return True

    reply_from_user = getattr(reply_to_message, "from_user", None)
    if reply_from_user is not None and getattr(reply_from_user, "id", None) == bot_id:
        return True
    return False


def has_enough_model_data(token_volume: int, min_tokens_for_model: int) -> bool:
    return token_volume >= min_tokens_for_model


def cooldown_allows_reply(now_ts: float, last_reply_ts: float, min_cooldown_sec: int) -> bool:
    return now_ts - last_reply_ts >= min_cooldown_sec


def should_reply_to_message(
    *,
    mentioned: bool,
    cooldown_ok: bool,
    reply_probability: float,
    random_value: float,
) -> bool:
    if mentioned:
        return True
    return cooldown_ok and random_value < reply_probability
