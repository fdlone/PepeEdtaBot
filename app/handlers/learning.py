from __future__ import annotations

import logging
import random
import time

from aiogram import F, Router
from aiogram.types import Message

from app.handlers._helpers import is_group_message, reply_humanized
from app.services import LearningService
from bot_policy import (
    bot_is_mentioned,
    cooldown_allows_reply,
    has_enough_model_data,
    should_reply_to_message,
)
from markov import MarkovGenerator, tokenize
from runtime_state import RuntimeState
from text_utils import sanitize_text

router = Router(name="learning")
logger = logging.getLogger("chat_markov")


def extract_context_tokens(
    message: Message,
    current_text: str,
    normalize_lower: bool,
    max_tokens: int,
    only_for_replies: bool,
    include_current_message: bool,
) -> list[str]:
    if only_for_replies and message.reply_to_message is None:
        return []

    context_parts: list[str] = []
    if message.reply_to_message and message.reply_to_message.text:
        context_parts.append(message.reply_to_message.text)
    if include_current_message and current_text:
        context_parts.append(current_text)

    if not context_parts:
        return []

    clean = sanitize_text(" ".join(context_parts))
    if not clean:
        return []

    tokens = tokenize(clean, normalize_lower=normalize_lower)
    return tokens[-max_tokens:] if len(tokens) > max_tokens else tokens


@router.message(F.text)
async def on_text_message(
    message: Message,
    learning_service: LearningService,
    generator: MarkovGenerator,
    state: RuntimeState,
    bot_username: str,
    bot_id: int,
) -> None:
    if not is_group_message(message):
        return
    if message.from_user is None:
        return
    if message.from_user.is_bot:
        return
    raw_text = message.text or ""
    if raw_text.startswith("/"):
        return

    clean = sanitize_text(raw_text)
    if len(clean) < 3 or len(clean) > 500:
        logger.debug(
            "Skip message by length: chat=%s len=%s", message.chat.id, len(clean)
        )
        return

    tokens = tokenize(clean, normalize_lower=state.normalize_lower)
    token_volume = await learning_service.record_message(
        chat_id=message.chat.id,
        raw_text=raw_text,
        tokens=tokens,
    )
    learned = state.learned_messages.get(message.chat.id, 0) + 1
    state.learned_messages[message.chat.id] = learned
    if learned == 1 or learned % 25 == 0:
        logger.info(
            "Прогресс обучения: chat_id=%s, сообщений=%s, объём_модели=%s",
            message.chat.id,
            learned,
            token_volume,
        )

    now = time.time()
    mentioned = bot_is_mentioned(message, bot_username, bot_id)

    enough_data = has_enough_model_data(token_volume, state.min_tokens_for_model)

    if mentioned and not enough_data:
        await reply_humanized(
            message,
            "Пока мало материала, поболтайте ещё 🙂",
            state.typing_min_ms,
            state.typing_max_ms,
        )
        return

    if not enough_data:
        logger.debug(
            "Skip reply: not enough model data chat=%s volume=%s min=%s",
            message.chat.id,
            token_volume,
            state.min_tokens_for_model,
        )
        return

    last_ts = state.last_reply_ts.get(message.chat.id, 0.0)
    cooldown_ok = cooldown_allows_reply(now, last_ts, state.min_cooldown_sec)
    should_reply = should_reply_to_message(
        mentioned=mentioned,
        cooldown_ok=cooldown_ok,
        reply_probability=state.reply_probability,
        random_value=random.random(),
    )

    if not should_reply:
        logger.debug(
            "Skip by trigger/cooldown: chat=%s mentioned=%s cooldown_ok=%s prob=%.2f",
            message.chat.id,
            mentioned,
            cooldown_ok,
            state.reply_probability,
        )
        return

    context_tokens: list[str] = []
    if state.use_reply_context:
        context_tokens = extract_context_tokens(
            message=message,
            current_text=raw_text,
            normalize_lower=state.normalize_lower,
            max_tokens=state.reply_context_max_tokens,
            only_for_replies=state.reply_context_only_for_replies,
            include_current_message=state.reply_context_include_current_message,
        )

    seed = None
    if seed is None and context_tokens:
        seed = context_tokens[-state.reply_context_last_tokens :]
        logger.debug(
            "Reply context prepared: chat=%s context_tokens=%s seed=%s",
            message.chat.id,
            len(context_tokens),
            seed,
        )
    reply_text = ""
    # Повторяем генерацию несколько раз: сначала с контекстом, потом без,
    # и отбрасываем результат если он дословно совпадает с уже виденным сообщением.
    for attempt in range(4):
        attempt_context_tokens = context_tokens if attempt < 2 else None
        candidate = await generator.generate_text(
            chat_id=message.chat.id,
            max_chars=state.max_reply_chars,
            max_tokens=state.max_reply_tokens,
            seed_tokens=seed,
            context_tokens=attempt_context_tokens,
            context_bias=state.reply_context_bias,
            context_start_bias=state.reply_context_start_bias,
            randomness_strength=state.randomness_strength,
            repetition_penalty_strength=state.repetition_penalty_strength,
            markov_order=state.markov_order,
            enable_backoff=state.enable_backoff,
            backoff_min_order=state.backoff_min_order,
        )
        if not candidate:
            logger.debug(
                "Generation attempt failed: chat=%s attempt=%s context=%s seed_len=%s",
                message.chat.id,
                attempt + 1,
                bool(attempt_context_tokens),
                len(seed or []),
            )
        elif await learning_service.is_duplicate(
            message.chat.id, candidate, state.normalize_lower
        ):
            logger.debug(
                "Generated duplicate, retrying: chat=%s attempt=%s",
                message.chat.id,
                attempt + 1,
            )
        else:
            reply_text = candidate
            break

        if attempt == 0 and context_tokens:
            seed = context_tokens[-state.reply_context_last_tokens :]
        else:
            seed = None

    if not reply_text:
        if mentioned:
            await reply_humanized(
                message,
                "Собираю мысли... Напишите ещё пару сообщений 🙂",
                state.typing_min_ms,
                state.typing_max_ms,
            )
            state.last_reply_ts[message.chat.id] = now
        logger.debug(
            "Generation failed: chat=%s mentioned=%s", message.chat.id, mentioned
        )
        return

    state.last_reply_ts[message.chat.id] = now
    await reply_humanized(
        message, reply_text, state.typing_min_ms, state.typing_max_ms
    )
