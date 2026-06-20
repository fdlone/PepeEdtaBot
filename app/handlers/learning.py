from __future__ import annotations

import logging
import random
import time
from collections import deque

from aiogram import F, Router
from aiogram.types import Message

from app.handlers._helpers import is_group_message, reply_humanized
from app.log_masking import mask_chat_id
from app.services import LearningService
from bot_policy import (
    bot_is_mentioned,
    cooldown_allows_reply,
    has_enough_model_data,
    should_reply_to_message,
)
from markov import (
    MarkovGenerator,
    escalated_randomness_strength,
    extract_best_seed,
    is_short_generated_reply,
    tokenize,
)
from runtime_state import RuntimeState
from text_utils import sanitize_text

router = Router(name="learning")
logger = logging.getLogger("chat_markov")

MIN_LEARN_MESSAGE_CHARS = 3
MAX_LEARN_MESSAGE_CHARS = 500
MIN_LEARN_MESSAGE_TOKENS = 2
# Сколько раз подряд пробуем сгенерировать ответ, прежде чем сдаёмся.
# Первые попытки идут с reply-контекстом, последние — без.
MAX_GENERATION_ATTEMPTS = 4
GENERATION_ATTEMPTS_WITH_CONTEXT = 5
MAX_TRAINING_PREFIX_RETRY_ATTEMPTS = 6
RECENT_SHORT_REPLY_LIMIT = 5


def is_learnable_message_length(clean_text: str) -> bool:
    length = len(clean_text)
    return length <= MAX_LEARN_MESSAGE_CHARS


def has_enough_tokens_for_learning(tokens: list[str]) -> bool:
    return len(tokens) >= MIN_LEARN_MESSAGE_TOKENS


def normalize_short_reply_text(text: str) -> str:
    return sanitize_text(text).lower()


def was_recent_short_reply(
    runtime_state: RuntimeState, chat_id: int, candidate: str
) -> bool:
    recent = runtime_state.recent_short_replies.get(chat_id)
    if not recent:
        return False
    return normalize_short_reply_text(candidate) in recent


def remember_short_reply(
    runtime_state: RuntimeState, chat_id: int, reply_text: str
) -> None:
    recent = runtime_state.recent_short_replies.get(chat_id)
    if recent is None:
        recent = deque(maxlen=RECENT_SHORT_REPLY_LIMIT)
        runtime_state.recent_short_replies[chat_id] = recent
    recent.append(normalize_short_reply_text(reply_text))


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
    runtime_state: RuntimeState,
    bot_username: str,
    bot_id: int,
    bot_text_aliases: frozenset[str],
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

    now = time.monotonic()
    runtime_state.note_chat_activity(message.chat.id, now)

    # Mention check before learning validation: a 1-token reply to the bot
    # (e.g. "ок", "?") should still trigger a response even if too short to learn.
    mentioned = bot_is_mentioned(message, bot_username, bot_id, bot_text_aliases)

    clean = sanitize_text(raw_text)
    tokens = tokenize(clean, normalize_lower=runtime_state.normalize_lower)
    learnable = is_learnable_message_length(clean) and has_enough_tokens_for_learning(tokens)

    if not learnable:
        if not mentioned:
            logger.debug(
                "Skip message by length/tokens: chat=%s len=%s tokens=%s",
                mask_chat_id(message.chat.id),
                len(clean),
                len(tokens),
            )
            return
        token_volume = await learning_service.get_token_volume(message.chat.id)
    else:
        token_volume = await learning_service.get_token_volume(message.chat.id)

    enough_data = has_enough_model_data(
        token_volume, runtime_state.min_tokens_for_model
    )
    current_message_normalized = clean.lower()

    try:
        if mentioned and not enough_data:
            await reply_humanized(
                message,
                "Пока мало материала, поболтайте ещё 🙂",
                runtime_state.typing_min_ms,
                runtime_state.typing_max_ms,
            )
            return

        if not enough_data:
            logger.debug(
                "Skip reply: not enough model data chat=%s volume=%s min=%s",
                mask_chat_id(message.chat.id),
                token_volume,
                runtime_state.min_tokens_for_model,
            )
            return

        last_ts = runtime_state.last_reply_ts.get(message.chat.id, 0.0)
        cooldown_ok = cooldown_allows_reply(now, last_ts, runtime_state.min_cooldown_sec)
        should_reply = should_reply_to_message(
            mentioned=mentioned,
            cooldown_ok=cooldown_ok,
            reply_probability=runtime_state.reply_probability,
            random_value=random.random(),
        )

        if not should_reply:
            logger.debug(
                "Skip by trigger/cooldown: chat=%s mentioned=%s cooldown_ok=%s prob=%.2f",
                mask_chat_id(message.chat.id),
                mentioned,
                cooldown_ok,
                runtime_state.reply_probability,
            )
            return

        context_tokens: list[str] = []
        if runtime_state.use_reply_context:
            only_for_replies = runtime_state.reply_context_only_for_replies
            include_current = runtime_state.reply_context_include_current_message
            # When mentioned directly without a reply, override so the current message
            # itself is used as context instead of generating with no context at all.
            if mentioned and message.reply_to_message is None:
                only_for_replies = False
                include_current = True
            context_tokens = extract_context_tokens(
                message=message,
                current_text=raw_text,
                normalize_lower=runtime_state.normalize_lower,
                max_tokens=runtime_state.reply_context_max_tokens,
                only_for_replies=only_for_replies,
                include_current_message=include_current,
            )

        seed = None
        if context_tokens:
            seed = extract_best_seed(context_tokens, runtime_state.reply_context_last_tokens)
            logger.debug(
                "Reply context prepared: chat=%s context_tokens=%s seed=%s",
                mask_chat_id(message.chat.id),
                len(context_tokens),
                seed,
            )
        reply_text = ""
        # Повторяем генерацию несколько раз: сначала с контекстом, потом без,
        # и отбрасываем результат, чьё начало совпадает с уже виденным сообщением.
        total_generation_attempts = (
            MAX_GENERATION_ATTEMPTS + MAX_TRAINING_PREFIX_RETRY_ATTEMPTS
        )
        for attempt in range(total_generation_attempts):
            attempt_context_tokens = (
                context_tokens if attempt < GENERATION_ATTEMPTS_WITH_CONTEXT else None
            )
            attempt_randomness_strength = escalated_randomness_strength(
                runtime_state.randomness_strength,
                attempt,
                total_generation_attempts,
            )
            candidate = await generator.generate_text(
                chat_id=message.chat.id,
                max_chars=runtime_state.max_reply_chars,
                max_tokens=runtime_state.max_reply_tokens,
                seed_tokens=seed,
                context_tokens=attempt_context_tokens,
                context_bias=runtime_state.reply_context_bias,
                context_start_bias=runtime_state.reply_context_start_bias,
                randomness_strength=attempt_randomness_strength,
                repetition_penalty_strength=runtime_state.repetition_penalty_strength,
                markov_order=runtime_state.markov_order,
                enable_backoff=runtime_state.enable_backoff,
                backoff_min_order=runtime_state.backoff_min_order,
            )
            if not candidate:
                logger.debug(
                    "Generation attempt failed: chat=%s attempt=%s context=%s seed_len=%s",
                    mask_chat_id(message.chat.id),
                    attempt + 1,
                    bool(attempt_context_tokens),
                    len(seed or []),
                )
            else:
                candidate_normalized = sanitize_text(candidate).lower()
                candidate_tokens = tokenize(
                    candidate, normalize_lower=runtime_state.normalize_lower
                )
                is_short_candidate = is_short_generated_reply(candidate_tokens)
                if candidate_normalized == current_message_normalized:
                    logger.debug(
                        "Generated text copied the current message, retrying: "
                        "chat=%s attempt=%s",
                        mask_chat_id(message.chat.id),
                        attempt + 1,
                    )
                elif is_short_candidate and was_recent_short_reply(
                    runtime_state, message.chat.id, candidate
                ):
                    logger.debug(
                        "Generated short reply was used recently, retrying: "
                        "chat=%s attempt=%s",
                        mask_chat_id(message.chat.id),
                        attempt + 1,
                    )
                elif not is_short_candidate and await learning_service.is_verbatim_copy(
                    message.chat.id, candidate
                ):
                    logger.debug(
                        "Generated text starts like a training sample, retrying: "
                        "chat=%s attempt=%s",
                        mask_chat_id(message.chat.id),
                        attempt + 1,
                    )
                else:
                    logger.debug(
                        "Reply generated: chat=%s attempt=%s tokens=%s context=%s",
                        mask_chat_id(message.chat.id),
                        attempt + 1,
                        len(candidate.split()),
                        bool(attempt_context_tokens),
                    )
                    reply_text = candidate
                    break

            seed = None

        if not reply_text:
            if mentioned:
                await reply_humanized(
                    message,
                    "Собираю мысли... Напишите ещё пару сообщений 🙂",
                    runtime_state.typing_min_ms,
                    runtime_state.typing_max_ms,
                )
                runtime_state.last_reply_ts[message.chat.id] = now
            logger.debug(
                "Generation failed: chat=%s mentioned=%s",
                mask_chat_id(message.chat.id),
                mentioned,
            )
            return

        runtime_state.last_reply_ts[message.chat.id] = now
        await reply_humanized(
            message,
            reply_text,
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
        )
        if is_short_generated_reply(
            tokenize(reply_text, normalize_lower=runtime_state.normalize_lower)
        ):
            remember_short_reply(runtime_state, message.chat.id, reply_text)
    finally:
        if learnable:
            learned_token_volume = await learning_service.record_message(
                chat_id=message.chat.id,
                raw_text=raw_text,
                tokens=tokens,
            )
            learned = runtime_state.learned_messages.get(message.chat.id, 0) + 1
            runtime_state.learned_messages[message.chat.id] = learned
            if learned == 1 or learned % 25 == 0:
                logger.info(
                    "Прогресс обучения: chat=%s, сообщений=%s, объём_модели=%s",
                    mask_chat_id(message.chat.id),
                    learned,
                    learned_token_volume,
                )
