from __future__ import annotations

import logging
import random
import re
import time
from datetime import UTC, datetime

from aiogram import F, Router
from aiogram.types import Message

from app.config.runtime_state import RuntimeState, get_recent_deque
from app.core.emoji import count_emojis
from app.core.hot_ngrams import extract_content_ngrams
from app.core.markov import (
    MarkovGenerator,
    is_short_generated_reply,
    tokenize,
)
from app.core.mood import (
    NEUTRAL_MODIFIERS,
    ChatMoodState,
    modifiers_for_mood,
    update_mood_state,
)
from app.core.reply_flavor import apply_rare_event, roll_rare_event
from app.core.reply_policy import (
    bot_is_mentioned,
    burst_factor,
    conversation_momentum,
    cooldown_allows_reply,
    effective_reply_probability,
    has_enough_model_data,
    should_reply_to_message,
    within_hourly_cap,
)
from app.core.response_generator import (
    GenerationRequest,
    ResponseGenerator,
    remember_recent_reply,
)
from app.core.text import sanitize_first_name, sanitize_text
from app.handlers._helpers import (
    is_group_message,
    reply_humanized_sequence_state,
    reply_humanized_state,
)
from app.log_masking import mask_chat_id
from app.presentation.fallback_phrases import (
    GENERATION_FAILED_PHRASES,
    NOT_ENOUGH_DATA_PHRASES,
    late_night_pool,
    mood_fallback_pool,
    next_quirk_vocative,
    pick_fallback_phrase,
)
from app.services import LearningService, PivoService

router = Router(name="learning")
logger = logging.getLogger("chat_markov")

MAX_LEARN_MESSAGE_CHARS = 2000
# Нижняя граница длины отдельной константой не задаётся: её обеспечивает
# требование к числу токенов ниже — сообщение короче него всё равно не
# проходит гейт обучаемости.
MIN_LEARN_MESSAGE_TOKENS = 2
RECENT_SHORT_REPLY_LIMIT = 5
RECENT_FALLBACK_LIMIT = 3

# Leading direct-address to the bot: "<alias><separator> ..." or a bare
# "<alias> ...". Users address the bot without a comma at least as often as
# with one ("пепе кто гнойный пидор"), and the unstripped form taught the
# corpus the bot's own name — replies then opened with "пепе ..." (audit
# SIM-8, 2026-07-12). A mid-sentence alias is preserved, and the lookahead
# keeps a bare alias with nothing after it intact.
_LEADING_VOCATIVE_RE = re.compile(r"^\s*([^\s,:;—–-]+)(?:\s*[,:;—–-]+\s*|\s+)(?=\S)")


def strip_leading_bot_vocative(text: str, aliases: frozenset[str]) -> str:
    """Remove a leading "<bot-alias>[<separator>]" direct address, if present."""
    if not aliases:
        return text
    match = _LEADING_VOCATIVE_RE.match(text)
    if match is None:
        return text
    if match.group(1).lower() not in {alias.lower() for alias in aliases}:
        return text
    return text[match.end() :]


def is_learnable_message_length(clean_text: str) -> bool:
    length = len(clean_text)
    return length <= MAX_LEARN_MESSAGE_CHARS


def has_enough_tokens_for_learning(tokens: list[str]) -> bool:
    return len(tokens) >= MIN_LEARN_MESSAGE_TOKENS


def normalize_short_reply_text(text: str) -> str:
    return sanitize_text(text).lower()


def remember_short_reply(
    runtime_state: RuntimeState, chat_id: int, reply_text: str
) -> None:
    recent = get_recent_deque(
        runtime_state.recent_short_replies, chat_id, RECENT_SHORT_REPLY_LIMIT
    )
    recent.append(normalize_short_reply_text(reply_text))


def next_fallback_phrase(
    runtime_state: RuntimeState,
    chat_id: int,
    pool: tuple[str, ...],
    rng: random.Random | None = None,
    now: datetime | None = None,
    mood: str | None = None,
) -> str:
    """Pick a fallback phrase avoiding the ones used recently in this chat.

    Late at night the pool is extended with late-night-flavored phrases (S4);
    a heated chat mood adds punchier phrases (M1). Both are additive, so the
    neutral pool always remains available.
    """
    recent = get_recent_deque(
        runtime_state.recent_fallbacks, chat_id, RECENT_FALLBACK_LIMIT
    )
    effective_pool = mood_fallback_pool(late_night_pool(pool, now), mood)
    phrase = pick_fallback_phrase(effective_pool, recent, rng=rng)
    recent.append(phrase)
    return phrase


def _reply_to_authored_by_bot(reply_to: object, bot_id: int) -> bool:
    """True when the replied-to message was sent by the bot itself."""
    author = getattr(reply_to, "from_user", None)
    return author is not None and getattr(author, "id", None) == bot_id


def extract_context_tokens(
    message: Message,
    current_text: str,
    normalize_lower: bool,
    max_tokens: int,
    only_for_replies: bool,
    include_current_message: bool,
    bot_id: int,
) -> list[str]:
    if only_for_replies and message.reply_to_message is None:
        return []

    context_parts: list[str] = []
    reply_to = message.reply_to_message
    # A reply to the bot's OWN message must not feed the bot's words back as
    # context. The bot's prior output is corpus-derived, so its n-grams match
    # stored start-states; the contextual-start path (reply_context_start_bias)
    # then re-anchors on them and the new reply opens by replaying the previous
    # bot message verbatim — the self-echo seen across consecutive bot messages.
    # The current message (the human line we are actually answering) still
    # anchors the reply via include_current_message.
    if (
        reply_to
        and reply_to.text
        and not _reply_to_authored_by_bot(reply_to, bot_id)
    ):
        context_parts.append(reply_to.text)
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
    pivo_service: PivoService,
) -> None:
    if not is_group_message(message):
        return
    if message.from_user is None:
        return
    if message.from_user.is_bot:
        return

    # Каждое сообщение — свежий снимок профиля отправителя: /pivo упоминает по
    # @username, а он мог смениться после /pivo_on. No-op для неподписанных.
    await pivo_service.refresh_member(message.chat.id, message.from_user)

    raw_text = message.text or ""
    if raw_text.startswith("/"):
        return

    now = time.monotonic()
    runtime_state.note_chat_activity(message.chat.id, now)

    # Mention check before learning validation: a 1-token reply to the bot
    # (e.g. "ок", "?") should still trigger a response even if too short to learn.
    mentioned = bot_is_mentioned(message, bot_username, bot_id, bot_text_aliases)

    # Anti-flood gate: mentions bypass the chat cooldown and the hourly cap by
    # design, so one user could force a generation+reply per message. A gated
    # mention is demoted to the unprompted-reply path (``address_reply=False``);
    # the raw ``mentioned`` flag still feeds mood/rhythm tracking below.
    address_reply = mentioned
    if mentioned and runtime_state.mention_cooldown_sec > 0:
        last_mention_ts = runtime_state.last_mention_reply_ts.get(
            (message.chat.id, message.from_user.id), 0.0
        )
        address_reply = cooldown_allows_reply(
            now, last_mention_ts, runtime_state.mention_cooldown_sec
        )

    # M1/M2: fold this message into the per-chat rhythm state before any early
    # return so even short/non-learnable messages still count toward the
    # conversation cadence. The rhythm EWMAs feed both the M1 mood modulation and
    # the M2 reply director, so they are tracked whenever either is enabled; the
    # mood *behaviour* (modifiers, fallback flavour, transition log) applies only
    # when mood itself is enabled.
    mood: str | None = None
    mood_modifiers = NEUTRAL_MODIFIERS
    chat_rhythm: ChatMoodState | None = None
    if runtime_state.mood_enabled or runtime_state.reply_director_enabled:
        previous_mood = runtime_state.chat_mood.get(message.chat.id)
        chat_rhythm = update_mood_state(
            previous_mood,
            now=now,
            text=raw_text,
            mentioned=mentioned,
            config=runtime_state.mood_config(),
        )
        runtime_state.chat_mood[message.chat.id] = chat_rhythm
        if runtime_state.mood_enabled:
            mood = chat_rhythm.mood
            if previous_mood is None or previous_mood.mood != chat_rhythm.mood:
                logger.debug(
                    "Chat mood -> %s: chat=%s rate=%.1f intensity=%.2f",
                    chat_rhythm.mood,
                    mask_chat_id(message.chat.id),
                    chat_rhythm.rate_ewma,
                    chat_rhythm.intensity_ewma,
                )
            mood_modifiers = modifiers_for_mood(
                chat_rhythm.mood, runtime_state.mood_modulation_strength
            )

    # M3 emoji channel: learn this chat's emoji vocabulary from the raw text
    # (the word model drops emojis). Recorded before the learnability gates so
    # emoji-only reactions — which never pass the word-token threshold — still
    # count toward what the bot can echo later.
    if runtime_state.emoji_append_chance > 0.0:
        emoji_counts = count_emojis(raw_text)
        if emoji_counts:
            await learning_service.record_emojis(message.chat.id, dict(emoji_counts))

    # Strip a leading "<bot-alias>, ..." direct address before learning so the
    # corpus does not absorb boilerplate openings. Mention detection above still
    # sees the original text. The same stripped text feeds both the model tokens
    # and the persisted normalized_text (see record_message below) to keep them
    # consistent.
    learn_source = strip_leading_bot_vocative(raw_text, bot_text_aliases)
    clean = sanitize_text(learn_source)
    tokens = tokenize(clean, normalize_lower=runtime_state.normalize_lower)
    learnable = is_learnable_message_length(clean) and has_enough_tokens_for_learning(tokens)

    if not learnable and not address_reply:
        logger.debug(
            "Skip message by length/tokens: chat=%s len=%s tokens=%s",
            mask_chat_id(message.chat.id),
            len(clean),
            len(tokens),
        )
        return
    token_volume = await learning_service.get_token_volume(message.chat.id)

    enough_data = has_enough_model_data(
        token_volume, runtime_state.min_tokens_for_model
    )
    current_message_normalized = clean.lower()

    async def send_fallback_reply(pool: tuple[str, ...]) -> None:
        await reply_humanized_state(
            message,
            next_fallback_phrase(
                runtime_state,
                message.chat.id,
                pool,
                now=datetime.now(),
                mood=mood,
            ),
            runtime_state,
            per_char=True,
        )

    async def count_answered_mention(user_id: int) -> None:
        # L2: an answered address counts as an interaction, fallback answers
        # included (the user did interact); gated on the knob so a zero
        # chance keeps the mention path write-free (L1 pattern).
        if runtime_state.user_quirk_chance > 0.0:
            await learning_service.record_user_interaction(
                message.chat.id, user_id
            )

    try:
        if address_reply and not enough_data:
            await send_fallback_reply(NOT_ENOUGH_DATA_PHRASES)
            runtime_state.note_mention_reply(message.chat.id, message.from_user.id, now)
            await count_answered_mention(message.from_user.id)
            return

        if not enough_data:
            logger.debug(
                "Skip reply: not enough model data chat=%s volume=%s min=%s",
                mask_chat_id(message.chat.id),
                token_volume,
                runtime_state.min_tokens_for_model,
            )
            return

        has_replied_before = message.chat.id in runtime_state.last_reply_ts
        last_ts = runtime_state.last_reply_ts.get(message.chat.id, 0.0)
        cooldown_ok = cooldown_allows_reply(now, last_ts, runtime_state.min_cooldown_sec)

        # M2: when the director is on, conversation momentum (rate/mention/thread
        # signals from the shared rhythm state) maps into the [min, max] band,
        # then the mood multiplier and the post-reply burst rhythm modulate it and
        # a per-hour cap guards against runaway chats. When off, the legacy flat
        # probability × mood multiplier is used unchanged.
        hourly_cap_ok = True
        if runtime_state.reply_director_enabled and chat_rhythm is not None:
            momentum = conversation_momentum(
                rate_ewma=chat_rhythm.rate_ewma,
                mention_ewma=chat_rhythm.mention_ewma,
                is_reply=message.reply_to_message is not None,
                lively_rate_per_min=runtime_state.mood_lively_rate_per_min,
            )
            # A chat with no reply yet gets the neutral factor (a negative sentinel):
            # last_ts=0.0 against monotonic time would otherwise fake a recent reply
            # right after process start and spuriously boost/suppress.
            seconds_since_reply = now - last_ts if has_replied_before else -1.0
            burst = burst_factor(
                seconds_since_reply=seconds_since_reply,
                boost_window_sec=runtime_state.reply_burst_boost_sec,
                boost_mult=runtime_state.reply_burst_boost_mult,
                suppress_window_sec=runtime_state.reply_burst_suppress_sec,
                suppress_mult=runtime_state.reply_burst_suppress_mult,
            )
            reply_prob = effective_reply_probability(
                base_min=runtime_state.reply_probability_min,
                base_max=runtime_state.reply_probability_max,
                momentum=momentum,
                mood_mult=mood_modifiers.reply_probability_mult,
                burst_mult=burst,
            )
            hourly_cap_ok = within_hourly_cap(
                runtime_state.recent_reply_times.get(message.chat.id, ()),
                now,
                runtime_state.reply_max_per_hour,
            )
        else:
            reply_prob = min(
                1.0,
                runtime_state.reply_probability
                * mood_modifiers.reply_probability_mult,
            )

        should_reply = should_reply_to_message(
            mentioned=address_reply,
            cooldown_ok=cooldown_ok,
            hourly_cap_ok=hourly_cap_ok,
            reply_probability=reply_prob,
            random_value=random.random(),
        )

        if not should_reply:
            logger.debug(
                "Skip by trigger/cooldown: chat=%s mentioned=%s cooldown_ok=%s "
                "cap_ok=%s prob=%.2f",
                mask_chat_id(message.chat.id),
                mentioned,
                cooldown_ok,
                hourly_cap_ok,
                reply_prob,
            )
            return

        context_tokens: list[str] = []
        if runtime_state.use_reply_context:
            only_for_replies = runtime_state.reply_context_only_for_replies
            include_current = runtime_state.reply_context_include_current_message
            # When mentioned directly without a reply, override so the current message
            # itself is used as context instead of generating with no context at all.
            if address_reply and message.reply_to_message is None:
                only_for_replies = False
                include_current = True
            context_tokens = extract_context_tokens(
                message=message,
                current_text=raw_text,
                normalize_lower=runtime_state.normalize_lower,
                max_tokens=runtime_state.reply_context_max_tokens,
                only_for_replies=only_for_replies,
                include_current_message=include_current,
                bot_id=bot_id,
            )

        # Phase 4.1d: context influences generation only through the hidden
        # contextual state (increment C); we no longer derive a literal seed from
        # the reply context, which used to make replies echo the prompt prefix.
        # The explicit seed_tokens API on the generator remains for direct callers.
        if context_tokens:
            logger.debug(
                "Reply context prepared: chat=%s context_tokens=%s",
                mask_chat_id(message.chat.id),
                len(context_tokens),
            )
        response_generator = ResponseGenerator(
            generator=generator,
            learning_service=learning_service,
            runtime_state=runtime_state,
            mood_modifiers=mood_modifiers,
            mood=mood,
        )
        # L1 running jokes: occasionally open an unprompted reply from a
        # phrase the chat has been hot on lately. Mention replies are never
        # seeded — a direct address should answer the person, not the meme.
        seed: list[str] | None = None
        if (
            not address_reply
            and runtime_state.hot_ngram_seed_chance > 0.0
            and random.random() < runtime_state.hot_ngram_seed_chance
        ):
            hot_ngrams = await learning_service.get_hot_ngrams(
                message.chat.id,
                min_count=runtime_state.hot_ngram_min_count,
                recency_share=runtime_state.hot_ngram_recency_share,
            )
            if hot_ngrams:
                # Only the length is logged, never the n-gram text: chat
                # content stays out of logs (project log-masking policy).
                seed = list(random.choice(hot_ngrams))
                logger.debug(
                    "Hot-ngram seed picked: chat=%s ngram_len=%s",
                    mask_chat_id(message.chat.id),
                    len(seed),
                )

        reply_text = await response_generator.generate(
            GenerationRequest(
                chat_id=message.chat.id,
                context_tokens=context_tokens,
                seed=seed,
                current_message_normalized=current_message_normalized,
            ),
            rng=random.Random(),
        )

        if not reply_text:
            if address_reply:
                await send_fallback_reply(GENERATION_FAILED_PHRASES)
                # A mention fallback is always sent; never count it against the cap.
                runtime_state.note_reply_sent(message.chat.id, now, unprompted=False)
                runtime_state.note_mention_reply(
                    message.chat.id, message.from_user.id, now
                )
                await count_answered_mention(message.from_user.id)
            logger.debug(
                "Generation failed: chat=%s mentioned=%s",
                mask_chat_id(message.chat.id),
                mentioned,
            )
            return

        # Mention answers are always sent and never counted against the per-hour
        # cap; only self-initiated (unprompted) replies feed the gate. A mention
        # demoted by the mention-cooldown (address_reply=False) goes through the
        # unprompted path and so does count.
        runtime_state.note_reply_sent(
            message.chat.id, now, unprompted=not address_reply
        )
        if address_reply:
            runtime_state.note_mention_reply(
                message.chat.id, message.from_user.id, now
            )
            await count_answered_mention(message.from_user.id)
        reply_parts = [reply_text]
        # UTC, как и остальные суточные механики (decay, /pivo-квоты): кап
        # сбрасывается в одну и ту же полночь независимо от TZ контейнера.
        today_iso = datetime.now(UTC).date().isoformat()
        # L2 user quirks: a regular's answered address occasionally gets a
        # short vocative as a separate first message. The chance is rolled
        # before the DB read so the common path stays read-free; the reply
        # text itself is untouched, so anti-repeat bookkeeping below keeps
        # working on reply_text by construction. At most one quirk per
        # (chat, user) per UTC day.
        quirked = False
        if (
            address_reply
            and runtime_state.user_quirk_chance > 0.0
            and runtime_state.can_fire_user_quirk(
                message.chat.id, message.from_user.id, today_iso
            )
            and random.random() < runtime_state.user_quirk_chance
        ):
            interactions = await learning_service.get_user_interaction_count(
                message.chat.id, message.from_user.id
            )
            if interactions >= runtime_state.user_quirk_min_interactions:
                # L2.1: part of the quirks address the regular by first name.
                # The name is read live from this update and sanitized; it is
                # never stored (the DB side stays an anonymous HMAC counter)
                # and never logged. An unusable name falls back to the pool.
                first_name: str | None = None
                if (
                    runtime_state.user_quirk_name_share > 0.0
                    and random.random() < runtime_state.user_quirk_name_share
                ):
                    first_name = sanitize_first_name(
                        message.from_user.first_name or ""
                    )
                reply_parts = [
                    next_quirk_vocative(
                        random.Random(), first_name=first_name
                    ),
                    reply_text,
                ]
                runtime_state.note_user_quirk(
                    message.chat.id, message.from_user.id, today_iso
                )
                quirked = True
                # Event kind only: no user id, no count, no vocative text in
                # logs (project log-masking policy).
                logger.debug(
                    "User quirk fired: chat=%s", mask_chat_id(message.chat.id)
                )
        # L3 rare events: a generated reply may break shape (verdict/CAPS/
        # double message) or false-start (filler → typing → real reply).
        # Bounded by a per-chat daily budget; fallback phrases never event.
        # A quirked reply already broke shape — skip the roll and spend no
        # budget (one shape break per reply).
        if not quirked and runtime_state.can_fire_rare_event(
            message.chat.id, today_iso
        ):
            event_kind = roll_rare_event(
                random.Random(),
                event_chance=runtime_state.rare_event_chance,
                false_start_chance=runtime_state.false_start_chance,
            )
            if event_kind is not None:
                event_parts = apply_rare_event(
                    event_kind, reply_text, random.Random()
                )
                if event_parts != [reply_text]:
                    reply_parts = event_parts
                    runtime_state.note_rare_event(message.chat.id, today_iso)
                    logger.debug(
                        "Rare event fired: chat=%s kind=%s parts=%s",
                        mask_chat_id(message.chat.id),
                        event_kind,
                        len(reply_parts),
                    )

        await reply_humanized_sequence_state(
            message, reply_parts, runtime_state, per_char=True
        )
        if is_short_generated_reply(
            tokenize(reply_text, normalize_lower=runtime_state.normalize_lower)
        ):
            remember_short_reply(runtime_state, message.chat.id, reply_text)
        remember_recent_reply(runtime_state, message.chat.id, reply_text)
    finally:
        if learnable:
            learned_token_volume = await learning_service.record_message(
                chat_id=message.chat.id,
                raw_text=learn_source,
                tokens=tokens,
            )
            # L1 running jokes: fold the learned message's content n-grams
            # into the sliding hot-ngram window. Gated on the channel knob so
            # a zero chance keeps the learn path write-free (M3 pattern).
            if runtime_state.hot_ngram_seed_chance > 0.0:
                content_ngrams = extract_content_ngrams(tokens)
                if content_ngrams:
                    await learning_service.record_hot_ngrams(
                        message.chat.id, content_ngrams
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
