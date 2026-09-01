"""Конвейер ответа на обычное сообщение чата.

До этого весь конвейер жил одной функцией ``on_text_message`` на 430 строк:
ритм и настроение чата, эмодзи-канал, гейты обучаемости, политика ответа,
затравка из «горячих» n-грамм, генерация, причуды завсегдатаев, редкие
события, отправка и обучение. Любое изменение политики требовало читать всю
функцию, потому что решение собиралось из переменных, объявленных на сотню
строк выше, а отдельные ветки проверялись только полным прогоном обработчика
с моками Telegram.

Здесь тот же конвейер разложен на три шага, которые обработчик вызывает по
очереди: ``observe`` (что за сообщение и стоит ли им заниматься),
``respond`` (отвечать ли и чем) и ``learn`` (запись в модель). Типов Telegram
модуль не знает: обработчик передаёт факты (``IncomingMessage``) и функцию
отправки, поэтому ветки политики проверяются без моков Telegram.

Порядок обращений к модулю ``random`` сохранён дословно — от него зависит
вывод при фиксированном зерне.
"""
from __future__ import annotations

import logging
import random
import re
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from app.config.runtime_state import ReplySlot, RuntimeState
from app.core.emoji import count_emojis
from app.core.generation_telemetry import UserQuirkGate
from app.core.hot_ngrams import extract_content_ngrams
from app.core.markov import (
    MarkovGenerator,
    is_short_generated_reply,
    tokenize,
)
from app.core.mood import (
    NEUTRAL_MODIFIERS,
    ChatMoodState,
    MoodModifiers,
    modifiers_for_mood,
    update_mood_state,
)
from app.core.reply_flavor import apply_rare_event, roll_rare_event
from app.core.reply_policy import (
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
from app.log_masking import mask_chat_id
from app.presentation.fallback_phrases import (
    GENERATION_FAILED_PHRASES,
    NOT_ENOUGH_DATA_PHRASES,
    late_night_pool,
    mood_fallback_pool,
    next_quirk_vocative,
    pick_fallback_phrase,
)
from app.services.learning_service import LearningService, MaintenanceAlert
from app.services.meme_analyzer import MemeSettings

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

# Отправка готовых частей ответа. Первая часть уходит ответом на сообщение,
# остальные — обычными сообщениями; конвейеру важно только «отправить».
# Возвращает число доставленных частей. Ответ бота бывает многочастным
# (фальстарт, двойное сообщение, причуда завсегдатая), и сбой на второй части
# оставляет первую в чате. Без этого числа откат бюджета трактовал бы
# частичную доставку как полное отсутствие ответа — и снимал кулдаун за
# ответ, который люди уже видят.
Sender = Callable[[Sequence[str]], Awaitable[int]]


class PartialDeliveryError(Exception):
    """Отправка оборвалась, часть сообщений уже в чате.

    Живёт здесь, а не в обработчике, потому что это часть контракта
    ``Sender``: контракт объявляет сервис, реализует обработчик, и зависимость
    идёт снизу вверх, как и весь остальной слой (см. ARCHITECTURE.md).

    Несёт число доставленных частей: оно решает судьбу бюджета ответа. Ответ,
    от которого в чате осталась хотя бы одна часть, несостоявшимся не
    считается, и возвращать за него кулдаун нельзя.
    """

    def __init__(self, delivered: int) -> None:
        super().__init__(f"delivered {delivered} part(s) before failing")
        self.delivered = delivered


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
    recent = runtime_state.recent_short_replies.setdefault(
        chat_id, deque(maxlen=RECENT_SHORT_REPLY_LIMIT)
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
    recent = runtime_state.recent_fallbacks.setdefault(
        chat_id, deque(maxlen=RECENT_FALLBACK_LIMIT)
    )
    effective_pool = mood_fallback_pool(late_night_pool(pool, now), mood)
    phrase = pick_fallback_phrase(effective_pool, recent, rng=rng)
    recent.append(phrase)
    return phrase


def extract_context_tokens(
    *,
    current_text: str,
    reply_context_text: str | None,
    is_reply: bool,
    normalize_lower: bool,
    max_tokens: int,
    only_for_replies: bool,
    include_current_message: bool,
) -> list[str]:
    """Токены контекста ответа.

    ``reply_context_text`` — текст сообщения, на которое отвечают, и только
    если его автор не бот: ответ на СВОЁ сообщение не должен возвращать слова
    бота в контекст. Его вывод построен по корпусу, поэтому n-граммы совпадают
    с сохранёнными стартовыми состояниями, и контекстный старт заново
    заякоривается на них — новый ответ открывается дословным повтором
    предыдущего (само-эхо в подряд идущих сообщениях бота). Текущее сообщение
    (реплика человека, на которую отвечаем) по-прежнему якорит ответ через
    ``include_current_message``.
    """
    if only_for_replies and not is_reply:
        return []

    context_parts: list[str] = []
    if reply_context_text:
        context_parts.append(reply_context_text)
    if include_current_message and current_text:
        context_parts.append(current_text)

    if not context_parts:
        return []

    clean = sanitize_text(" ".join(context_parts))
    if not clean:
        return []

    tokens = tokenize(clean, normalize_lower=normalize_lower)
    return tokens[-max_tokens:] if len(tokens) > max_tokens else tokens


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """Факты о сообщении — всё, что конвейеру нужно от Telegram."""

    chat_id: int
    user_id: int
    first_name: str
    text: str
    mentioned: bool
    is_reply: bool
    # Текст сообщения, на которое отвечают, — только если его автор не бот.
    reply_context_text: str | None
    bot_aliases: frozenset[str]
    monotonic_now: float


@dataclass(frozen=True, slots=True)
class MessageObservation:
    """Что конвейер выяснил о сообщении до решения об ответе."""

    address_reply: bool
    mood: str | None
    mood_modifiers: MoodModifiers
    chat_rhythm: ChatMoodState | None
    learn_source: str
    tokens: list[str]
    learnable: bool
    token_volume: int
    enough_data: bool
    current_message_normalized: str


class ReplyPipeline:
    """Ритм чата, политика ответа, генерация и обучение на одном сообщении."""

    def __init__(
        self,
        *,
        learning_service: LearningService,
        generator: MarkovGenerator,
        runtime_state: RuntimeState,
    ) -> None:
        self._learning_service = learning_service
        self._generator = generator
        self._runtime_state = runtime_state

    async def observe(self, msg: IncomingMessage) -> MessageObservation | None:
        """Ритм, настроение, эмодзи-канал и гейты обучаемости.

        ``None`` означает «сообщением заниматься не нужно»: оно не годится ни
        для обучения, ни для ответа. Всё, что копится по каждому сообщению
        (активность чата, ритм, эмодзи), к этому моменту уже записано.
        """
        state = self._runtime_state
        now = msg.monotonic_now
        state.note_chat_activity(msg.chat_id, now)

        # Anti-flood gate: mentions bypass the chat cooldown and the hourly cap
        # by design, so one user could force a generation+reply per message. A
        # gated mention is demoted to the unprompted-reply path
        # (``address_reply=False``); the raw ``mentioned`` flag still feeds
        # mood/rhythm tracking below.
        # Знаменатель обращений — по сырому `msg.mentioned`, ДО гейта
        # `mention_cooldown_sec` ниже: упоминание, погашенное кулдауном, тоже
        # нагружает путь, у которого нет ни кулдауна между ответами, ни
        # часового предела (O15). Первая редакция считала уже отфильтрованный
        # `address_reply`, и доля ответов выходила тождественно равной 100%.
        if msg.mentioned:
            self._generator.telemetry.note_mention_seen()
        address_reply = msg.mentioned
        if msg.mentioned and state.mention_cooldown_sec > 0:
            last_mention_ts = state.last_mention_reply_ts.get(
                (msg.chat_id, msg.user_id), 0.0
            )
            address_reply = cooldown_allows_reply(
                now, last_mention_ts, state.mention_cooldown_sec
            )

        # M1/M2: fold this message into the per-chat rhythm state before any
        # early return so even short/non-learnable messages still count toward
        # the conversation cadence. The rhythm EWMAs feed both the M1 mood
        # modulation and the M2 reply director, so they are tracked whenever
        # either is enabled; the mood *behaviour* (modifiers, fallback flavour,
        # transition log) applies only when mood itself is enabled.
        mood: str | None = None
        mood_modifiers = NEUTRAL_MODIFIERS
        chat_rhythm: ChatMoodState | None = None
        if state.mood_enabled or state.reply_director_enabled:
            previous_mood = state.chat_mood.get(msg.chat_id)
            chat_rhythm = update_mood_state(
                previous_mood,
                now=now,
                text=msg.text,
                mentioned=msg.mentioned,
                config=state.mood_config(),
            )
            state.chat_mood[msg.chat_id] = chat_rhythm
            # Темп наблюдается здесь — до любого раннего возврата: короткие и
            # необучаемые сообщения тоже задают темп чата, и распределение,
            # собранное только по «хорошим» сообщениям, отвечало бы не на тот
            # вопрос (E2-1).
            self._generator.telemetry.note_chat_tempo(
                chat_rhythm.rate_ewma,
                lively_at=state.mood_lively_rate_per_min,
            )
            if state.mood_enabled:
                mood = chat_rhythm.mood
                if previous_mood is None or previous_mood.mood != chat_rhythm.mood:
                    logger.debug(
                        "Chat mood -> %s: chat=%s rate=%.1f intensity=%.2f",
                        chat_rhythm.mood,
                        mask_chat_id(msg.chat_id),
                        chat_rhythm.rate_ewma,
                        chat_rhythm.intensity_ewma,
                    )
                mood_modifiers = modifiers_for_mood(
                    chat_rhythm.mood, state.mood_modulation_strength
                )

        # M3 emoji channel: learn this chat's emoji vocabulary from the raw text
        # (the word model drops emojis). Recorded before the learnability gates
        # so emoji-only reactions — which never pass the word-token threshold —
        # still count toward what the bot can echo later.
        if state.emoji_append_chance > 0.0:
            emoji_counts = count_emojis(msg.text)
            if emoji_counts:
                await self._learning_service.record_emojis(
                    msg.chat_id, dict(emoji_counts)
                )

        # Strip a leading "<bot-alias>, ..." direct address before learning so
        # the corpus does not absorb boilerplate openings. Mention detection
        # still sees the original text. The same stripped text feeds both the
        # model tokens and the persisted normalized_text (see record_message)
        # to keep them consistent.
        learn_source = strip_leading_bot_vocative(msg.text, msg.bot_aliases)
        clean = sanitize_text(learn_source)
        tokens = tokenize(clean, normalize_lower=state.normalize_lower)
        learnable = is_learnable_message_length(
            clean
        ) and has_enough_tokens_for_learning(tokens)

        if not learnable and not address_reply:
            logger.debug(
                "Skip message by length/tokens: chat=%s len=%s tokens=%s",
                mask_chat_id(msg.chat_id),
                len(clean),
                len(tokens),
            )
            return None

        token_volume = await self._learning_service.get_token_volume(msg.chat_id)
        return MessageObservation(
            address_reply=address_reply,
            mood=mood,
            mood_modifiers=mood_modifiers,
            chat_rhythm=chat_rhythm,
            learn_source=learn_source,
            tokens=tokens,
            learnable=learnable,
            token_volume=token_volume,
            enough_data=has_enough_model_data(
                token_volume, state.min_tokens_for_model
            ),
            current_message_normalized=clean.lower(),
        )

    async def respond(
        self, msg: IncomingMessage, obs: MessageObservation, send: Sender
    ) -> None:
        """Решает, отвечать ли, и отправляет ответ через ``send``."""
        state = self._runtime_state
        now = msg.monotonic_now

        if obs.address_reply and not obs.enough_data:
            # Оба окна занимаются ДО отправки, а не после: между ними лежит
            # `await`, и пачка обращений успевала пройти обе проверки, пока
            # запись стояла ниже (O8). Ответ отправлен, поэтому кулдаун и
            # берст его видят; unprompted=False — часовой кап нет.
            state.note_reply_sent(msg.chat_id, now, unprompted=False)
            state.note_mention_reply(msg.chat_id, msg.user_id, now)
            # Фолбэк — тоже ответ на обращение и тоже нагрузка на путь без
            # предела: без этой строки числитель терял целую ветку, а доля
            # ответов занижалась ровно на неё (O15).
            self._generator.telemetry.note_mention_answered(at=now)
            await self._send_fallback(msg, obs, NOT_ENOUGH_DATA_PHRASES, send)
            await self._count_answered_mention(msg.chat_id, msg.user_id)
            return

        if not obs.enough_data:
            logger.debug(
                "Skip reply: not enough model data chat=%s volume=%s min=%s",
                mask_chat_id(msg.chat_id),
                obs.token_volume,
                state.min_tokens_for_model,
            )
            return

        reply_prob, cooldown_ok, hourly_cap_ok = self._reply_odds(msg, obs)
        should_reply = should_reply_to_message(
            mentioned=obs.address_reply,
            cooldown_ok=cooldown_ok,
            hourly_cap_ok=hourly_cap_ok,
            reply_probability=reply_prob,
            random_value=random.random(),
        )

        if not should_reply:
            logger.debug(
                "Skip by trigger/cooldown: chat=%s mentioned=%s cooldown_ok=%s "
                "cap_ok=%s prob=%.2f",
                mask_chat_id(msg.chat_id),
                msg.mentioned,
                cooldown_ok,
                hourly_cap_ok,
                reply_prob,
            )
            return

        # Бюджет занимается здесь — в момент, когда решение принято, а не
        # после отправки. Апдейты обрабатываются конкурентно, и пока запись
        # стояла ниже генерации, пачка сообщений успевала пройти все проверки
        # до первой записи: кулдаун и часовой кап не держали ровно в
        # burst-режиме, ради которого написаны (O8). Резервация откатывается
        # ниже, если ответ не состоялся.
        #
        # Упоминания не считаются в часовой кап и здесь: тот же признак
        # `unprompted`, что и прежде. Упоминание, пониженное кулдауном
        # обращений (address_reply=False), идёт по самостоятельному пути и
        # потому считается.
        # E2-2: у пути обращений нет ни кулдауна, ни часового предела, поэтому
        # собственный счётчик — единственное место, где его нагрузка вообще
        # видна. Знаменатель растёт и у обращения без ответа (см. ранние
        # возвраты выше), числитель — здесь, где ответ решён.
        if obs.address_reply:
            self._generator.telemetry.note_mention_answered(at=now)
        # E2-1: в какой фазе берст-ритма сыгран ответ. Устойчивый ноль в
        # подавлении при ненулевом усилении и будет доказательством, что фаза
        # отхода недостижима, — фактом, а не выводом из формулы.
        #
        # Ответы на обращения исключены: они отправляются в обход вероятности
        # (`should_reply_to_message`: `if mentioned: return True`), то есть
        # берст-фактор на них не действовал вообще. Считать их значило бы
        # набрать ненулевое подавление из данных, к фазе отношения не имеющих,
        # и закрыть O14 в неверную сторону.
        #
        # Фаза определяется самим `burst_factor`, а не переписанными рядом
        # границами: рукописная копия логики возле оригинала — тот приём,
        # который уже стоил вердикта в C2-1.
        if state.reply_director_enabled and not obs.address_reply:
            multiplier = burst_factor(
                seconds_since_reply=now
                - state.last_reply_ts.get(msg.chat_id, float("-inf")),
                boost_window_sec=state.reply_burst_boost_sec,
                boost_mult=state.reply_burst_boost_mult,
                suppress_window_sec=state.reply_burst_suppress_sec,
                suppress_mult=state.reply_burst_suppress_mult,
            )
            if multiplier > 1.0:
                self._generator.telemetry.note_burst_phase(suppressing=False)
            elif multiplier < 1.0:
                self._generator.telemetry.note_burst_phase(suppressing=True)
        slot = state.reserve_reply_slot(
            msg.chat_id, now, unprompted=not obs.address_reply
        )
        try:
            await self._respond_reserved(msg, obs, send, slot)
        except PartialDeliveryError:
            # Часть ответа в чате есть — бюджет остаётся потраченным. Возврат
            # снял бы кулдаун за ответ, который люди уже видят, и следующее
            # сообщение получило бы второй. Исключение поднимается только при
            # ненулевой доставке; «не ушло ничего» приходит сюда обычным
            # исключением и обрабатывается веткой ниже.
            raise
        except Exception:
            state.release_reply_slot(slot)
            raise

    async def _respond_reserved(
        self,
        msg: IncomingMessage,
        obs: MessageObservation,
        send: Sender,
        slot: ReplySlot,
    ) -> None:
        """Подготовка и отправка ответа под уже занятый бюджет.

        Выделено из ``respond`` не ради длины, а ради границы: всё, что здесь
        падает или возвращается без ответа, обязано вернуть бюджет, и одна
        точка входа делает это проверяемым.
        """
        state = self._runtime_state
        now = msg.monotonic_now

        context_tokens = self._context_tokens(msg, obs)
        # Phase 4.1d: context influences generation only through the hidden
        # contextual state (increment C); we no longer derive a literal seed
        # from the reply context, which used to make replies echo the prompt
        # prefix. The explicit seed_tokens API on the generator remains for
        # direct callers.
        if context_tokens:
            logger.debug(
                "Reply context prepared: chat=%s context_tokens=%s",
                mask_chat_id(msg.chat_id),
                len(context_tokens),
            )
        response_generator = ResponseGenerator(
            generator=self._generator,
            learning_service=self._learning_service,
            runtime_state=state,
            mood_modifiers=obs.mood_modifiers,
            mood=obs.mood,
        )
        seed = await self._hot_ngram_seed(msg, obs)

        reply_text = await response_generator.generate(
            GenerationRequest(
                chat_id=msg.chat_id,
                context_tokens=context_tokens,
                seed=seed,
                current_message_normalized=obs.current_message_normalized,
            ),
            rng=random.Random(),
        )

        if not reply_text:
            if obs.address_reply:
                await self._send_fallback(
                    msg, obs, GENERATION_FAILED_PHRASES, send
                )
                # Фолбэк на упоминание отправлен — бюджет остаётся занятым:
                # ответ в чате был. Резервация уже сделана с
                # unprompted=False, то есть в часовой кап не попала, как и
                # прежде.
                state.note_mention_reply(msg.chat_id, msg.user_id, now)
                await self._count_answered_mention(msg.chat_id, msg.user_id)
            else:
                # Самостоятельный ответ не состоялся: в чат ничего не ушло,
                # поэтому бюджет возвращается. Возврат разрешает ответ на
                # следующее сообщение и не переигрывает текущее.
                state.release_reply_slot(slot)
            logger.debug(
                "Generation failed: chat=%s mentioned=%s",
                mask_chat_id(msg.chat_id),
                msg.mentioned,
            )
            return

        # Бюджет уже занят резервацией в `respond` — с тем же признаком
        # `unprompted`, что стоял здесь раньше.
        if obs.address_reply:
            state.note_mention_reply(msg.chat_id, msg.user_id, now)
            await self._count_answered_mention(msg.chat_id, msg.user_id)

        # UTC, как и остальные суточные механики (decay, /pivo-квоты): кап
        # сбрасывается в одну и ту же полночь независимо от TZ контейнера.
        today_iso = datetime.now(UTC).date().isoformat()
        # Причуда уже сломала форму ответа, поэтому редкое событие не роллится
        # и бюджет на него не тратится — один слом формы на ответ.
        reply_parts = await self._apply_user_quirk(msg, obs, reply_text, today_iso)
        if reply_parts is None:
            reply_parts = self._apply_rare_event(msg, reply_text, today_iso)

        delivered = await send(reply_parts)
        if not delivered:
            # Отправитель ничего не доставил и не бросил исключение: ответа в
            # чате нет, бюджет возвращается, анти-повтор не запоминает то,
            # чего никто не видел.
            state.release_reply_slot(slot)
            return
        if is_short_generated_reply(
            tokenize(reply_text, normalize_lower=state.normalize_lower)
        ):
            remember_short_reply(state, msg.chat_id, reply_text)
        remember_recent_reply(state, msg.chat_id, reply_text)

    async def learn(self, msg: IncomingMessage, obs: MessageObservation) -> None:
        """Запись сообщения в модель — выполняется даже при сбое ответа."""
        if not obs.learnable:
            return
        state = self._runtime_state
        learned_token_volume = await self._learning_service.record_message(
            chat_id=msg.chat_id,
            raw_text=obs.learn_source,
            tokens=obs.tokens,
            incremental_cache=state.markov_cache_incremental,
            # M2R-210: писатель обязан гасить короткий слой тем же периодом
            # полураспада, которым его читают; без ручки запись шла бы на
            # константе defaults.py, а чтение — на runtime-значении.
            short_half_life_days=state.markov_short_half_life_days,
        )
        # L1 running jokes: fold the learned message's content n-grams into the
        # sliding hot-ngram window. Gated on the channel knob so a zero chance
        # keeps the learn path write-free (M3 pattern).
        if state.hot_ngram_seed_chance > 0.0:
            content_ngrams = extract_content_ngrams(obs.tokens)
            if content_ngrams:
                await self._learning_service.record_hot_ngrams(
                    msg.chat_id, content_ngrams
                )
        learned = state.learned_messages.get(msg.chat_id, 0) + 1
        state.learned_messages[msg.chat_id] = learned
        if learned == 1 or learned % 25 == 0:
            logger.info(
                "Прогресс обучения: chat=%s, сообщений=%s, объём_модели=%s",
                mask_chat_id(msg.chat_id),
                learned,
                learned_token_volume,
            )

    async def run_due_maintenance(self) -> MaintenanceAlert | None:
        """Отложенное обслуживание базы — отдельным шагом от обучения.

        Планировщика в проекте нет, поэтому суточное затухание «личностных»
        счётчиков крутится отсюда. Отдельный шаг, а не часть записи сообщения:
        обслуживание не должно стоить выученного текста, и это свойство должно
        держаться устройством кода, а не аккуратностью вызывающего.

        Возвращает сигнал владельцу, если о состоянии обслуживания пора
        сказать; отправляет его вызывающий.

        Ручки мемо-пасса передаются отсюда: они /set-абельны, а сервис сам
        RuntimeState не держит — без этого пасс жил бы на конструкторных
        значениях до перезапуска.
        """
        state = self._runtime_state
        # ponytail: state — вид чата, чьё сообщение запустило пасс, а пасс
        # обходит все чаты; per-chat оверрайды мемо-ручек сюда протекают.
        # Глобальный /set (типичный случай для фоновой задачи) корректен.
        return await self._learning_service.run_due_maintenance(
            MemeSettings(
                min_joint_count=state.markov_meme_min_joint_count,
                min_support=state.markov_meme_min_support,
                recency_days=state.markov_meme_recency_days,
                max_entries=state.markov_collocation_max_entries,
            )
        )

    # --- шаги политики ответа ---

    async def _send_fallback(
        self,
        msg: IncomingMessage,
        obs: MessageObservation,
        pool: tuple[str, ...],
        send: Sender,
    ) -> None:
        await send(
            [
                next_fallback_phrase(
                    self._runtime_state,
                    msg.chat_id,
                    pool,
                    now=datetime.now(self._runtime_state.chat_timezone),
                    mood=obs.mood,
                )
            ]
        )

    async def _count_answered_mention(self, chat_id: int, user_id: int) -> None:
        # L2: an answered address counts as an interaction, fallback answers
        # included (the user did interact); gated on the knob so a zero chance
        # keeps the mention path write-free (L1 pattern).
        if self._runtime_state.user_quirk_chance > 0.0:
            await self._learning_service.record_user_interaction(chat_id, user_id)

    def _reply_odds(
        self, msg: IncomingMessage, obs: MessageObservation
    ) -> tuple[float, bool, bool]:
        """``(вероятность ответа, кулдаун пройден, часовой лимит не исчерпан)``.

        M2: when the director is on, conversation momentum (rate/mention/thread
        signals from the shared rhythm state) maps into the [min, max] band,
        then the mood multiplier and the post-reply burst rhythm modulate it and
        a per-hour cap guards against runaway chats. When off, the legacy flat
        probability × mood multiplier is used unchanged.
        """
        state = self._runtime_state
        now = msg.monotonic_now
        has_replied_before = msg.chat_id in state.last_reply_ts
        last_ts = state.last_reply_ts.get(msg.chat_id, 0.0)
        cooldown_ok = cooldown_allows_reply(now, last_ts, state.min_cooldown_sec)

        if state.reply_director_enabled and obs.chat_rhythm is not None:
            momentum = conversation_momentum(
                rate_ewma=obs.chat_rhythm.rate_ewma,
                mention_ewma=obs.chat_rhythm.mention_ewma,
                is_reply=msg.is_reply,
                lively_rate_per_min=state.mood_lively_rate_per_min,
            )
            # A chat with no reply yet gets the neutral factor (a negative
            # sentinel): last_ts=0.0 against monotonic time would otherwise fake
            # a recent reply right after process start and spuriously
            # boost/suppress.
            seconds_since_reply = now - last_ts if has_replied_before else -1.0
            burst = burst_factor(
                seconds_since_reply=seconds_since_reply,
                boost_window_sec=state.reply_burst_boost_sec,
                boost_mult=state.reply_burst_boost_mult,
                suppress_window_sec=state.reply_burst_suppress_sec,
                suppress_mult=state.reply_burst_suppress_mult,
            )
            reply_prob = effective_reply_probability(
                base_min=state.reply_probability_min,
                base_max=state.reply_probability_max,
                momentum=momentum,
                mood_mult=obs.mood_modifiers.reply_probability_mult,
                burst_mult=burst,
            )
            hourly_cap_ok = within_hourly_cap(
                state.recent_reply_times.get(msg.chat_id, ()),
                now,
                state.reply_max_per_hour,
            )
            return reply_prob, cooldown_ok, hourly_cap_ok

        reply_prob = min(
            1.0,
            state.reply_probability * obs.mood_modifiers.reply_probability_mult,
        )
        # reply_max_per_hour — самостоятельный предохранитель, а не деталь
        # директора: легаси-ветка обязана соблюдать его так же.
        hourly_cap_ok = within_hourly_cap(
            state.recent_reply_times.get(msg.chat_id, ()),
            now,
            state.reply_max_per_hour,
        )
        return reply_prob, cooldown_ok, hourly_cap_ok

    def _context_tokens(
        self, msg: IncomingMessage, obs: MessageObservation
    ) -> list[str]:
        state = self._runtime_state
        if not state.use_reply_context:
            return []
        only_for_replies = state.reply_context_only_for_replies
        include_current = state.reply_context_include_current_message
        # When mentioned directly without a reply, override so the current
        # message itself is used as context instead of generating with no
        # context at all.
        if obs.address_reply and not msg.is_reply:
            only_for_replies = False
            include_current = True
        return extract_context_tokens(
            current_text=msg.text,
            reply_context_text=msg.reply_context_text,
            is_reply=msg.is_reply,
            normalize_lower=state.normalize_lower,
            max_tokens=state.reply_context_max_tokens,
            only_for_replies=only_for_replies,
            include_current_message=include_current,
        )

    async def _hot_ngram_seed(
        self, msg: IncomingMessage, obs: MessageObservation
    ) -> list[str] | None:
        """L1 running jokes: изредка открыть самостоятельный ответ «горячей» фразой.

        Ответ на прямое обращение никогда не сидируется — адресованный вопрос
        должен отвечать человеку, а не мему.
        """
        state = self._runtime_state
        if (
            obs.address_reply
            or state.hot_ngram_seed_chance <= 0.0
            or random.random() >= state.hot_ngram_seed_chance
        ):
            return None
        hot_ngrams = await self._learning_service.get_hot_ngrams(
            msg.chat_id,
            min_count=state.hot_ngram_min_count,
            recency_share=state.hot_ngram_recency_share,
            meme_ordering=state.markov_hot_ngram_meme_ordering,
        )
        # M3R-141: counted only once the roll has already decided to seed, so
        # the denominator is "draws that asked for a hot n-gram". At the default
        # thresholds this selection returns nothing at all (map §3.2б) — a
        # channel switched off by data, indistinguishable until now from one
        # switched off by its knob.
        self._generator.telemetry.note_hot_ngram_draw(empty=not hot_ngrams)
        if not hot_ngrams:
            return None
        # Only the length is logged, never the n-gram text: chat content stays
        # out of logs (project log-masking policy).
        seed = list(random.choice(hot_ngrams))
        logger.debug(
            "Hot-ngram seed picked: chat=%s ngram_len=%s",
            mask_chat_id(msg.chat_id),
            len(seed),
        )
        return seed

    async def _apply_user_quirk(
        self,
        msg: IncomingMessage,
        obs: MessageObservation,
        reply_text: str,
        today_iso: str,
    ) -> list[str] | None:
        """L2 user quirks: короткое обращение к завсегдатаю отдельным сообщением.

        Ролл делается до чтения из базы, чтобы обычный путь оставался
        read-free; сам текст ответа не трогается, поэтому анти-повторный учёт
        ниже по-прежнему работает с ``reply_text``. Не чаще одной причуды на
        пару (чат, пользователь) в сутки UTC. ``None`` — причуда не сработала.
        """
        state = self._runtime_state
        telemetry = self._generator.telemetry
        # Гейты идут цепочкой ранних возвратов, а не одним коротко замкнутым
        # `and`: из слитого условия нельзя узнать, какое звено отказало. Цепочка
        # вычисляет те же звенья в том же порядке и останавливается в той же
        # точке, поэтому розыгрыш по-прежнему берётся четвёртым и только при
        # прошедших первых трёх — число обращений к ГСЧ не меняется.
        if not obs.address_reply:
            telemetry.note_user_quirk_outcome(UserQuirkGate.ADDRESSED)
            return None
        if state.user_quirk_chance <= 0.0:
            telemetry.note_user_quirk_outcome(UserQuirkGate.CHANCE)
            return None
        if not state.can_fire_user_quirk(msg.chat_id, msg.user_id, today_iso):
            telemetry.note_user_quirk_outcome(UserQuirkGate.DAILY_LIMIT)
            return None
        if random.random() >= state.user_quirk_chance:
            telemetry.note_user_quirk_outcome(UserQuirkGate.ROLL)
            return None
        interactions = await self._learning_service.get_user_interaction_count(
            msg.chat_id, msg.user_id
        )
        if interactions < state.user_quirk_min_interactions:
            telemetry.note_user_quirk_outcome(UserQuirkGate.THRESHOLD)
            return None
        # L2.1: part of the quirks address the regular by first name. The name
        # is read live from this update and sanitized; it is never stored (the
        # DB side stays an anonymous HMAC counter) and never logged. An unusable
        # name falls back to the pool.
        first_name: str | None = None
        if (
            state.user_quirk_name_share > 0.0
            and random.random() < state.user_quirk_name_share
        ):
            first_name = sanitize_first_name(msg.first_name)
        reply_parts = [
            next_quirk_vocative(random.Random(), first_name=first_name),
            reply_text,
        ]
        state.note_user_quirk(msg.chat_id, msg.user_id, today_iso)
        telemetry.note_user_quirk_outcome(None)
        # Event kind only: no user id, no count, no vocative text in logs
        # (project log-masking policy).
        logger.debug("User quirk fired: chat=%s", mask_chat_id(msg.chat_id))
        return reply_parts

    def _apply_rare_event(
        self, msg: IncomingMessage, reply_text: str, today_iso: str
    ) -> list[str]:
        """L3 rare events: слом формы или фальстарт вместо ровного ответа.

        Ограничено суточным бюджетом чата; запасные фразы не эвентятся.
        Ответ с причудой форму уже сломал — там ролл не делается и бюджет не
        тратится (один слом формы на ответ).
        """
        state = self._runtime_state
        if not state.can_fire_rare_event(msg.chat_id, today_iso):
            return [reply_text]
        event_kind = roll_rare_event(
            random.Random(),
            event_chance=state.rare_event_chance,
            false_start_chance=state.false_start_chance,
        )
        if event_kind is None:
            return [reply_text]
        event_parts = apply_rare_event(event_kind, reply_text, random.Random())
        if event_parts == [reply_text]:
            return [reply_text]
        state.note_rare_event(msg.chat_id, today_iso)
        logger.debug(
            "Rare event fired: chat=%s kind=%s parts=%s",
            mask_chat_id(msg.chat_id),
            event_kind,
            len(event_parts),
        )
        return event_parts
