from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from app import log_masking
from app.config.runtime_state import runtime_state_from_settings
from app.config.settings import Settings, load_settings
from app.core import gen_trace_log
from app.core.markov import MarkovGenerator
from app.core.slot_mutation import warm_up as warm_up_morphology
from app.domain.pivo import PivoSecurity
from app.handlers import admin as admin_handlers
from app.handlers import common as common_handlers
from app.handlers import errors as error_handlers
from app.handlers import learning as learning_handlers
from app.handlers import pivo as pivo_handlers
from app.infrastructure.database import Database
from app.middlewares import ChatSettingsMiddleware, ThrottlingMiddleware
from app.presentation.bot_messages import TELEGRAM_COMMANDS
from app.services import LearningService, PivoService
from app.services.db_snapshot import send_startup_snapshot, startup_snapshot
from app.services.meme_analyzer import MemeSettings

# Per-user per-chat cooldowns, assigned by class of command rather than one by
# one, so a new command inherits a window by where it belongs:
#   - cheap read, answered from memory        -> 15 s
#   - read that touches the database          -> 60 s
#   - write to the database                   -> 30 s
#   - admin knob tuning (/set, /setprob)      -> 3 s
# Without them any participant could loop a command and make the bot answer
# indefinitely — traffic amplification in the bot's name and a flood-ban risk.
# /set and /setprob are AdminOrOwner, not owner-only: any chat admin can call
# them, and each call produces an immediate reply. The window is short on
# purpose — long enough to stop a reply loop, short enough not to hinder
# iterative tuning.
# The list is pinned by a test; see COOLDOWN_EXEMPT_COMMANDS for what is
# deliberately left out.
COMMAND_COOLDOWNS_SECONDS = {
    "ping": 15.0,
    "help": 15.0,
    "config": 15.0,
    "pivo_privacy": 15.0,
    "stats": 60.0,
    "pivo_on": 30.0,
    "pivo_off": 30.0,
    "clear": 60.0 * 60.0,
    "set": 3.0,
    "setprob": 3.0,
}

# Commands intentionally without a window, each for a stated reason:
#   - pivo_check / quirk_stats are owner-only (OwnerOnly filter), so no
#     participant can use them to amplify traffic, and rate-limiting the
#     owner against themselves buys nothing.
#   - pivo was deliberately removed from throttling in 282051f: the middleware
#     drops a call *before* the handler, so a throttled /pivo answered with
#     silence instead of the "daily quota exhausted" reply. Its daily quota is
#     the rate limit; re-adding a window here would restore that bug.
#   - db_snapshot is owner-only and private-chat-only for the same reason as
#     pivo_check / quirk_stats: no participant can amplify traffic with it.
COOLDOWN_EXEMPT_COMMANDS = frozenset(
    {"pivo_check", "quirk_stats", "pivo", "db_snapshot"}
)

# Silence reads as "the bot is broken" only where the user asked for something
# and expects confirmation; elsewhere a refusal reply is itself the traffic we
# are damping. /set and /setprob belong here for the same reason as /clear:
# an admin tuning a knob must see that the attempt was throttled, not applied.
COMMANDS_NOTIFIED_ON_THROTTLE = {"clear", "pivo_on", "pivo_off", "set", "setprob"}


def build_meme_settings(settings: Settings) -> MemeSettings:
    """MemeSettings из env-настроек: иначе MARKOV_MEME_* парсятся, но не доходят
    до анализатора, и дневной проход работает на зашитых умолчаниях."""
    return MemeSettings(
        min_joint_count=settings.markov_meme_min_joint_count,
        min_support=settings.markov_meme_min_support,
        recency_days=settings.markov_meme_recency_days,
        max_entries=settings.markov_collocation_max_entries,
    )


def configure_dispatcher(
    dp: Dispatcher,
    *,
    db: Database,
    generator: MarkovGenerator,
    pivo_service: PivoService,
    learning_service: LearningService,
    runtime_state: object,
    settings: Settings,
    bot_username: str,
    bot_id: int,
) -> Dispatcher:
    dp.message.middleware(
        ThrottlingMiddleware(
            limits=COMMAND_COOLDOWNS_SECONDS,
            notify_on_throttle=COMMANDS_NOTIFIED_ON_THROTTLE,
            state_ttl_sec=settings.throttle_state_ttl_sec,
            state_max_keys=settings.throttle_state_max_keys,
            # Так "/clear@ЧужойБот" не сжигает кулдаун этого бота.
            bot_username=bot_username,
        )
    )
    # Registered after throttling so a dropped update never pays for building
    # the per-chat view. Handlers keep reading `runtime_state`; this decides
    # which view that name points at.
    dp.message.middleware(ChatSettingsMiddleware())
    dp["db"] = db
    dp["generator"] = generator
    dp["pivo_service"] = pivo_service
    dp["learning_service"] = learning_service
    dp["runtime_state"] = runtime_state
    dp["settings"] = settings
    dp["bot_username"] = bot_username
    dp["bot_id"] = bot_id
    dp["bot_text_aliases"] = settings.bot_text_aliases
    dp.include_router(common_handlers.router)
    dp.include_router(admin_handlers.router)
    dp.include_router(pivo_handlers.router)
    dp.include_router(learning_handlers.router)
    dp.include_router(error_handlers.router)
    return dp


async def run_bot() -> None:
    settings: Settings = load_settings()
    log_masking.init_masking(settings.pivo_hmac_secret)
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("chat_markov")
    # Маскирование `chat_id` — свойство логгера, а не обязанность каждого
    # вызывающего. Правило §4 CLAUDE.md трижды держалось перечислением точек
    # и трижды промахивалось на новом месте; фильтр видит все записи, включая
    # трассы `exc_info`, которые обёртка вокруг `str(exc)` не покрывает
    # вовсе. Ставится сразу после `init_masking`: до него ключа ещё нет.
    #
    # Вешается и на `aiogram` тоже — сырой идентификатор туда кладёт как раз
    # он, и его собственные предупреждения идут мимо логгера проекта.
    for name in ("chat_markov", "aiogram"):
        logging.getLogger(name).addFilter(log_masking.MaskingFilter())
    logging.getLogger("aiogram").setLevel(logging.WARNING)
    gen_trace_log.configure(settings.gen_trace_log)

    # Снимок обязан быть снят ДО init(): до миграций он одновременно точка
    # отката и «свежая копия прода» для замеров; после init это уже состояние
    # новой сборки. Отправка — позже, когда появится Bot.
    startup_snapshot_path = await startup_snapshot(settings)

    db = Database(
        settings.db_path,
        messages_retention_per_chat=settings.messages_retention_per_chat,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
        wal_autocheckpoint_pages=settings.sqlite_wal_autocheckpoint_pages,
    )
    await db.init()
    # Генератору отдаётся репозиторий цепи, а не фасад: ядру нужен только
    # порт чтения (app/core/markov_port.py), и реализует его именно репозиторий.
    generator = MarkovGenerator(
        db=db.markov, cache_limit=settings.markov_cache_max_entries
    )
    pivo_security = PivoSecurity(
        hmac_secret=settings.pivo_hmac_secret,
        encryption_secret=settings.pivo_encryption_secret,
    )
    pivo_service = PivoService(db=db, security=pivo_security)
    pivo_service.configure_call_limits(
        explicit_mentions_limit=settings.pivo_explicit_mentions_limit,
        subscriber_fanout_limit=settings.pivo_subscriber_fanout_limit,
    )
    learning_service = LearningService(
        db=db,
        generator=generator,
        text_cache_max_messages=settings.text_cache_max_messages,
        # L2 user quirks: interaction counters are keyed by the same HMAC as
        # /pivo subscriptions — no reversible identity ever reaches the DB.
        user_hasher=pivo_security.hmac_value,
        # Кэши обучения ключуются chat_id, как и остальное состояние в памяти,
        # и живут по той же политике: тот же срок хранения и тот же предел по
        # числу чатов, что у runtime-состояния. Отдельные ручки не заводятся —
        # смысл настройки тот же самый: сколько чатов бот помнит и как долго.
        cache_ttl_sec=settings.runtime_state_ttl_sec,
        cache_max_chats=settings.runtime_state_max_chats,
        meme_settings=build_meme_settings(settings),
    )
    runtime_state = runtime_state_from_settings(settings)

    # Slot mutations parse morphology through pymorphy3, whose dictionaries
    # take ~1s to load on first use. That first use would otherwise land in
    # the async reply path and block the event loop for every chat. Warmed
    # unconditionally and in a worker thread: `slot_mutation_probability` is
    # runtime-mutable via /set, so gating on its startup value would leave the
    # stall in exactly the case where someone enables mutations live.
    await asyncio.to_thread(warm_up_morphology)

    bot = Bot(token=settings.bot_token)
    me = await bot.get_me()
    bot_username = (me.username or "").lower()
    await bot.delete_webhook(drop_pending_updates=False)
    await bot.set_my_commands(
        [
            BotCommand(command=command, description=description)
            for command, description in TELEGRAM_COMMANDS
        ]
    )

    dp = configure_dispatcher(
        Dispatcher(),
        db=db,
        generator=generator,
        pivo_service=pivo_service,
        learning_service=learning_service,
        runtime_state=runtime_state,
        settings=settings,
        bot_username=bot_username,
        bot_id=me.id,
    )

    # Доставка стартового снимка — фоном: polling не ждёт за отправкой файла,
    # а отказ доставки гасится внутри задачи предупреждением.
    snapshot_task: asyncio.Task[None] | None = None
    if startup_snapshot_path is not None and settings.owner_id is not None:
        snapshot_task = asyncio.create_task(
            send_startup_snapshot(bot, settings.owner_id, startup_snapshot_path)
        )

    logger.info("Бот %s запущен (polling).", me.username)
    logger.info("Статус: работает.")
    try:
        # Конкурентная обработка оставлена намеренно: aiogram заводит задачу на
        # каждый апдейт, и это правильно — пачка из getUpdates не должна ждать
        # за чужой имитацией набора (до 4 с) и за суточным пассом (~215 мс).
        #
        # `tasks_concurrency_limit=1` рассматривался и отвергнут (O8, решение
        # владельца 2026-08-26): он чинит симптом одной строкой, но
        # сериализует обработку целиком и превращает предохранитель от флуда в
        # источник лагов, заметных людям раньше, чем флуд-бан.
        #
        # Пределы держатся не ограничением конкурентности, а атомарностью
        # решений: слот ответа резервируется в момент решения
        # (`RuntimeState.reserve_reply_slot`), окно команды занимается до
        # вызова хендлера (`ThrottlingMiddleware`). Оба участка «проверил →
        # записал» не содержат `await`, поэтому внутри одного event loop не
        # прерываются.
        await dp.start_polling(bot)
    finally:
        logger.info("Статус: остановка...")
        if snapshot_task is not None and not snapshot_task.done():
            snapshot_task.cancel()
        await db.close()
        await bot.session.close()
        logger.info("Статус: остановлен.")


if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except (KeyboardInterrupt, SystemExit):
        pass
