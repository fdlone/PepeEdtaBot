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

# Per-user per-chat cooldowns, assigned by class of command rather than one by
# one, so a new command inherits a window by where it belongs:
#   - cheap read, answered from memory        -> 15 s
#   - read that touches the database          -> 60 s
#   - write to the database                   -> 30 s
# Without them any participant could loop a command and make the bot answer
# indefinitely — traffic amplification in the bot's name and a flood-ban risk.
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
}

# Commands intentionally without a window, each for a stated reason:
#   - set / setprob / pivo_check / quirk_stats are owner-only, so
#     no participant can use them to amplify traffic, and knob tuning is
#     iterative by nature — rate-limiting the owner against themselves buys
#     nothing.
#   - pivo was deliberately removed from throttling in 282051f: the middleware
#     drops a call *before* the handler, so a throttled /pivo answered with
#     silence instead of the "daily quota exhausted" reply. Its daily quota is
#     the rate limit; re-adding a window here would restore that bug.
COOLDOWN_EXEMPT_COMMANDS = frozenset(
    {"set", "setprob", "pivo_check", "quirk_stats", "pivo"}
)

# Silence reads as "the bot is broken" only where the user asked for something
# and expects confirmation; elsewhere a refusal reply is itself the traffic we
# are damping.
COMMANDS_NOTIFIED_ON_THROTTLE = {"clear", "pivo_on", "pivo_off"}


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
    logging.getLogger("aiogram").setLevel(logging.WARNING)
    gen_trace_log.configure(settings.gen_trace_log)

    db = Database(
        settings.db_path,
        messages_retention_per_chat=settings.messages_retention_per_chat,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
        wal_autocheckpoint_pages=settings.sqlite_wal_autocheckpoint_pages,
    )
    await db.init()
    generator = MarkovGenerator(db=db)
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

    logger.info("Бот %s запущен (polling).", me.username)
    logger.info("Статус: работает.")
    try:
        await dp.start_polling(bot)
    finally:
        logger.info("Статус: остановка...")
        await db.close()
        await bot.session.close()
        logger.info("Статус: остановлен.")


if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except (KeyboardInterrupt, SystemExit):
        pass
