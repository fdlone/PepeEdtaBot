from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from app import log_masking
from app.handlers import admin as admin_handlers
from app.handlers import common as common_handlers
from app.handlers import learning as learning_handlers
from app.handlers import pivo as pivo_handlers
from app.middlewares import ThrottlingMiddleware
from app.services import LearningService, PivoService
from bot_messages import TELEGRAM_COMMANDS
from db import Database
from markov import MarkovGenerator
from pivo import PivoSecurity
from runtime_state import runtime_state_from_settings
from settings import Settings, load_settings

COMMAND_COOLDOWNS_SECONDS = {
    "clear": 60.0 * 60.0,
}


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
            notify_on_throttle={"clear"},
            state_ttl_sec=settings.throttle_state_ttl_sec,
            state_max_keys=settings.throttle_state_max_keys,
        )
    )
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

    db = Database(settings.db_path)
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
    learning_service = LearningService(db=db, generator=generator)
    runtime_state = runtime_state_from_settings(settings)

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
