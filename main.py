from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand

from app.handlers import admin as admin_handlers
from app.handlers import common as common_handlers
from app.handlers import learning as learning_handlers
from app.handlers import pivo as pivo_handlers
from app.services import LearningService, PivoService
from bot_messages import TELEGRAM_COMMANDS
from db import Database
from markov import MarkovGenerator
from pivo import PivoSecurity
from runtime_state import runtime_state_from_settings
from settings import Settings, load_settings


async def run_bot() -> None:
    settings: Settings = load_settings()
    logging.basicConfig(
        level=logging.INFO,
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
    learning_service = LearningService(db=db, generator=generator)
    state = runtime_state_from_settings(settings)

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

    dp = Dispatcher()
    dp["db"] = db
    dp["generator"] = generator
    dp["pivo_service"] = pivo_service
    dp["learning_service"] = learning_service
    dp["state"] = state
    dp["settings"] = settings
    dp["bot_username"] = bot_username
    dp["bot_id"] = me.id
    dp.include_router(common_handlers.router)
    dp.include_router(admin_handlers.router)
    dp.include_router(pivo_handlers.router)
    dp.include_router(learning_handlers.router)

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
