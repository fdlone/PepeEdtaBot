from __future__ import annotations

import logging
from datetime import datetime

from aiogram import Bot, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from app.config.runtime_state import RuntimeState
from app.config.settings import Settings
from app.domain.pivo import PIVO_PRIVACY_MESSAGE
from app.filters import GroupOnly, is_admin_or_owner
from app.handlers._helpers import reply_humanized_state
from app.services import PivoService
from app.services.pivo_parser import parse_pivo_command
from app.services.pivo_service import PivoCallLimitError

router = Router(name="pivo")
logger = logging.getLogger("chat_markov")


@router.message(Command("pivo"), GroupOnly())
async def cmd_pivo(
    message: Message,
    pivo_service: PivoService,
    runtime_state: RuntimeState,
    bot: Bot,
    settings: Settings,
) -> None:
    if message.from_user is None:
        return
    command_args = parse_pivo_command(message)
    is_owner = (
        settings.owner_id is not None and message.from_user.id == settings.owner_id
    )
    quota = None

    try:
        text, mentions_count, pool_picks = await pivo_service.build_call_message(
            chat_id=message.chat.id,
            caller_user_id=message.from_user.id,
            planned_time=command_args.planned_time,
            target=command_args.target,
            explicit_mentions=command_args.explicit_mentions,
            recent_pool_window=runtime_state.pivo_recent_pool_window,
            temporal_flavor_chance=runtime_state.pivo_temporal_flavor_chance,
            now=datetime.now(),
        )
    except PivoCallLimitError as exc:
        await reply_humanized_state(message, str(exc), runtime_state)
        logger.info("pivo command rejected by call limit: %s", exc)
        return

    if not is_owner:
        is_privileged = await is_admin_or_owner(message, bot, settings)
        quota = await pivo_service.consume_daily_call_quota(
            chat_id=message.chat.id,
            user_id=message.from_user.id,
            is_admin_or_owner=is_privileged,
        )
        if not quota.allowed:
            await reply_humanized_state(
                message,
                f"Лимит /pivo на сегодня исчерпан: {quota.limit} раз(а) в сутки.",
                runtime_state,
            )
            logger.info(
                "pivo command rejected by daily quota: limit=%s day=%s",
                quota.limit,
                quota.usage_day,
            )
            return

    try:
        await message.reply(text, parse_mode=ParseMode.HTML)
    except Exception:
        if quota is not None:
            await pivo_service.refund_daily_call_quota(
                chat_id=message.chat.id,
                user_id=message.from_user.id,
                usage_day=quota.usage_day,
            )
            logger.exception(
                "pivo command failed after quota consumption; quota refunded"
            )
        else:
            logger.exception("pivo command failed")
        raise

    # Anti-repeat state is recorded only after the reply is actually delivered,
    # so quota-rejected or failed calls do not rotate the template pools.
    await pivo_service.record_pool_usage(
        message.chat.id,
        pool_picks,
        recent_pool_window=runtime_state.pivo_recent_pool_window,
    )

    logger.info("pivo command executed")
    logger.info("mentions count: %s", mentions_count)


@router.message(Command("pivo_on"), GroupOnly())
async def cmd_pivo_on(
    message: Message, pivo_service: PivoService, runtime_state: RuntimeState
) -> None:
    if message.from_user is None:
        return
    if message.from_user.is_bot:
        await reply_humanized_state(
            message, "Ботов в пивной список не добавляю.", runtime_state
        )
        return

    await pivo_service.subscribe(message.chat.id, message.from_user)
    logger.info("pivo member subscribed")
    await reply_humanized_state(
        message, "Готово, теперь я буду звать тебя через /pivo.", runtime_state
    )


@router.message(Command("pivo_off"), GroupOnly())
async def cmd_pivo_off(
    message: Message, pivo_service: PivoService, runtime_state: RuntimeState
) -> None:
    if message.from_user is None:
        return

    await pivo_service.unsubscribe(message.chat.id, message.from_user.id)
    logger.info("pivo member removed")
    await reply_humanized_state(
        message, "Готово, больше не буду звать тебя через /pivo.", runtime_state
    )


@router.message(Command("pivo_privacy"), GroupOnly())
async def cmd_pivo_privacy(message: Message, runtime_state: RuntimeState) -> None:
    await reply_humanized_state(message, PIVO_PRIVACY_MESSAGE, runtime_state)
