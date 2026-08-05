from __future__ import annotations

import logging
from datetime import datetime

from aiogram import Bot, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message

from app.config.runtime_state import RuntimeState
from app.config.settings import Settings
from app.domain.pivo import (
    MENTION_BY_ID,
    MENTION_BY_USERNAME,
    MENTION_SKIPPED,
    PIVO_PRIVACY_MESSAGE,
    count_mention_paths,
)
from app.filters import GroupOnly, OwnerOnly, is_admin_or_owner
from app.handlers._helpers import reply_humanized_state
from app.log_masking import LogMaskingNotInitialized, mask_chat_id
from app.services import PivoService
from app.services.pivo_parser import parse_pivo_command
from app.services.pivo_service import PivoCallLimitError, PivoCallMessage

router = Router(name="pivo")
logger = logging.getLogger("chat_markov")


def _entity_counts(sent: Message | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entity in (sent.entities if sent is not None else None) or ():
        counts[entity.type] = counts.get(entity.type, 0) + 1
    return counts


async def _report_mentions_to_owner(
    message: Message,
    bot: Bot,
    settings: Settings,
    call: PivoCallMessage,
    sent: Message | None,
) -> None:
    """Шлёт владельцу в личку тот же агрегат, что уходит в лог.

    O2: логи владельцу недоступны, поэтому сигнал, по которому принимается
    решение («появился ли `text_mention`»), обязан доходить до него сам.
    Название чата здесь есть — маска из лога человеку ничего не говорит, —
    а персональных данных подписчиков нет.

    Сообщение уже доставлено в чат, поэтому любая ошибка отправки отчёта
    гасится: отчёт не может отменить состоявшийся вызов.
    """
    if settings.owner_id is None:
        return

    counts = _entity_counts(sent)
    paths = call.mention_paths
    chat_title = message.chat.title or message.chat.full_name or str(message.chat.type)
    report = (
        f"/pivo в «{chat_title}»\n"
        f"упомянуто: {call.mentions_count}\n"
        f"по user_id: {paths.get(MENTION_BY_ID, 0)}, "
        f"строкой: {paths.get(MENTION_BY_USERNAME, 0)}, "
        f"пропущено: {paths.get(MENTION_SKIPPED, 0)}\n"
        f"entity сервера: text_mention={counts.get('text_mention', 0)}, "
        f"mention={counts.get('mention', 0)}"
    )
    try:
        await bot.send_message(settings.owner_id, report)
    except Exception:
        # Самый вероятный случай — владелец не начинал диалог с ботом.
        logger.warning("pivo owner report not delivered", exc_info=True)


def _log_mention_aggregate(
    chat_id: int, call: PivoCallMessage, sent: Message | None
) -> None:
    """Пишет агрегат по упоминаниям: только числа и маскированный chat_id.

    O2: доставку уведомления бот наблюдать не может, а вот entity в объекте
    отправленного сообщения — это разбор сервера, единственный машинный
    сигнал о том, приняты ли упоминания по user_id. Персональных данных в
    записи быть не должно.
    """
    entity_counts = _entity_counts(sent)
    try:
        masked_chat = mask_chat_id(chat_id)
    except LogMaskingNotInitialized:
        # Сообщение уже доставлено — потерять строку в логе допустимо,
        # уронить на ней хендлер нет. Сырой chat_id при этом не пишем.
        masked_chat = "unmasked"
    logger.info(
        "pivo mentions: chat=%s subscribers=%s paths=%s text_mention=%s mention=%s",
        masked_chat,
        call.mentions_count,
        call.mention_paths,
        entity_counts.get("text_mention", 0),
        entity_counts.get("mention", 0),
    )


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
        call = await pivo_service.build_call_message(
            chat_id=message.chat.id,
            caller_user_id=message.from_user.id,
            planned_time=command_args.planned_time,
            target=command_args.target,
            explicit_mentions=command_args.explicit_mentions,
            recent_pool_window=runtime_state.pivo_recent_pool_window,
            temporal_flavor_chance=runtime_state.pivo_temporal_flavor_chance,
            now=datetime.now(),
            mention_by_id=runtime_state.pivo_mention_by_id,
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
        sent = await message.reply(call.text, parse_mode=ParseMode.HTML)
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
        call.picks,
        recent_pool_window=runtime_state.pivo_recent_pool_window,
    )

    logger.info("pivo command executed")
    logger.info("mentions count: %s", call.mentions_count)
    _log_mention_aggregate(message.chat.id, call, sent)
    if runtime_state.pivo_report_to_owner:
        await _report_mentions_to_owner(message, bot, settings, call, sent)


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


@router.message(Command("pivo_check"), GroupOnly(), OwnerOnly())
async def cmd_pivo_check(
    message: Message,
    pivo_service: PivoService,
    runtime_state: RuntimeState,
) -> None:
    """O2-диагностика: каким путём построено упоминание каждого подписчика.

    Ответ намеренно не содержит разметки упоминаний: диагностика не должна
    пинговать чат. Поэтому подпись показывается без ведущей ``@`` — иначе
    Telegram снова разберёт её как упоминание и разошлёт уведомления.
    """
    if message.from_user is None:
        return

    results = await pivo_service.describe_mention_paths(
        message.chat.id,
        message.from_user.id,
        mention_by_id=runtime_state.pivo_mention_by_id,
    )
    if not results:
        await reply_humanized_state(
            message, "В этом чате нет подписчиков /pivo.", runtime_state
        )
        return

    lines = [f"Упоминания /pivo (by_id={runtime_state.pivo_mention_by_id}):"]
    lines += [
        f"- {result.label.lstrip('@') or '?'}: {result.path} ({result.reason})"
        for result in results
    ]
    counts = count_mention_paths(results)
    lines.append(
        f"Итого: {counts[MENTION_BY_ID]} по id, "
        f"{counts[MENTION_BY_USERNAME]} строкой, "
        f"{counts[MENTION_SKIPPED]} пропущено."
    )
    # Расшифрованные ники уходят только в ответ (они и так видны в /pivo),
    # но не в лог — там остаётся один агрегат.
    logger.info("pivo_check executed: paths=%s", counts)
    await reply_humanized_state(message, "\n".join(lines), runtime_state)


@router.message(Command("pivo_check"), GroupOnly())
async def cmd_pivo_check_denied(
    message: Message, runtime_state: RuntimeState
) -> None:
    """Fallback: вызывается, когда OwnerOnly отказал в правах для /pivo_check."""
    await reply_humanized_state(
        message, "Эта команда доступна только владельцу бота.", runtime_state
    )


@router.message(Command("pivo_privacy"), GroupOnly())
async def cmd_pivo_privacy(message: Message, runtime_state: RuntimeState) -> None:
    await reply_humanized_state(message, PIVO_PRIVACY_MESSAGE, runtime_state)
