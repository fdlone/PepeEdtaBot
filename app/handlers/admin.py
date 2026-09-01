from __future__ import annotations

import logging

from aiogram import Bot, Router
from aiogram.filters import Command
from aiogram.types import Message

from app.config.registry import (
    UNKNOWN_RUNTIME_KEY_MESSAGE,
    InvalidRuntimeSettingValueError,
    UnknownRuntimeSettingError,
    apply_runtime_setting,
    clear_runtime_setting,
)
from app.config.runtime_state import RuntimeState
from app.config.settings import Settings
from app.core.markov import MarkovGenerator
from app.filters import (
    GROUP_ONLY,
    AdminOrOwner,
    OwnerOnly,
    is_admin_or_owner,
    is_owner,
)
from app.handlers._helpers import (
    reply_humanized_sequence_state,
    reply_humanized_state,
)
from app.infrastructure.database import Database
from app.presentation.bot_messages import (
    format_clear_confirmation_message,
    format_config_message,
    format_set_help_message,
    split_for_telegram,
)
from app.services import LearningService, PivoService

router = Router(name="admin")
logger = logging.getLogger("chat_markov")


def _extract_command_arg(text: str) -> str:
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) >= 2 else ""


async def _reply_no_permission(
    message: Message, runtime_state: RuntimeState
) -> None:
    """Отказ для /set и /setprob тем, кто не админ этого чата и не владелец."""
    await reply_humanized_state(
        message,
        "Команда доступна админам этого чата и OWNER_ID.",
        runtime_state,
    )


async def _reply_global_denied(
    message: Message, runtime_state: RuntimeState
) -> None:
    """Отказ на глобальную форму тому, кто админ чата, но не владелец.

    Значение при этом НЕ применяется к чату вызова: человек просил изменить
    всё, и тихая подмена области скрыла бы, что произошло не то.
    """
    await reply_humanized_state(
        message,
        "Глобальная форма доступна только OWNER_ID. "
        "Без слова global настройка меняется в этом чате.",
        runtime_state,
    )


@router.message(Command("config"), GROUP_ONLY)
async def cmd_config(
    message: Message,
    runtime_state: RuntimeState,
    runtime_state_base: RuntimeState,
    bot: Bot,
    settings: Settings,
) -> None:
    raw = _extract_command_arg(message.text or "")
    full = raw.strip().lower() == "full"
    # Краткая форма открыта всем: понимать, почему бот ведёт себя так, вправе
    # каждый участник. Полная — это карта модерации (пороги, кулдауны, лимиты),
    # и читать её логично тому же кругу, который может её менять через /set.
    if full and not await is_admin_or_owner(message, bot, settings):
        await reply_humanized_state(
            message,
            "Полный список настроек доступен админам этого чата и OWNER_ID.\n"
            "Основные значения: /config",
            runtime_state,
        )
        return
    # runtime_state is already this chat's view, so the values shown are the
    # effective ones; the base state only says which of them were overridden.
    text = format_config_message(
        runtime_state,
        full=full,
        overridden=runtime_state_base.chat_overrides.get(message.chat.id, {}),
    )
    # Полная форма растёт с каждой ручкой реестра и однажды перестанет
    # помещаться в одно сообщение. Резать её заранее дешевле, чем узнать об
    # этом отказом sendMessage: усечение противоречило бы слову «полный» так
    # же, как пропущенная настройка. Краткая форма проходит одной частью.
    await reply_humanized_sequence_state(
        message, split_for_telegram(text), runtime_state
    )


# O5 (docs/OPEN.md) was about reach, not about who asks: /set used to retune
# every chat at once, so any chat's admins could affect all the others. The
# overlay closed that — the scoped form only touches the chat it was sent in,
# which its admins already control (they can evict the bot or /clear it). So
# the scoped form is AdminOrOwner again; the `global` form still carries the
# process-wide effect and stays OWNER_ID-only, checked inside the handler
# because the scope is a word in the text, not a separate command.
@router.message(Command("set"), GROUP_ONLY, AdminOrOwner())
async def cmd_set(
    message: Message,
    runtime_state: RuntimeState,
    settings: Settings,
    runtime_state_base: RuntimeState,
    learning_service: LearningService,
) -> None:
    raw = _extract_command_arg(message.text or "")
    if raw.strip().lower() == "help":
        await reply_humanized_state(
            message, format_set_help_message(), runtime_state
        )
        return
    usage = (
        "Использование: /set <key> <value>\n"
        "Глобально во всех чатах: /set global <key> <value>\n"
        "Вернуть чат к глобальному значению: /set reset <key>\n"
        "Подсказка: /set help"
    )
    if not raw:
        await reply_humanized_state(message, usage, runtime_state)
        return
    parts = raw.split(maxsplit=2)
    scope_word = parts[0].strip().lower()

    if scope_word == "reset":
        if len(parts) != 2:
            await reply_humanized_state(message, usage, runtime_state)
            return
        key = parts[1].strip().lower()
        # runtime_state — вид чата, снятый ДО сброса: эффективное значение до
        # снятия оверрайда читается отсюда (M2R-210, см. ниже).
        previous_half_life = runtime_state.markov_short_half_life_days
        try:
            cleared = clear_runtime_setting(runtime_state_base, message.chat.id, key)
        except UnknownRuntimeSettingError:
            await reply_humanized_state(
                message,
                f"{UNKNOWN_RUNTIME_KEY_MESSAGE}\n\nПодсказка: /set help",
                runtime_state,
            )
            return
        except InvalidRuntimeSettingValueError as exc:
            await reply_humanized_state(
                message, f"Сброс не применён: {exc}", runtime_state
            )
            return
        text = (
            f"{key}: чат вернулся к глобальному значению."
            if cleared
            else f"{key}: в этом чате и так глобальное значение."
        )
        # M2R-210: сброс оверрайда — тоже смена периода полураспада, если
        # глобальное значение отличается; короткий слой чата к нему
        # несопоставим. Совпадающие значения слой не трогают.
        if (
            cleared
            and key == "markov_short_half_life_days"
            and runtime_state_base.markov_short_half_life_days
            != previous_half_life
        ):
            await learning_service.reset_short_layer(message.chat.id)
            text += (
                "\n\nВнимание: короткий слой чата обнулён — счётчик свежести "
                "математически привязан к прежнему периоду полураспада. "
                "Долгая память чата не тронута, свежесть накопится заново "
                "примерно за "
                f"{runtime_state_base.markov_short_half_life_days:g} дн."
            )
        await reply_humanized_state(message, text, runtime_state)
        return

    is_global = scope_word == "global"
    if is_global and not await is_owner(message, settings):
        await _reply_global_denied(message, runtime_state)
        return

    arguments = parts[1:] if is_global else parts
    if len(arguments) != 2:
        await reply_humanized_state(message, usage, runtime_state)
        return

    key, value = arguments[0].strip().lower(), arguments[1].strip()
    # M2R-210 / TZ §7.2: the short layer is mathematically tied to the half-life
    # it accumulated under, so changing that knob discards it. Read the old
    # value before applying, so an unchanged value stays a no-op. The global
    # form compares against the global BASE value: the invoking chat's
    # override would otherwise mask a real base change (reset skipped) or
    # fake one (base unchanged, override differs — spurious wipe).
    previous_half_life = (
        runtime_state_base if is_global else runtime_state
    ).markov_short_half_life_days
    try:
        apply_runtime_setting(
            runtime_state_base,
            key,
            value,
            chat_id=None if is_global else message.chat.id,
        )
    except UnknownRuntimeSettingError:
        await reply_humanized_state(
            message,
            f"{UNKNOWN_RUNTIME_KEY_MESSAGE}\n\nПодсказка: /set help",
            runtime_state,
        )
        return
    except InvalidRuntimeSettingValueError as exc:
        # Конфликт с оверрайдами чата несёт своё сообщение (какой чат, какие
        # ключи); обычная ошибка значения остаётся коротким отказом.
        detail = str(exc)
        await reply_humanized_state(
            message,
            "Некорректное значение для этого ключа."
            + (f"\n{detail}" if detail else ""),
            runtime_state,
        )
        return

    scope = "глобально, во всех чатах" if is_global else "в этом чате"
    text = f"Обновлено: {key}={value} ({scope}, до перезапуска)"

    if key == "markov_short_half_life_days":
        new_half_life = float(value)
        if new_half_life != previous_half_life:
            reset_chat = None if is_global else message.chat.id
            await learning_service.reset_short_layer(reset_chat)
            text += (
                "\n\nВнимание: короткий слой обнулён — счётчик свежести "
                "математически привязан к прежнему периоду полураспада и "
                f"несопоставим с новым. Долгая память чата не тронута, "
                f"свежесть накопится заново примерно за {value} дн."
            )

    await reply_humanized_state(message, text, runtime_state)


@router.message(Command("set"), GROUP_ONLY)
async def cmd_set_denied(message: Message, runtime_state: RuntimeState) -> None:
    """Fallback: вызывается, когда AdminOrOwner отказал в правах для /set."""
    await _reply_no_permission(message, runtime_state)


# Same split as /set: scoped form to chat admins, `global` to OWNER_ID only.
@router.message(Command("setprob"), GROUP_ONLY, AdminOrOwner())
async def cmd_setprob(
    message: Message,
    runtime_state: RuntimeState,
    settings: Settings,
    runtime_state_base: RuntimeState,
) -> None:
    raw = _extract_command_arg(message.text or "").strip()
    usage = "Использование: /setprob 0.2 (или /setprob global 0.2)"
    if not raw:
        await reply_humanized_state(message, usage, runtime_state)
        return

    parts = raw.split()
    is_global = parts[0].lower() == "global"
    if is_global and not await is_owner(message, settings):
        await _reply_global_denied(message, runtime_state)
        return
    if is_global:
        parts = parts[1:]
    if len(parts) != 1:
        await reply_humanized_state(message, usage, runtime_state)
        return

    # Goes through the shared apply path instead of assigning the field: that
    # is what routes the value into the right scope and keeps cross-field
    # validation in one place.
    try:
        apply_runtime_setting(
            runtime_state_base,
            "reply_probability",
            parts[0],
            chat_id=None if is_global else message.chat.id,
        )
    except (UnknownRuntimeSettingError, InvalidRuntimeSettingValueError):
        await reply_humanized_state(
            message, "Нужно число в диапазоне 0..1", runtime_state
        )
        return

    scope = "глобально, во всех чатах" if is_global else "в этом чате"
    # Ручка записана, но при включённом директоре она не участвует в решении
    # отвечать: полосу задают reply_probability_min/max. Молчаливое
    # подтверждение «теперь 0.2» отправляло человека искать причину в
    # генерации, хотя причина в том, что он крутит не ту ручку.
    note = (
        ""
        if not runtime_state.reply_director_enabled
        else (
            "\nДиректор ответов включён, поэтому шанс ведёт полоса "
            f"{runtime_state.reply_probability_min}..{runtime_state.reply_probability_max}; "
            "эта ручка сейчас не действует. Менять — /set reply_probability_min|max, "
            "выключить директора — /set reply_director_enabled false"
        )
    )
    await reply_humanized_state(
        message, f"REPLY_PROBABILITY теперь: {parts[0]} ({scope}){note}", runtime_state
    )


@router.message(Command("setprob"), GROUP_ONLY)
async def cmd_setprob_denied(
    message: Message, runtime_state: RuntimeState
) -> None:
    """Fallback: вызывается, когда AdminOrOwner отказал в правах для /setprob."""
    await _reply_no_permission(message, runtime_state)


@router.message(Command("quirk_stats"), GROUP_ONLY, OwnerOnly())
async def cmd_quirk_stats(
    message: Message,
    runtime_state: RuntimeState,
    learning_service: LearningService,
) -> None:
    """Достижим ли порог «завсегдатая» в этом чате.

    Фича, которая никогда не срабатывает, снаружи неотличима от сломанной:
    причуды L2 живут с 2026-07-16, а обращений в чате никто не видел. Ответ —
    только числа: счётчики хранятся анонимно (HMAC), и диагностика этого не
    отменяет.
    """
    threshold = runtime_state.user_quirk_min_interactions
    (
        people,
        max_count,
        at_or_above,
        buckets,
    ) = await learning_service.get_user_interaction_stats(message.chat.id, threshold)
    if people == 0:
        await reply_humanized_state(
            message,
            "Обращений к боту пока не накоплено — счётчиков в этом чате нет.",
            runtime_state,
        )
        return

    # Распределение — группировкой, а не значением на человека: перечень
    # счётчиков превратил бы обезличенный агрегат в профиль участников.
    # Границы привязаны к действующему порогу, поэтому суммирование справа
    # прямо отвечает на «сколько добавится, если порог опустить».
    distribution = ", ".join(
        f"{lower}+: {count}" if upper == 0 else f"{lower}–{upper - 1}: {count}"
        for lower, upper, count in buckets
    )
    text = (
        f"Счётчики обращений (порог завсегдатая: {threshold}):\n"
        f"людей со счётчиком: {people}\n"
        f"максимум: {max_count}\n"
        f"достигли порога: {at_or_above}\n"
        f"распределение: {distribution}"
    )
    if at_or_above == 0:
        text += "\n\nПорог пока не взял никто — причуды не срабатывают поэтому."
    await reply_humanized_state(message, text, runtime_state)


@router.message(Command("quirk_stats"), GROUP_ONLY)
async def cmd_quirk_stats_denied(
    message: Message, runtime_state: RuntimeState
) -> None:
    """Fallback: вызывается, когда OwnerOnly отказал в правах."""
    await reply_humanized_state(
        message, "Команда доступна только OWNER_ID.", runtime_state
    )


@router.message(Command("clear"), GROUP_ONLY, AdminOrOwner())
async def cmd_clear(
    message: Message,
    db: Database,
    runtime_state: RuntimeState,
    generator: MarkovGenerator,
    settings: Settings,
    pivo_service: PivoService,
    learning_service: LearningService,
) -> None:
    raw = _extract_command_arg(message.text or "")
    if raw.strip().lower() != "confirm":
        await reply_humanized_state(
            message, format_clear_confirmation_message(), runtime_state
        )
        return
    await db.clear_chat(message.chat.id)
    await pivo_service.clear_chat_data(message.chat.id)
    generator.invalidate_chat_cache(message.chat.id)
    # Кэши обучения переживали очистку: гейт дословного повтора продолжал
    # считать цитатой сообщения, которых в базе уже нет.
    learning_service.forget_chat(message.chat.id)
    runtime_state.forget_chat(message.chat.id)
    await reply_humanized_state(
        message, "Данные чата очищены (включая подписки /pivo).", runtime_state
    )


@router.message(Command("clear"), GROUP_ONLY)
async def cmd_clear_denied(message: Message, runtime_state: RuntimeState) -> None:
    """Fallback: вызывается, когда AdminOrOwner отказал в правах для /clear."""
    await reply_humanized_state(
        message,
        "Недостаточно прав. Нужен OWNER_ID или права админа чата.",
        runtime_state,
    )
