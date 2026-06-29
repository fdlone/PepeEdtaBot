from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config.runtime_state import RuntimeState

TELEGRAM_COMMANDS: tuple[tuple[str, str], ...] = (
    ("help", "Список команд"),
    ("ping", "Проверить, что бот онлайн"),
    ("pivo", "Позвать в Discord: время, повод, упоминания"),
    ("pivo_on", "Включить себя в список /pivo"),
    ("pivo_off", "Выключить себя из списка /pivo"),
    ("pivo_privacy", "Как работает список /pivo"),
    ("stats", "Краткая статистика модели"),
    ("config", "Основные настройки"),
    ("set", "Изменить настройку до перезапуска"),
    ("setprob", "Быстро изменить шанс ответа"),
    ("clear", "Очистка данных чата с подтверждением"),
)


def format_help_message() -> str:
    return (
        "Команды:\n"
        "\n"
        "Основное:\n"
        "/help - список команд\n"
        "/ping - проверить, что бот онлайн\n"
        "/stats - краткая статистика модели\n"
        "/pivo [время] [повод] [@кого] - позвать в Discord\n"
        "  примеры: /pivo, /pivo 20:00, /pivo фильм,\n"
        "  /pivo 20:00 фильм, /pivo 20:00 фильм @user, /pivo @user\n"
        "/pivo_on - включить себя в список /pivo\n"
        "/pivo_off - выключить себя из списка /pivo\n"
        "/pivo_privacy - как используются данные для /pivo\n"
        "\n"
        "Настройки:\n"
        "/config - основные настройки\n"
        "/config full - все runtime-настройки\n"
        "/set help - подсказка по ключам\n"
        "/set <key> <value> - изменить настройку до перезапуска\n"
        "/setprob 0.2 - быстро изменить шанс ответа\n"
        "\n"
        "Админское:\n"
        "/clear - инструкция по очистке данных чата"
    )


def format_stats_message(stats: dict[str, int], min_tokens_for_model: int) -> str:
    effective_volume = stats["volume"]
    ready_text = "достаточно" if effective_volume >= min_tokens_for_model else "мало"
    return (
        "Статистика:\n"
        f"сообщений: {stats['messages']}\n"
        f"объём модели: {effective_volume}\n"
        f"готовность: {ready_text}"
    )


def format_config_message(state: RuntimeState, full: bool = False) -> str:
    lines = [
        "Настройки:",
        f"шанс ответа: {state.reply_probability}",
        f"пауза: {state.min_cooldown_sec} сек",
        f"минимум модели: {state.min_tokens_for_model}",
        f"длина ответа: {state.max_reply_tokens} токенов",
        f"вариативность: {state.randomness_strength}",
        f"штраф повторов: {state.repetition_penalty_strength}",
        f"reply-контекст: {state.use_reply_context}",
    ]
    if full:
        lines.extend(
            [
                "",
                "Дополнительно:",
                f"max_reply_chars={state.max_reply_chars}",
                f"normalize_lower={state.normalize_lower}",
                f"typing_min_ms={state.typing_min_ms}",
                f"typing_max_ms={state.typing_max_ms}",
                f"markov_order={state.markov_order}",
                f"enable_backoff={state.enable_backoff}",
                f"backoff_min_order={state.backoff_min_order}",
                f"reply_context_max_tokens={state.reply_context_max_tokens}",
                f"reply_context_last_tokens={state.reply_context_last_tokens}",
                f"reply_context_bias={state.reply_context_bias}",
                f"reply_context_start_bias={state.reply_context_start_bias}",
                f"reply_context_only_for_replies={state.reply_context_only_for_replies}",
                f"reply_context_include_current_message={state.reply_context_include_current_message}",
            ]
        )
    lines.append("")
    lines.append("Изменения через /set действуют до перезапуска.")
    return "\n".join(lines)


def format_set_help_message() -> str:
    return (
        "Настройки через /set:\n"
        "\n"
        "Чаще всего:\n"
        "/set reply_probability 0.08\n"
        "/set min_cooldown_sec 45\n"
        "/set min_tokens_for_model 200\n"
        "/set max_reply_tokens 45\n"
        "/set randomness_strength 2.0\n"
        "/set repetition_penalty_strength 1.0\n"
        "\n"
        "Reply-контекст:\n"
        "/set use_reply_context true\n"
        "/set reply_context_max_tokens 12\n"
        "/set reply_context_bias 1.8\n"
        "\n"
        "Полный список текущих значений: /config full"
    )


def format_clear_confirmation_message() -> str:
    return (
        "Очистка удалит данные обучения текущего чата.\n"
        "Для подтверждения отправьте: /clear confirm"
    )
