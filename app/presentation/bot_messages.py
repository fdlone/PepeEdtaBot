from __future__ import annotations

from typing import TYPE_CHECKING

from app.config.registry import field_hint

if TYPE_CHECKING:
    from app.config.runtime_state import RuntimeState

# New dialogue-generation /set knobs surfaced in /help, one per feature.
# The accepted-range hint is generated from the registry (see field_hint),
# so it can never drift from the value the parser actually enforces.
_DIALOGUE_HELP_KNOBS: tuple[tuple[str, str], ...] = (
    ("mood_enabled", "настроение чата"),
    ("reply_director_enabled", "директор шанса ответа"),
    ("reply_max_per_hour", "лимит ответов в час"),
    ("mention_cooldown_sec", "пауза на упоминания, сек"),
    ("emoji_append_chance", "эмодзи в ответах"),
    ("markov_jump_probability", "дрейф темы"),
    ("hot_ngram_seed_chance", "подхват мемов чата"),
    ("rare_event_chance", "редкие фишки в ответах"),
    ("false_start_chance", "фальстарты"),
    ("pivo_temporal_flavor_chance", "вариации /pivo"),
)

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


def _dialogue_knobs_block() -> str:
    lines = ["Диалог (новое, PR50-58):"]
    for key, description in _DIALOGUE_HELP_KNOBS:
        lines.append(f"/set {key} - {description} ({field_hint(key)})")
    return "\n".join(lines)


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
        f"{_dialogue_knobs_block()}\n"
        "\n"
        "Админское:\n"
        "/clear - инструкция по очистке данных чата"
    )


def format_stats_message(stats: dict[str, int]) -> str:
    return f"объём модели: {stats['volume']}"


def format_config_message(state: RuntimeState, full: bool = False) -> str:
    lines = [
        "Настройки:",
        f"шанс ответа: {state.reply_probability}",
        f"пауза: {state.min_cooldown_sec} сек",
        f"минимум модели: {state.min_tokens_for_model}",
        f"длина ответа: {state.max_reply_tokens} токенов",
        f"вариативность: {state.randomness_strength}",
        f"температура отбора: {state.candidate_selection_temperature}",
        f"вариатор формы: {state.reply_flavor_strength}",
        f"штраф повторов: {state.repetition_penalty_strength}",
        f"штраф недавних ответов: {state.recent_reply_penalty_strength}",
        "веса длины ответа (short,medium,long): "
        + ",".join(str(weight) for weight in state.length_mode_weights),
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
                f"typing_per_char_ms={state.typing_per_char_ms}",
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
        "Очистка удалит данные обучения текущего чата,\n"
        "а также подписки /pivo и их квоты.\n"
        "Для подтверждения отправьте: /clear confirm"
    )
