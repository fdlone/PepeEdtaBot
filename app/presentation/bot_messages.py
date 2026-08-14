from __future__ import annotations

from collections.abc import Iterable
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
        "/config full - все runtime-настройки (админам чата и OWNER_ID)\n"
        "/set help - подсказка по ключам\n"
        "/set <key> <value> - изменить настройку до перезапуска\n"
        "/setprob 0.2 - быстро изменить шанс ответа\n"
        "\n"
        f"{_dialogue_knobs_block()}\n"
        "\n"
        "Админское:\n"
        "/clear - инструкция по очистке данных чата"
    )


def format_stats_message(
    stats: dict[str, int],
    telemetry: dict[str, float | int | None] | None = None,
    collocations: dict[str, int] | None = None,
) -> str:
    """Объём модели плюс телеметрия генерации (Markov 2.0R Phase 1).

    Телеметрия — счётчики за время жизни процесса; до первой генерации после
    рестарта показывать нечего, и блок опускается целиком. ``collocations`` —
    размер реестра коллокаций чата по статусам (M2R-300); пустой реестр
    строку не печатает.
    """
    lines = [f"объём модели: {stats['volume']}"]
    if collocations:
        lines.append(
            "коллокации: "
            + ", ".join(
                f"{status}={count}"
                for status, count in sorted(collocations.items())
            )
        )
    if telemetry and telemetry.get("generations"):
        lines.append(f"генераций с рестарта: {telemetry['generations']}")
        # M3R-140/141: доля режима — вес, с которым вердикт гейта переносится на
        # прод, и она нигде не хранится, кроме этих счётчиков. Рядом —
        # молчаливая потеря контекста: тот же режим, но испортившийся по дороге.
        ctx_share = telemetry.get("ctx_generation_share")
        if ctx_share is not None:
            dropped = telemetry.get("context_dropped_rate")
            dropped_text = (
                f", контекст терялся в {dropped:.0%} из них"
                if dropped is not None
                else ""
            )
            lines.append(f"с контекстом: {ctx_share:.0%} ответов{dropped_text}")
        # Каналы, выключенные данными: ноль в числителе при живом знаменателе
        # значит «спросили и ответил», а не «никто не спрашивал».
        no_corpus = telemetry.get("seed_ranking_no_corpus_rate")
        if no_corpus is not None:
            lines.append(
                f"seeded: корпус пуст в {no_corpus:.0%} из "
                f"{telemetry.get('seed_ranking_asked')} обращений"
            )
        hot_empty = telemetry.get("hot_ngram_empty_rate")
        if hot_empty is not None:
            lines.append(
                f"горячие фразы: пусто в {hot_empty:.0%} из "
                f"{telemetry.get('hot_ngram_draws')} розыгрышей"
            )
        mean_entropy = telemetry.get("mean_normalized_entropy")
        mean_branching = telemetry.get("mean_branching")
        if mean_entropy is not None and mean_branching is not None:
            lines.append(
                f"энтропия шага (норм.): {mean_entropy:.2f}, "
                f"ветвление: {mean_branching:.1f}"
            )
        temperature = telemetry.get("mean_applied_temperature")
        if temperature is not None:
            lines.append(f"температура шага: {temperature:.2f}")
        # M2R-210: покрытие короткого слоя и сдвиг распределения — эффект
        # blend'а, а не его настройка. Выключенный blend читается как два нуля.
        coverage = telemetry.get("blend_step_coverage")
        displacement = telemetry.get("mean_blend_displacement")
        if coverage is not None and displacement is not None:
            lines.append(
                f"свежий слой: покрытие {coverage:.0%}, "
                f"сдвиг распределения {displacement:.3f}"
            )
        hit_rate = telemetry.get("cache_hit_rate")
        if hit_rate is not None:
            lines.append(f"кэш распределений: {hit_rate:.0%} попаданий")
        eligible = telemetry.get("shadow_order4_eligible")
        share = telemetry.get("shadow_order4_selected_share")
        if eligible and share is not None:
            lines.append(
                f"order-4 (тень, оценка по окну): выбрался бы в {share:.0%} "
                f"из {eligible} шагов"
            )
        # M2R-410: два раздельных знаменателя — присутствовал ли seeded-кандидат
        # в пуле и, отдельно, выиграл ли при наличии. Слитая в одну ставка
        # скрыла бы, редко ли он появляется или появляется, но проигрывает.
        seeded_present = telemetry.get("seeded_present_rate")
        if seeded_present is not None:
            win = telemetry.get("seeded_win_rate_given_present")
            win_text = f"{win:.0%}" if win is not None else "н/д"
            lines.append(
                f"seeded: присутствовал в {seeded_present:.0%} генераций, "
                f"побеждал при наличии в {win_text}"
            )
        # M2R-320: вес в конфиге — намерение, эти счётчики — эффект. Отдельный
        # withheld и есть свидетельство, что гард доступности не мёртвый код.
        applied = (
            (telemetry.get("collocation_bonus_hits") or 0)
            + (telemetry.get("collocation_penalty_hits") or 0)
            + (telemetry.get("collocation_withheld") or 0)
        )
        if applied:
            lines.append(
                "коллокации в скоринге: "
                f"бонусов {telemetry.get('collocation_bonus_hits')}, "
                f"штрафов {telemetry.get('collocation_penalty_hits')}, "
                f"удержано {telemetry.get('collocation_withheld')}"
            )
    # M2R-300: стоимость суточного анализа — вне блока генераций, проход
    # случается и до первой генерации после рестарта.
    if telemetry and telemetry.get("meme_passes"):
        mean_ms = telemetry.get("meme_mean_pass_ms") or 0.0
        lines.append(
            f"мем-анализ: {telemetry['meme_passes']} проходов, "
            f"пар оценено {telemetry.get('meme_scored_pairs')}, "
            f"среднее {mean_ms:.0f} мс"
        )
    return "\n".join(lines)


def format_config_message(
    state: RuntimeState,
    full: bool = False,
    *,
    overridden: Iterable[str] = (),
) -> str:
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
                # Phase 2 knobs that change what the chat hears. The clamps are
                # a safety net and stay out of the listing; gain and pivot are
                # what explain a changed voice, so they are readable here.
                f"markov_entropy_temp_gain={state.markov_entropy_temp_gain}",
                f"markov_entropy_pivot={state.markov_entropy_pivot}",
                f"markov_branching_degenerate_max={state.markov_branching_degenerate_max}",
                # Phase 3 temporal blend: the alphas decide how much of the
                # chat's fresh language is mixed in, and the half-life and
                # compression shape decide what "fresh" and "historical" even
                # weigh — all four explain a changed voice, so all four are
                # readable here.
                f"markov_alpha_sleepy={state.markov_alpha_sleepy}",
                f"markov_alpha_calm={state.markov_alpha_calm}",
                f"markov_alpha_lively={state.markov_alpha_lively}",
                f"markov_alpha_heated={state.markov_alpha_heated}",
                f"markov_short_half_life_days={state.markov_short_half_life_days}",
                f"markov_long_compression={state.markov_long_compression}",
                f"markov_long_compression_beta={state.markov_long_compression_beta}",
                # Phase 4 collocation scoring (M2R-320): the weights that let
                # the chat's memes nudge candidate selection, plus the
                # hot-ngram ordering switch (M2R-310). All neutral until the
                # phase gate says otherwise.
                f"markov_collocation_bonus={state.markov_collocation_bonus}",
                f"markov_collocation_break_penalty={state.markov_collocation_break_penalty}",
                f"markov_hot_ngram_meme_ordering={state.markov_hot_ngram_meme_ordering}",
                # Phase 5 lexical anchoring (M2R-410): the seeded-candidate
                # share plus the seed-choice band. Neutral (ratio 0) until the
                # promotion gate says otherwise.
                f"markov_seeded_candidate_ratio={state.markov_seeded_candidate_ratio}",
                f"markov_seed_branch_min={state.markov_seed_branch_min}",
                f"markov_seed_branch_ideal={state.markov_seed_branch_ideal}",
                f"markov_seed_branch_max={state.markov_seed_branch_max}",
                f"markov_seed_min_score={state.markov_seed_min_score}",
                f"markov_seed_head_share={state.markov_seed_head_share}",
                f"reply_context_max_tokens={state.reply_context_max_tokens}",
                f"reply_context_bias={state.reply_context_bias}",
                f"reply_context_start_bias={state.reply_context_start_bias}",
                f"reply_context_only_for_replies={state.reply_context_only_for_replies}",
                f"reply_context_include_current_message={state.reply_context_include_current_message}",
            ]
        )
    lines.append("")
    if overridden:
        # The values above are what THIS chat sees; say which of them stopped
        # following the global default, otherwise the reader cannot tell a
        # tuned chat from an untouched one.
        lines.append(
            "Переопределено для этого чата: " + ", ".join(sorted(overridden))
        )
        lines.append("Вернуть к глобальному: /set reset <key>")
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
        "Очистка удалит данные обучения текущего чата\n"
        "(включая анонимные счётчики взаимодействий),\n"
        "а также подписки /pivo и их квоты.\n"
        "Для подтверждения отправьте: /clear confirm"
    )
