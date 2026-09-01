from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from app.config.registry import field_hint, runtime_field_names
from app.core.generation_telemetry import UserQuirkGate

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


# Отказы гейтов причуд по-русски, в порядке исполнения: воронка читается
# сверху вниз, и знаменатель каждого гейта — то, что пропустил предыдущий.
_QUIRK_GATE_LABELS = {
    UserQuirkGate.ADDRESSED: "адресность",
    UserQuirkGate.CHANCE: "ручка",
    UserQuirkGate.DAILY_LIMIT: "сутки",
    UserQuirkGate.ROLL: "розыгрыш",
    UserQuirkGate.THRESHOLD: "порог",
}


# `/app/BUILD_AT` — штамп времени сборки, который пишет `RUN date -Is` в
# Dockerfile. Путь считается от пакета, а не задан абсолютом: в контейнере это
# и есть `/app/BUILD_AT`, при локальном запуске — корень репозитория, где файла
# нет, и строка честно говорит `unknown`.
_BUILD_STAMP_PATH = Path(__file__).parents[2] / "BUILD_AT"


def _build_stamp() -> str:
    """Время сборки образа либо ``unknown``, если штампа нет.

    Читается на каждый ``/stats``, а не кэшируется на импорте: команда редкая,
    файл — одна строка, а кэш пришлось бы обходить в тестах.
    """
    try:
        return _BUILD_STAMP_PATH.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        # Файла нет (локальный запуск) либо он не читается — оба случая для
        # читателя означают одно: сборка не опознана.
        return "unknown"


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
    # Идентичность сборки — первой строкой и **всегда**, включая `unknown`.
    # Всё ниже читается через неё: отсутствие любой строки счётчиков значит
    # «нечего показать» только при известной сборке, иначе — «счётчика нет в
    # выкаченной сборке» (`docs/OPERATIONS.md`). Печатать её условно означало бы
    # вернуть ровно ту неоднозначность, ради которой строка и появилась: рестарт
    # 14.08.2026 увёз сборку старше M3R-140, и различить это было нечем.
    lines = [
        f"сборка: {_build_stamp()}",
        f"объём модели: {stats['volume']}",
    ]
    if collocations:
        lines.append(
            "коллокации: "
            + ", ".join(
                f"{status}={count}"
                for status, count in sorted(collocations.items())
            )
        )
    # Темп чата печатается ВНЕ гейта по числу генераций: он наблюдается на
    # каждом сообщении, а чат, где бот с рестарта ни разу не ответил, — как
    # раз тот случай, ради которого счётчик и заведён (E2-1). Прятать его за
    # «были ли генерации» значило бы гасить измерение ровно там, где оно
    # информативнее всего.
    if telemetry and telemetry.get("tempo_observations") is not None:
        tempo_total = telemetry["tempo_observations"]
        shares = [
            f"{name} {telemetry[f'tempo_share_{name}']:.0%}"
            for name in ("штиль", "медленно", "оживлённо", "кипит")
            if telemetry.get(f"tempo_share_{name}")
        ]
        lines.append(f"темп чата ({tempo_total} набл.): " + ", ".join(shares))
    # Обращения и фаза берста — там же, вне гейта по генерациям, и по той же
    # причине: обращение считается в знаменатель, даже когда ответа не было,
    # а «бот молчит на обращения» — это ровно то, что должно быть видно.
    if telemetry and telemetry.get("mentions_observed") is not None:
        answered = telemetry.get("mention_answer_share") or 0.0
        lines.append(
            f"обращений: {telemetry['mentions_observed']}, "
            f"отвечено {answered:.0%}, "
            f"пик за час {telemetry.get('mention_answers_peak_hour') or 0}"
        )
    if telemetry and telemetry.get("burst_phase_replies") is not None:
        suppress = telemetry.get("burst_suppress_share") or 0.0
        lines.append(
            f"берст-ритм: {telemetry['burst_phase_replies']} ответов, "
            f"из них в фазе отхода {suppress:.0%}"
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
        # E2-1/E2-2: темп чата и нагрузка на путь обращений. Обе величины
        # решают судьбу двух найденных дефектов, и обе до сих пор были
        # ненаблюдаемы — правки откладывались именно поэтому.
        # W2-2: отложенный якорь, не доживший до вклейки, помечается как
        # обычный глобальный ответ. Доля вклейки — единственное место, где
        # видно, что канал вообще отработал.
        deferred = telemetry.get("anchor_splice_deferred")
        if deferred is not None:
            spliced = telemetry.get("anchor_splice_share") or 0.0
            lines.append(
                f"контекстный якорь: отложен {deferred} раз, вклеен {spliced:.0%}"
            )
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
    # Воронка причуд L2 — вне блока генераций и без гейта на ненулевые числа:
    # канал молчит с 2026-07-16, и «строки нет» здесь неотличимо от «канал
    # выключен ручкой». Обе стороны пары печатаются всегда, включая нули.
    if telemetry is not None and "user_quirk_reached" in telemetry:
        lines.append(
            f"причуды: сработало {telemetry['user_quirk_fired']} "
            f"из {telemetry['user_quirk_reached']} ответов"
        )
        lines.append(
            "причуды по гейтам: "
            + ", ".join(
                f"{label} {telemetry[f'user_quirk_rejected_{gate}']}"
                f"/{telemetry[f'user_quirk_reached_{gate}']}"
                for gate, label in _QUIRK_GATE_LABELS.items()
            )
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


# Лимит текста одного сообщения Telegram — 4096 символов. Режем с запасом:
# вывод растёт на ~30 символов с каждой новой ручкой реестра, и упереться в
# потолок он должен на нашей проверке, а не на отказе sendMessage.
TELEGRAM_TEXT_LIMIT = 4096
_CHUNK_LIMIT = 3500


def split_for_telegram(text: str, limit: int = _CHUNK_LIMIT) -> list[str]:
    """Разбить длинный вывод по границам строк, не теряя ни одной.

    Усечение здесь недопустимо: `/config full` обещает полный набор значений,
    и вывод, оборванный по длине, противоречит этому обещанию так же, как
    пропущенная настройка. Поэтому режем по строкам и отдаём все части.

    Строка длиннее лимита целиком остаётся своей частью: резать её посередине
    значило бы испортить пару «ключ=значение», а таких строк в выводе реестра
    не бывает — самое длинное имя ручки вдвое короче лимита.
    """
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.split("\n"):
        addition = len(line) + (1 if current else 0)
        if current and current_len + addition > limit:
            parts.append("\n".join(current))
            current, current_len = [line], len(line)
            continue
        current.append(line)
        current_len += addition
    if current:
        parts.append("\n".join(current))
    return parts


def format_config_message(
    state: RuntimeState,
    full: bool = False,
    *,
    overridden: Iterable[str] = (),
) -> str:
    # При включённом директоре плоская reply_probability не читается вовсе:
    # _reply_odds уходит в ветку полосы [min..max] по моментуму беседы
    # (reply_pipeline.py, ветка `state.reply_director_enabled`). Печатать здесь
    # одно число значило показывать ручку, которая на действующих дефолтах ни
    # на что не влияет, — и именно на неё человек смотрит первой строкой.
    if state.reply_director_enabled:
        odds_line = (
            f"шанс ответа: {state.reply_probability_min}..{state.reply_probability_max}"
            " (директор ведёт по моментуму)"
        )
    else:
        odds_line = f"шанс ответа: {state.reply_probability}"
    lines = [
        "Настройки:",
        odds_line,
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
        # Порождается из реестра, а не перечисляется руками. Рукописный список
        # показывал 52 ключа из 97: невидимыми оказались весь директор ответа,
        # настроение, причуды, слот-мутации и `pivo_mention_by_id` — ручка
        # отката механизма упоминаний O2, то есть ровно то, чем чат отличается
        # от дефолта. Дрейф был заложен конструкцией: инструкция «как добавить
        # runtime-mutable параметр» реестр называет, а этот вывод — нет.
        # Спека chat-scoped-settings требовала «полный набор действующих
        # значений» с самого начала; теперь это утверждение проверяемо.
        lines.append("")
        lines.append("Дополнительно:")
        lines.extend(
            f"{name}={getattr(state, name)}" for name in runtime_field_names()
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
