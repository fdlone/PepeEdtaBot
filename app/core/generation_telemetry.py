"""Process-lifetime generation telemetry (Markov 2.0R Phase 1, M2R-010/020/030).

Counters only — numbers and labels, no text, no chat identifiers (the privacy
contract of the generation-telemetry spec). Owned by ``MarkovGenerator``,
surfaced through ``/stats``; reset on restart by construction.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class CandidateRoute(StrEnum):
    """The mechanism that produced a candidate (M3R-103).

    Closed enumeration on purpose: the route is the axis every per-route number
    is broken down by, so a typo must be an error rather than a new category.

    Only routes with a live producer are members. A member with nothing
    producing it would show up in every report as a permanent zero row, which
    reads as "the mechanism runs and never wins" — the opposite of the truth.
    Adding a route is one member here; the aggregation below walks the
    enumeration, so a new one appears in the breakdown without touching it.
    """

    VANILLA = "vanilla"
    SEEDED = "seeded"
    MUTATED = "mutated"
    EXTENSION = "extension"


class UserQuirkGate(StrEnum):
    """Conditions the L2 user-quirk channel passes, in execution order.

    Ordered on purpose: the funnel is read top to bottom, and the denominator
    of each gate is what the one above it let through. Membership is closed for
    the same reason ``CandidateRoute`` is — a typo must be an error, not a new
    category nobody notices.
    """

    ADDRESSED = "addressed"
    CHANCE = "chance"
    DAILY_LIMIT = "daily_limit"
    ROLL = "roll"
    THRESHOLD = "threshold"


# Предел списка меток времени ответов на обращения. Для оценки пика за
# скользящий час хватает последних наблюдений; точный исторический максимум за
# недели аптайма этой величине не нужен, а расти без предела она не должна.
MENTION_TIME_WINDOW_MAX = 4096


def _peak_in_sliding_hour(times: deque[float]) -> int | None:
    """Наибольшее число меток, попадающих в любое окно длиной час.

    Скользящее окно, а не фиксированные корзины: границы корзин привязаны к
    произвольной эпохе, и всплеск, легший на стык, делится пополам. Тем же
    окном считает `within_hourly_cap` для самостоятельных ответов — величины
    сравниваются между собой, значит и считаться должны одинаково.
    """
    if not times:
        return None
    ordered = sorted(times)
    peak = 0
    left = 0
    for right, moment in enumerate(ordered):
        while moment - ordered[left] >= 3600.0:
            left += 1
        peak = max(peak, right - left + 1)
    return peak


@dataclass(slots=True)
class GenerationTelemetry:
    generations: int = 0
    entropy_bits_sum: float = 0.0
    normalized_entropy_sum: float = 0.0
    branching_sum: float = 0.0
    applied_temperature_sum: float = 0.0
    # M2R-210: effect of the temporal blend, summed over steps.
    blend_covered_steps: int = 0
    blend_displacement_sum: float = 0.0
    # M3R-142: the same displacement measured against the RAW counts — the
    # number that stays non-zero when the short layer is empty and the pair
    # above reads "inert" while sampling runs on compressed weights (map §3.3).
    blend_raw_displacement_sum: float = 0.0
    # M2R-901: effect of the order interpolation, summed over steps. Coverage
    # counts only steps where the order-2 projection actually ADDED a token, so
    # "beta is set but the projection is just as sparse" stays distinguishable
    # from "beta is not set" — the same intent-versus-effect rule the blend
    # above already follows.
    interp_covered_steps: int = 0
    interp_displacement_sum: float = 0.0
    diagnostic_steps: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    shadow_order4_eligible: int = 0
    shadow_order4_selected: int = 0
    # M2R-320: what the active collocations did to candidates. Withheld counts
    # breaks that were NOT penalized because the chain never offered the right
    # token there — reported separately because it is the evidence that the
    # availability guard earns its place (generation-telemetry spec).
    collocation_bonus_hits: int = 0
    collocation_penalty_hits: int = 0
    collocation_withheld: int = 0
    # M2R-300: cost of the daily analyzer passes. Growth of the corpus must
    # show up as a number here before it shows up as a stall of the learn path.
    meme_passes: int = 0
    meme_scored_pairs: int = 0
    meme_pass_ms_sum: float = 0.0
    # M2R-410: two separate denominators (TZ §9.6). "A seeded candidate was
    # present in the pool" and "a seeded candidate won when present" are
    # different findings for the promotion decision, and a single win-rate over
    # all generations hides which one is true.
    seeded_generations: int = 0
    seeded_present: int = 0
    seeded_won: int = 0
    # M3R-103: the same two-denominator rule, per route rather than for seeded
    # alone. ``attempts`` is the mechanism running at all, ``present`` is it
    # putting a candidate in the pool: that pair is what separates "the route is
    # off" from "the route is on and produced nothing" from "it produced and
    # lost". Without it a zero win-rate is unreadable.
    # W2-2: отложенный контекстный якорь мог не дожить до вклейки — прогулка
    # упирается в символьный предел раньше, чем в разыгранную позицию. Такой
    # ответ помечается как `global`, а счётчики совпадений обнуляются, то есть
    # канал молча маскируется под «якоря не было». При этом всю прогулку
    # `anchor_pending` был True, и ветка джампа под ним не исполнялась —
    # вероятность отступления была ровно нулевой без записи об этом.
    #
    # Пара со знаменателем по образцу M3R-141: «якорь отложили» и «якорь
    # вклеили». Расхождение и есть находка.
    anchor_splice_deferred: int = 0
    anchor_splice_done: int = 0
    # E2-1/E2-2 (ревью 2026-08-26): темп ответов наблюдаем, а не выводим из
    # формулы. Оба дефекта — недостижимая фаза отхода берста и отсутствие
    # пер-чатового предела на обращения — зависят от реального темпа чата, а
    # он не измерялся ни разу. Правило §5: пока нет счётчика со знаменателем,
    # «канал не нагружен» неотличимо от «канал не измеряется».
    #
    # Темп — распределением, не средним: поведение переключается скачком на
    # границах, а чат, который то спит, то кипит, даёт среднее, не
    # соответствующее ни одному своему режиму.
    tempo_buckets: dict[str, int] = field(default_factory=dict)
    # Пара со знаменателем: обращений было / на скольких бот ответил. Путь
    # обращений не подчиняется ни кулдауну, ни часовому капу, поэтому своего
    # счётчика у него нет нигде больше.
    mentions_observed: int = 0
    mentions_answered: int = 0
    # Верхняя оценка нагрузки: максимум ответов на обращения, пришедшийся на
    # любой СКОЛЬЗЯЩИЙ час — тем же окном, которым считает `within_hourly_cap`
    # для самостоятельных ответов. Первая редакция раскладывала по
    # фиксированным корзинам `now // 3600`, границы которых привязаны к эпохе
    # монотонных часов и потому произвольны: десять ответов на 59-й минуте и
    # девять на 61-й давали «пик 10» при худшем реальном окне в 19. Занижение
    # до двух раз и всегда в одну сторону — то есть в сторону «предел не
    # нужен», а именно этим числом решается O15.
    mention_answer_times: deque[float] = field(default_factory=deque)
    # Прямой замер E2-1: устойчивый ноль в фазе подавления при ненулевом
    # усилении означает, что фаза недостижима — по факту, а не по расчёту.
    burst_phase_boost: int = 0
    burst_phase_suppress: int = 0
    route_attempts: dict[str, int] = field(default_factory=dict)
    route_present: dict[str, int] = field(default_factory=dict)
    route_won: dict[str, int] = field(default_factory=dict)
    route_rejected: dict[str, dict[str, int]] = field(default_factory=dict)
    # M3R-140: the mode each reply was *requested* in — with context tokens or
    # without. Decided by the request the pipeline built, never by whether the
    # answer turned out anchored (that is a quality metric with its own life).
    # The share is the weight a gate verdict is carried by (roadmap §4) and it
    # exists nowhere else: no table records whether a reply had context.
    ctx_generations: int = 0
    noctx_generations: int = 0
    # M3R-141: channels that ran and had nothing to answer with. Each is a pair
    # with its own denominator — "the channel was asked for" — so a zero
    # numerator reads as "asked and always answered" instead of "never asked".
    # That ambiguity is not hypothetical: a bare seeded win-rate without its
    # denominator is what phase 5 had to redo.
    seed_ranking_asked: int = 0
    seed_ranking_no_corpus: int = 0
    hot_ngram_draws: int = 0
    hot_ngram_empty: int = 0
    # Denominator of this one is ctx_generations: only a generation that
    # started with context can lose it.
    context_dropped: int = 0
    # L2 quirks: the channel passes five gates in a row and only the last two
    # were observable, so "never fires" could not be told apart from "fires and
    # loses the roll". One counter per gate is deliberately NOT enough — the
    # denominators below are derived by walking the funnel, and a gate with a
    # zero denominator means "nothing ever reached it", not "it never rejected".
    quirk_reached: int = 0
    quirk_rejected: dict[str, int] = field(default_factory=dict)
    quirk_fired: int = 0

    def note_cache(self, *, hit: bool) -> None:
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def note_generation(
        self,
        *,
        entropy_bits_sum: float,
        normalized_entropy_sum: float,
        branching_sum: float,
        steps: int,
        applied_temperature_sum: float = 0.0,
        blend_covered_steps: int = 0,
        blend_displacement_sum: float = 0.0,
        blend_raw_displacement_sum: float = 0.0,
        interp_covered_steps: int = 0,
        interp_displacement_sum: float = 0.0,
    ) -> None:
        self.generations += 1
        self.entropy_bits_sum += entropy_bits_sum
        self.normalized_entropy_sum += normalized_entropy_sum
        self.branching_sum += branching_sum
        self.applied_temperature_sum += applied_temperature_sum
        self.blend_covered_steps += blend_covered_steps
        self.blend_displacement_sum += blend_displacement_sum
        self.blend_raw_displacement_sum += blend_raw_displacement_sum
        self.interp_covered_steps += interp_covered_steps
        self.interp_displacement_sum += interp_displacement_sum
        self.diagnostic_steps += steps

    def note_shadow(self, *, eligible: int, selected: int) -> None:
        self.shadow_order4_eligible += eligible
        self.shadow_order4_selected += selected

    def note_collocations(
        self, *, bonus_hits: int, penalty_hits: int, withheld: int
    ) -> None:
        self.collocation_bonus_hits += bonus_hits
        self.collocation_penalty_hits += penalty_hits
        self.collocation_withheld += withheld

    def note_meme_pass(self, *, scored_pairs: int, duration_ms: float) -> None:
        self.meme_passes += 1
        self.meme_scored_pairs += scored_pairs
        self.meme_pass_ms_sum += duration_ms

    def note_seeded(self, *, present: bool, won: bool) -> None:
        """One generation's seeded outcome (M2R-410). ``won`` implies ``present``."""
        self.seeded_generations += 1
        if present:
            self.seeded_present += 1
            if won:
                self.seeded_won += 1

    def note_routes(
        self,
        *,
        attempted: Iterable[str],
        present: Iterable[str],
        winner: str | None,
    ) -> None:
        """One generation's per-route outcome (M3R-103).

        ``attempted`` — routes whose mechanism ran; ``present`` — those that put
        at least one candidate in the pool; ``winner`` — the route of the
        selected candidate, or None when nothing was selected. ``present``
        implies ``attempted`` and the caller is responsible for that.
        """
        for route in attempted:
            self.route_attempts[route] = self.route_attempts.get(route, 0) + 1
        for route in present:
            self.route_present[route] = self.route_present.get(route, 0) + 1
        if winner is not None:
            self.route_won[winner] = self.route_won.get(winner, 0) + 1

    def note_route_rejected(self, route: str, reason: str) -> None:
        """One candidate discarded before the pool, by route and reason.

        Reasons are the existing trace reasons; the F-class taxonomy (M3R-021)
        layers on top of them later and does not replace them.
        """
        self.route_rejected.setdefault(route, {})[reason] = (
            self.route_rejected.setdefault(route, {}).get(reason, 0) + 1
        )

    def note_context_mode(self, *, with_context: bool) -> None:
        """One generation's requested mode (M3R-140), exactly once per call.

        Counted at the start of the generation, so a reply that collects no
        candidates still lands in the denominator: the mode is a property of
        what was asked, not of what came out.
        """
        if with_context:
            self.ctx_generations += 1
        else:
            self.noctx_generations += 1

    def note_seed_ranking(self, *, no_corpus: bool) -> None:
        """One seed-ranking request and whether the chat had no corpus (M3R-141).

        ``no_corpus`` is the ``n_docs == 0`` case specifically — the df
        aggregate has not accumulated yet, so the channel cannot score anything
        however good the message is. Other empty outcomes (no eligible tokens,
        every score below the floor) are not this counter's business.
        """
        self.seed_ranking_asked += 1
        if no_corpus:
            self.seed_ranking_no_corpus += 1

    def note_hot_ngram_draw(self, *, empty: bool) -> None:
        """One hot-n-gram seed draw and whether the selection came back empty."""
        self.hot_ngram_draws += 1
        if empty:
            self.hot_ngram_empty += 1

    def note_context_dropped(self) -> None:
        """One generation that started with context and ran out of with-context
        attempts (M3R-141). Per generation, never per attempt: the finding it
        measures is a share of answers, and a per-attempt count would move with
        the attempt budget instead."""
        self.context_dropped += 1

    def note_anchor_splice(self, *, spliced: bool) -> None:
        """Один отложенный контекстный якорь и дожил ли он до вклейки (W2-2).

        Знаменатель растёт при откладывании, числитель — при вклейке. Ответ,
        потерявший якорь, неотличим в трассе от обычного глобального: метка
        переписывается на `global`, а счётчики совпадений обнуляются. Без этой
        пары доля работы канала занижена на неизвестную величину, причём в
        сторону «канал не работал» — а её как раз и свипует M3R-110.
        """
        self.anchor_splice_deferred += 1
        if spliced:
            self.anchor_splice_done += 1

    def note_chat_tempo(self, rate_per_min: float, *, lively_at: float) -> None:
        """Один наблюдённый темп чата, разложенный по диапазонам (E2-1).

        Границы привязаны к ``lively_at`` — тому самому порогу, которым темп
        переключает настроение и насыщает моментум, — а не к круглым числам:
        распределение должно отвечать на вопрос «в каком режиме живёт чат», а
        не на «какие числа красивее печатать».
        """
        if rate_per_min < lively_at / 6.0:
            bucket = "штиль"
        elif rate_per_min < lively_at / 2.0:
            bucket = "медленно"
        elif rate_per_min < lively_at:
            bucket = "оживлённо"
        else:
            bucket = "кипит"
        self.tempo_buckets[bucket] = self.tempo_buckets.get(bucket, 0) + 1

    def note_mention_seen(self) -> None:
        """Знаменатель: бота упомянули (E2-2).

        Считается **сырое** упоминание, до гейта `mention_cooldown_sec`.
        Первая редакция считала уже отфильтрованный признак `address_reply`,
        и доля ответов получалась тождественно равной 100%: решение отвечать
        на обращение принимается безусловно (`should_reply_to_message`:
        `if mentioned: return True`), поэтому ветка «обращение без ответа»
        была недостижима. Счётчик, заведённый ради решения по O15, не мог
        показать ничего, кроме единицы.

        Упоминание, погашенное кулдауном обращений, — это тоже нагрузка на
        путь, у которого нет ни кулдауна между ответами, ни часового предела.
        Именно она и должна быть видна.
        """
        self.mentions_observed += 1

    def note_mention_answered(self, *, at: float) -> None:
        """Числитель: бот ответил на обращение.

        ``at`` — монотонная метка времени; по ней считается пик за скользящий
        час. Список ограничен сверху, чтобы процесс с долгим аптаймом не рос
        без предела: для оценки пика хватает последних наблюдений, а точный
        исторический максимум за недели этой величине не нужен.

        Обязательный, без значения по умолчанию: с `None` потеря аргумента в
        месте вызова превращала пик в вечный ноль молча, а `/stats` печатает
        его рядом с пределом 20, по которому решается O15. Теперь это ошибка
        типов, а не тихая метрика-обманка.
        """
        self.mentions_answered += 1
        self.mention_answer_times.append(at)
        while len(self.mention_answer_times) > MENTION_TIME_WINDOW_MAX:
            self.mention_answer_times.popleft()

    def note_burst_phase(self, *, suppressing: bool) -> None:
        """Один ответ, сыгранный в фазе усиления или подавления берст-ритма.

        Нейтральная фаза не считается: вопрос ровно в том, попадает ли хоть
        один ответ в окно отхода, — а ответы вне обоих окон на него не
        отвечают.
        """
        if suppressing:
            self.burst_phase_suppress += 1
        else:
            self.burst_phase_boost += 1

    def note_user_quirk_outcome(self, rejected_by: UserQuirkGate | None) -> None:
        """One pass of the L2 quirk channel: which gate stopped it, or none.

        Exactly one call per pass, made from the gate that returned — so the
        funnel cannot double-count and cannot lose a case. ``None`` means every
        gate passed and the quirk fired.

        Counting only: no draw is taken here, and the caller's sequence of
        ``random`` calls is unchanged by construction.
        """
        self.quirk_reached += 1
        if rejected_by is None:
            self.quirk_fired += 1
            return
        self.quirk_rejected[rejected_by] = self.quirk_rejected.get(rejected_by, 0) + 1

    def user_quirk_funnel(self) -> dict[str, int]:
        """Per-gate ``(reached, rejected)`` pairs, flattened for ``/stats``.

        The denominator of each gate is what the previous one let through, so
        the numbers converge arithmetically: whatever survives the last gate is
        ``user_quirk_fired``. A gate never reached shows ``0/0`` rather than
        disappearing — the distinction the channel was blind to.
        """
        funnel: dict[str, int] = {"user_quirk_reached": self.quirk_reached}
        reached = self.quirk_reached
        for gate in UserQuirkGate:
            rejected = self.quirk_rejected.get(gate, 0)
            funnel[f"user_quirk_reached_{gate}"] = reached
            funnel[f"user_quirk_rejected_{gate}"] = rejected
            reached -= rejected
        funnel["user_quirk_fired"] = self.quirk_fired
        return funnel

    def route_breakdown(self) -> dict[str, dict[str, float | int | None]]:
        """Per-route share, win-rate and rejection reasons.

        Kept out of ``snapshot`` because it is nested: ``/stats`` shows flat
        numbers, and the breakdown is for reports that compare routes against
        each other.
        """
        breakdown: dict[str, dict[str, float | int | None]] = {}
        for route in CandidateRoute:
            attempts = self.route_attempts.get(route, 0)
            present = self.route_present.get(route, 0)
            breakdown[str(route)] = {
                "attempts": attempts,
                "present": present,
                "present_rate": present / attempts if attempts else None,
                "won": self.route_won.get(route, 0),
                "win_rate_given_present": (
                    self.route_won.get(route, 0) / present if present else None
                ),
                "rejected": sum(self.route_rejected.get(route, {}).values()),
            }
        return breakdown

    def route_rejection_reasons(self) -> Mapping[str, Mapping[str, int]]:
        """Rejection reason counts per route, as recorded."""
        return {route: dict(reasons) for route, reasons in self.route_rejected.items()}

    def snapshot(self) -> dict[str, float | int | None]:
        """Aggregates for ``/stats``; ``None`` where no data exists yet."""
        steps = self.diagnostic_steps
        lookups = self.cache_hits + self.cache_misses
        eligible = self.shadow_order4_eligible
        modes = self.ctx_generations + self.noctx_generations
        values: dict[str, float | int | None] = {
            # M3R-140/141: every one of these is a rate with its own
            # denominator next to it, so "never asked" is readable as None
            # rather than as a healthy zero.
            "context_mode_generations": modes,
            "ctx_generation_share": (
                self.ctx_generations / modes if modes else None
            ),
            "seed_ranking_asked": self.seed_ranking_asked,
            "seed_ranking_no_corpus_rate": (
                self.seed_ranking_no_corpus / self.seed_ranking_asked
                if self.seed_ranking_asked
                else None
            ),
            "hot_ngram_draws": self.hot_ngram_draws,
            "hot_ngram_empty_rate": (
                self.hot_ngram_empty / self.hot_ngram_draws
                if self.hot_ngram_draws
                else None
            ),
            "context_dropped_rate": (
                self.context_dropped / self.ctx_generations
                if self.ctx_generations
                else None
            ),
            "generations": self.generations,
            "mean_entropy_bits": self.entropy_bits_sum / steps if steps else None,
            "mean_normalized_entropy": (
                self.normalized_entropy_sum / steps if steps else None
            ),
            "mean_branching": self.branching_sum / steps if steps else None,
            "mean_applied_temperature": (
                self.applied_temperature_sum / steps if steps else None
            ),
            "blend_step_coverage": (
                self.blend_covered_steps / steps if steps else None
            ),
            "mean_blend_displacement": (
                self.blend_displacement_sum / steps if steps else None
            ),
            "mean_blend_raw_displacement": (
                self.blend_raw_displacement_sum / steps if steps else None
            ),
            "interp_step_coverage": (
                self.interp_covered_steps / steps if steps else None
            ),
            "mean_interp_displacement": (
                self.interp_displacement_sum / steps if steps else None
            ),
            "cache_hit_rate": self.cache_hits / lookups if lookups else None,
            "shadow_order4_eligible": eligible,
            "shadow_order4_selected_share": (
                self.shadow_order4_selected / eligible if eligible else None
            ),
            "collocation_bonus_hits": self.collocation_bonus_hits,
            "collocation_penalty_hits": self.collocation_penalty_hits,
            "collocation_withheld": self.collocation_withheld,
            "meme_passes": self.meme_passes,
            "meme_scored_pairs": self.meme_scored_pairs,
            "meme_mean_pass_ms": (
                self.meme_pass_ms_sum / self.meme_passes
                if self.meme_passes
                else None
            ),
            "seeded_generations": self.seeded_generations,
            "seeded_present_rate": (
                self.seeded_present / self.seeded_generations
                if self.seeded_generations
                else None
            ),
            "seeded_win_rate_given_present": (
                self.seeded_won / self.seeded_present
                if self.seeded_present
                else None
            ),
        }
        # Absolute counts, not rates: the funnel is read by comparing a gate's
        # rejections against what reached it, and a pre-divided share would
        # throw away exactly the denominator that makes it readable.
        values.update(self.user_quirk_funnel())
        # E2-1/E2-2: доли рядом со своими знаменателями, как у соседей выше.
        # Пустой знаменатель даёт None, а не ноль: «не наблюдали» и «наблюдали
        # и вышел ноль» — разные утверждения.
        tempo_total = sum(self.tempo_buckets.values())
        values["tempo_observations"] = tempo_total or None
        for bucket in ("штиль", "медленно", "оживлённо", "кипит"):
            values[f"tempo_share_{bucket}"] = (
                self.tempo_buckets.get(bucket, 0) / tempo_total
                if tempo_total
                else None
            )
        values["mentions_observed"] = self.mentions_observed or None
        values["mention_answer_share"] = (
            self.mentions_answered / self.mentions_observed
            if self.mentions_observed
            else None
        )
        values["mention_answers_peak_hour"] = _peak_in_sliding_hour(
            self.mention_answer_times
        )
        values["anchor_splice_deferred"] = self.anchor_splice_deferred or None
        values["anchor_splice_share"] = (
            self.anchor_splice_done / self.anchor_splice_deferred
            if self.anchor_splice_deferred
            else None
        )
        burst_total = self.burst_phase_boost + self.burst_phase_suppress
        values["burst_phase_replies"] = burst_total or None
        values["burst_suppress_share"] = (
            self.burst_phase_suppress / burst_total if burst_total else None
        )
        return values
