# generation-telemetry Specification

## Purpose
Defines what the Markov generator must be able to report about its own distributions and machinery — uncertainty diagnostics, shadow order-selection statistics, cache effectiveness — where those numbers surface, and what must never leak into them. Normative sources: TZ §6 (formulas), §20 (observability), doc 03 M2R-010/020.
## Requirements
### Requirement: Distribution diagnostics are computed for every sampled pool

For every transition pool the generation walk samples from, the system SHALL compute entropy in bits (`H = -Σ p_i·log2(p_i)` over the pool's normalized weights), branching factor (pool size), normalized entropy (`H / log2(B)`, defined as 0 when branching ≤ 1), and confidence (`1 − H_norm`). Computing diagnostics SHALL NOT change generation behavior and SHALL NOT consume random draws.

Diagnostics SHALL be computed from the model's raw-count proportions, not from the temperature-adjusted sampling weights, so that a feature consuming the diagnostics cannot feed its own effect back into them. Where a feature does consume them, the behavioral change SHALL be attributable to that feature's own setting: with every such consumer at its neutral setting, output SHALL remain byte-identical to the frozen baseline.

#### Scenario: Diagnostics on a walk step

- **WHEN** a walk step samples a continuation from a pool of B > 1 candidates
- **THEN** the step's entropy, normalized entropy, branching factor, and confidence are available in the generation trace

#### Scenario: Degenerate pool

- **WHEN** a pool contains a single candidate
- **THEN** normalized entropy is 0 and confidence is 1, with no division error

#### Scenario: Behavior unchanged

- **WHEN** the same generation runs with every diagnostics consumer at its neutral setting and with the diagnostics code absent, under the same seed
- **THEN** the generated text is byte-identical

#### Scenario: Diagnostics are not self-referential

- **WHEN** a consumer changes the sampling weights based on a pool's entropy
- **THEN** the entropy reported for that pool is unchanged, because it is computed from raw counts rather than from the adjusted weights

### Requirement: Shadow order-4 selection is measured without an order-4 index

The system SHALL estimate, for each order-3 walk step with at least 4 tokens of history, whether a variable-order 4→3→2 selector would have chosen order 4 — using order-4 support estimated from the retained message window (no order-4 storage exists before Phase 7; the estimate feeds that phase's pre-registered gate). The shadow computation SHALL NOT alter the walk, SHALL NOT consume random draws, and SHALL be disableable by a runtime knob. Telemetry SHALL publish both the number of eligible steps and the share where order 4 would have been selected, and SHALL record that the estimator is window-based (a conservative lower bound).

#### Scenario: Shadow counters accumulate

- **WHEN** generations run with the shadow selector enabled
- **THEN** telemetry reports eligible-step count and would-select-order-4 share for the process lifetime

#### Scenario: Shadow disabled

- **WHEN** the shadow knob is off
- **THEN** no shadow computation runs and generation output is unchanged either way

### Requirement: Cache effectiveness is observable

The distribution caches SHALL count hits and misses, and `/stats` SHALL report the hit-rate alongside the model volume. Counters SHALL be per-process (reset on restart) and cheap enough to leave always on.

#### Scenario: Hit-rate in /stats

- **WHEN** `/stats` is called after some generations
- **THEN** the reply includes the cache hit-rate derived from the process counters

### Requirement: Telemetry stays inside the privacy rules

Telemetry values SHALL be numbers and enum-like labels only: no raw message or candidate text outside the existing opt-in trace log, and chat identifiers only in masked form in any log line (the existing log-privacy rules extend to every new telemetry line).

#### Scenario: New telemetry log line

- **WHEN** a telemetry line mentioning a chat is written to the log
- **THEN** the chat identifier appears only masked, and no message text appears

### Requirement: The blend reports what it actually did

Every generation SHALL report the mixing weight it sampled at and how much the
blend actually changed what was sampled — the share of sampled steps where the
short layer contributed any weight, and how far the blended distribution moved
away from the long layer alone, averaged over the walk. These numbers SHALL come
from the values the walk used, not from the configured settings, so that a
setting which cannot take effect is distinguishable from one that did.

A mixing weight is a configured intent; the distance the distribution actually
moved is the effect. Reporting only the former is how a knob that changes
nothing gets read as a knob that works.

#### Scenario: Blend active

- **WHEN** a generation runs with a non-zero mixing weight over a chain with a populated short layer
- **THEN** the trace reports the mixing weight together with the short layer's step coverage and the mean distance the blend moved the distribution

#### Scenario: Neutral configuration reads as neutral

- **WHEN** a generation runs with the mixing weight at zero
- **THEN** the reported mixing weight is zero and the reported movement is zero, not absent

#### Scenario: Blend enabled but short layer empty

- **WHEN** the mixing weight is non-zero but no state in the walk has short-layer weight
- **THEN** the reported coverage and movement are both zero while the mixing weight is reported as set, distinguishing "configured but inert" from "not configured"

### Requirement: Temporal telemetry stays within the privacy rules

Temporal telemetry SHALL consist of aggregate numbers only. Observation times
SHALL NOT be logged in a form that reconstructs when an individual message was
sent, and chat identifiers in temporal log lines SHALL be masked exactly as the
existing log-privacy rules require.

#### Scenario: Temporal trace line emitted

- **WHEN** a trace line carrying temporal numbers is written
- **THEN** it contains only aggregates, carries no per-message timestamp, and its chat identifier is masked

### Requirement: Seeded generation reports its two denominators separately

The system SHALL report how often a seeded candidate was present in the pool
and, separately, how often a seeded candidate won when one was present. The two
SHALL NOT be collapsed into a single rate: "a seeded candidate rarely appears"
and "it appears but loses" are different findings with different consequences
for the promotion decision, and a single win-rate over all generations hides
which one is true.

As with the temporal blend and the collocation counters, the configured seeded
share is the intent and these counts are the effect: a configuration whose
seeded branch never produces a candidate SHALL be distinguishable from one that
is not configured.

#### Scenario: Seeded generation active

- **WHEN** replies are generated with a non-zero seeded share
- **THEN** the count of generations with a seeded candidate present, and the count of those won by a seeded candidate, are both reported

#### Scenario: Configured but never anchoring

- **WHEN** the seeded share is non-zero but no chat has a token clearing the minimum seed score
- **THEN** the seeded-present count is zero and is distinguishable from the seeded share being zero


### Requirement: The collocation registry and its effect are observable

The system SHALL expose how many collocations a chat has in each status, and how
often the bonus and the penalty were actually applied during generation. As with
the temporal blend, the configured weight is the intent and the application
counts are the effect: a configuration whose rules never match SHALL be
distinguishable from one that is not configured.

The withheld penalty — the case where a candidate broke a collocation but the
chain never offered its right token — SHALL be counted separately, because it is
the guard against punishing the corpus and its size is the evidence that the
guard matters.

#### Scenario: Scoring with an active registry

- **WHEN** replies are generated in a chat with active collocations
- **THEN** the counts of applied bonuses, applied penalties and withheld penalties are reported

#### Scenario: Configured but never matching

- **WHEN** the bonus is non-zero but no candidate ever contains an active collocation
- **THEN** the reported application count is zero while the registry size is reported as non-zero

#### Scenario: Registry visible per chat

- **WHEN** chat statistics are requested
- **THEN** they include the number of collocations by status for that chat

### Requirement: Analyzer cost is reported, not assumed

Each maintenance pass of the analyzer SHALL record how long it took and how many
pairs it scored, so growth in the corpus shows up as a number before it shows up
as a stall.

#### Scenario: Pass completes

- **WHEN** the analyzer finishes a pass
- **THEN** its duration and the number of scored pairs are available in the telemetry, with chat identifiers masked per the existing log-privacy rules

### Requirement: Every candidate carries the route that produced it

Каждый кандидат, попадающий в пул отбора, SHALL нести признак маршрута —
механизма, которым он собран. Набор маршрутов SHALL быть замкнутым перечислением
с местом под будущие маршруты, а не свободной строкой: маршрут — ось разбора
телеметрии, и опечатка в нём обязана быть ошибкой, а не новой категорией.

Признак SHALL быть виден в трассе генерации для каждого кандидата, а не только
для победителя, и SHALL проставляться в точке создания кандидата, а не
восстанавливаться постфактум по косвенным признакам его текста.

#### Scenario: Pool assembled from several mechanisms

- **WHEN** пул собран из кандидатов разного происхождения
- **THEN** у каждого кандидата в трассе виден его маршрут, и кандидаты разных механизмов различимы без внешнего инструмента

#### Scenario: Rejected candidate

- **WHEN** кандидат отклонён гейтом и не попал в пул
- **THEN** его маршрут виден в трассе вместе с причиной отклонения

### Requirement: Telemetry is broken down by route

Счётчики генерации SHALL быть разложены по маршрутам, а не только суммарно:
доля пула, доля побед, причины отклонений, тематичность и доля копирования —
каждый в разрезе маршрута. Суммарное число без разложения SHALL NOT считаться
достаточным основанием для решения о маршруте: «качество выросло» без указания,
какой механизм его дал, не атрибутирует вклад и не позволяет ни включить, ни
откатить конкретный маршрут.

Разложение SHALL сохранять уже действующее правило двух знаменателей:
«маршрут не появился в пуле» и «появился и проиграл» остаются различимыми для
каждого маршрута, а не только для seeded.

Маршрут, не давший ни одного кандидата, SHALL быть отличим от маршрута,
выключенного настройкой, — тем же правилом «интент против эффекта», что уже
действует для временнóй смеси, коллокаций и seeded-доли.

#### Scenario: Two routes contribute to the pool

- **WHEN** replies are generated with more than one candidate-producing mechanism active
- **THEN** доля пула и доля побед сообщаются отдельно для каждого маршрута

#### Scenario: Route active but never producing

- **WHEN** маршрут включён настройкой, но ни разу не дал кандидата
- **THEN** его счётчик присутствия равен нулю и отличим от случая, когда маршрут выключен

### Requirement: Route labels carry no chat content

Признак маршрута и его счётчики SHALL оставаться внутри действующих
privacy-правил телеметрии: маршрут — это метка механизма из фиксированного
перечисления, и он SHALL NOT содержать ни текста сообщений, ни токенов чата, ни
идентификаторов, ни производных от них значений.

#### Scenario: Telemetry emitted with routes

- **WHEN** телеметрия с по-маршрутной разбивкой выгружается
- **THEN** она содержит только метки маршрутов из перечисления и числа

### Requirement: Order interpolation reports its intent next to its effect

Система SHALL сообщать два числа о смешивании порядков: долю шагов, где слой
order-2 добавил в пул **хотя бы одного нового** кандидата, и среднее расхождение
слитого распределения с чистым order-3.

Пара обязательна по той же причине, по которой она обязательна для временнóй
смеси: настроенный вес — это интент, а эти два числа — эффект. Конфигурация, чей
слой order-2 не добавил ничего (разреженность до дна: там тоже одно
продолжение), SHALL быть отличима от конфигурации, где вес не выставлен. Без
такого разделения нулевой результат читается как «механизм не работает», хотя он
может означать «механизму не на чем работать» — и это разные выводы с разными
последствиями для решения о промоушене.

Расхождение SHALL измеряться как расстояние между распределениями, а не как
разница выбранных токенов: механизм может сдвинуть массу, не изменив
победителя, и такой сдвиг обязан быть виден.

#### Scenario: Interpolation active

- **WHEN** ответы генерируются с ненулевым весом order-2
- **THEN** доля шагов с добавленным кандидатом и среднее расхождение с чистым order-3 сообщаются оба

#### Scenario: Configured but nothing to add

- **WHEN** вес order-2 ненулевой, но проекции не предлагают ни одного нового кандидата
- **THEN** доля добавивших шагов равна нулю и отличима от случая, когда вес равен нулю

#### Scenario: Telemetry stays within the privacy rules

- **WHEN** счётчики интерполяции выгружаются
- **THEN** они содержат только числа — без текста кандидатов, без токенов и без идентификаторов чата

### Requirement: The context mode of every reply is counted

The system SHALL count, per process, how many replies were generated **with**
context tokens and how many **without**, and SHALL publish the share in the same
place the other generation aggregates surface (`/stats`). The mode is decided by
the request the pipeline built, not by whether the reply happened to use the
context: a request carrying no context tokens is `noctx` even if the generator
found an anchor by other means.

The counters exist so that a gate verdict can be weighted by the measured
production share of each mode (`docs/PRE_ROADMAP.md` §4). A verdict SHALL NOT
claim a production-weighted result while this share is unmeasured.

Counting SHALL NOT consume random draws and SHALL NOT change generation output.

#### Scenario: Reply with context

- **WHEN** a reply is generated from a request whose context token list is non-empty
- **THEN** the `ctx` counter increments and the `noctx` counter does not

#### Scenario: Reply without context

- **WHEN** a reply is generated from a request whose context token list is empty
- **THEN** the `noctx` counter increments and the `ctx` counter does not

#### Scenario: Share is reported

- **WHEN** `/stats` is called after replies in both modes
- **THEN** the reply includes the share of generations built with context, derived from the process counters

#### Scenario: Output unchanged

- **WHEN** the same generation runs with the mode counters present and absent, under the same seed
- **THEN** the generated text is byte-identical

### Requirement: A channel emptied by data is distinguishable from one turned off by a knob

For every data-fed generation channel, the system SHALL count the case where the
channel **ran and returned nothing because its data source was empty**,
separately from the case where its knob is neutral and the channel never ran. At
minimum this SHALL cover:

- the seeded channel skipped because the chat's document count is zero, counted
  only when the seeded ratio knob is non-zero (that is: the channel was asked
  for and could not answer);
- the hot-n-gram seeding channel whose selection query returned an empty list
  when a seed was actually drawn for;
- the context of a generation dropped by exhausting the with-context attempt
  budget, counted per generation.

Each counter SHALL be published with the denominator that makes it readable —
the number of generations in which the channel was asked for — so that a zero
rate means "asked and always answered" rather than "never asked". A channel with
its knob at the neutral value SHALL report zero in both numerator and
denominator rather than being silently absent.

Counting SHALL NOT consume random draws and SHALL NOT change generation output.

#### Scenario: Seeded channel starved of document frequency

- **WHEN** the seeded ratio knob is non-zero and the chat's document count is zero
- **THEN** the "seeded skipped, no corpus" counter increments and its denominator (generations where seeding was asked for) increments too

#### Scenario: Seeded channel not asked for

- **WHEN** the seeded ratio knob is at its neutral value
- **THEN** neither the skip counter nor its denominator increments, and the reported rate is undefined rather than zero

#### Scenario: Hot n-grams selection returns nothing

- **WHEN** a seed draw queries the hot n-grams and the query returns an empty list
- **THEN** the "hot n-grams empty" counter increments against the number of draws that queried it

#### Scenario: Context dropped by attempt budget

- **WHEN** a generation consumes more attempts than the with-context budget allows, so later attempts run without the context
- **THEN** the generation is counted as one whose context was dropped, and the generation trace records that it happened

#### Scenario: Trace records the drop

- **WHEN** the optional per-generation trace is enabled and a generation drops its context by attempt budget
- **THEN** the trace line for that generation states the drop, alongside the existing context fields

#### Scenario: Output unchanged

- **WHEN** the same generation runs with the empty-channel counters present and absent, under the same seed
- **THEN** the generated text is byte-identical
