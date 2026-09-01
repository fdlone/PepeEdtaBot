# generation-entropy-sampling Specification

## Purpose
Defines how the transition distribution's own uncertainty is allowed to modulate generation — the entropy-to-temperature mapping and its bounds, the branching-aware candidate target, the neutrality contract that keeps the frozen 1.x behavior reachable, and the boundary between a soft reweighting and the hard acceptance gates. Normative sources: TZ §6, ADR-003, roadmap doc 03 M2R-100/110.

## Requirements

### Requirement: Sampling temperature follows pool entropy

For each transition pool the walk samples from, the system SHALL derive the sampling temperature from the pool's normalized entropy as `T = T_base · (1 + GAIN · (H_norm − H_pivot))`, clamped to a configured `[T_min, T_max]`. `T_base` SHALL remain governed by the existing randomness setting, so that setting keeps its established meaning as the overall scale. Deriving and applying the temperature SHALL NOT consume random draws beyond those the 1.x sampler already consumes.

#### Scenario: Wide pool at positive gain

- **WHEN** a walk step samples from a pool whose normalized entropy is above the pivot, with a positive gain
- **THEN** the applied temperature is above `T_base`, and the resulting weights are flatter than the 1.x weights for the same pool

#### Scenario: Degenerate pool

- **WHEN** a walk step samples from a pool of one candidate
- **THEN** normalized entropy is 0 by definition, the temperature is computed without a division error, and the single candidate is still chosen

#### Scenario: Clamp holds

- **WHEN** a gain and pivot combination would drive the temperature outside the configured bounds
- **THEN** the applied temperature is the nearer bound, and no configuration of gain or pivot can produce a non-positive or unbounded temperature

### Requirement: Zero gain reproduces the frozen baseline bit-for-bit

With the entropy feature disabled, or enabled with a gain of zero, generation output SHALL be byte-identical to the frozen Markov 1.x baseline for the same seed, snapshot and settings. This identity SHALL be demonstrated by a reproducible hash over generated text, not asserted. The default configuration SHALL satisfy this identity until an ablation gate says otherwise.

#### Scenario: Knob off

- **WHEN** the same seeded generation runs on the released build with the entropy knob off and on the frozen baseline
- **THEN** the generation hashes are equal

#### Scenario: Gain set to zero

- **WHEN** the entropy feature is enabled but the gain is zero
- **THEN** the generation hash still equals the frozen baseline's, so enabling the machinery alone changes nothing

#### Scenario: Reverting live

- **WHEN** the gain is set back to zero at runtime without a restart
- **THEN** subsequent generations match the baseline again, with no restart and no data migration

### Requirement: Entropy never overrides acceptance gates

Entropy modulation SHALL affect only the weighting of candidates inside a pool that the generator is already allowed to sample from. It SHALL NOT admit a token that a hard rule excludes, SHALL NOT relax the candidate acceptance gates (copy, repetition, length and the rest), and SHALL NOT change which pool a step samples from.

#### Scenario: Excluded token stays excluded

- **WHEN** a continuation is excluded from a step's pool by an existing rule and the entropy-adjusted temperature would otherwise favor it
- **THEN** the continuation is still absent from the pool and cannot be selected

#### Scenario: Rejected candidate stays rejected

- **WHEN** a generated candidate fails an acceptance gate
- **THEN** it is rejected regardless of the entropy conditions under which it was produced

### Requirement: Candidate target follows observed branching

The number of candidates the generator aims to produce before scoring SHALL be derivable from the branching the generator actually observed, within a configured floor and the existing attempt budget: a chain whose pools are near-degenerate SHALL be allowed to stop early, since further attempts on such a chain differ only marginally, while a wide-branching chain SHALL be allowed the full target. With the feature disabled the target SHALL equal the previously fixed constant.

Этот целевой размер SHALL быть бюджетом **всего** пула, а не только основного
обхода. Всякий дополнительный производитель кандидатов (маршрут) SHALL брать
слоты изнутри него и SHALL NOT расширять пул сверх целевого размера.

Основание — измеримое: страховочный инвариант разнообразия пре-регистрирован
как **абсолютное** число существенно различных траекторий, поэтому пул,
растущий с числом маршрутов, обесценивает порог именно на тех правках, ради
которых порог заведён. Наращивание пула вычеркнуто отдельно и с цифрами
(глобальные обходы с низкой результативностью плюс расход бюджета латентности),
и маршрут, добавляющий кандидатов сверх, возвращает вычеркнутое.

Бюджет маршрутов SHALL быть ограничен так, чтобы основной обход сохранял
большинство слотов пула: пул без единого кандидата основного обхода — это
подмена механизма, а не конкуренция за места.

#### Scenario: Degenerate chain stops early

- **WHEN** the observed branching of the produced candidates is at or below the configured degenerate bound
- **THEN** the generator stops at the reduced target, and the reply is still produced from at least one accepted candidate

#### Scenario: Feature disabled

- **WHEN** the branching-aware target is disabled
- **THEN** the candidate target equals the constant used before this change, and generation output is byte-identical to the baseline

#### Scenario: No empty reply from an early stop

- **WHEN** the reduced target would be reached with zero accepted candidates
- **THEN** the generator keeps attempting up to the existing budget rather than returning no reply

#### Scenario: Маршрут включён — пул не растёт

- **WHEN** маршрут включён своей ручкой и произвёл кандидатов
- **THEN** размер пула не превышает целевого размера, а слоты маршрута заняты за счёт слотов основного обхода

#### Scenario: Маршрут не может вытеснить основной обход

- **WHEN** ручка маршрута выставлена в максимум своего диапазона
- **THEN** маршрут получает не больше половины слотов пула, и в пуле остаётся хотя бы один кандидат основного обхода, если обход вообще способен его произвести

#### Scenario: Маршрут ничего не произвёл

- **WHEN** маршрут включён, но не отдал ни одного кандидата
- **THEN** его слоты достаются основному обходу, и пул собирается до целевого размера как обычно

### Requirement: The shipped default is decided by a pre-registered gate

The default values of the entropy gain and of the branching-aware target SHALL be set from a protocol evaluation run whose thresholds were registered before the results were examined. If the gate does not pass, the feature SHALL ship disabled and the phase SHALL be recorded as closed with a negative result, citing the report. A default that changes live behavior SHALL NOT be chosen by preference alone.

#### Scenario: Gate passes

- **WHEN** the ablation run meets every pre-registered Phase 2 threshold
- **THEN** the calibrated gain becomes the default and the report backing that number is cited from the change

#### Scenario: Gate fails

- **WHEN** any pre-registered Phase 2 threshold is missed
- **THEN** the default stays at the neutral value, the feature remains available behind its knob, and the outcome is recorded with the report reference

### Requirement: The applied temperature is observable

Generation telemetry SHALL report the temperature actually applied, aggregated per generation, alongside the entropy it was derived from, so the live effect of the setting is visible without reading the source. The reported values SHALL be numbers only, subject to the existing telemetry privacy rules.

#### Scenario: Trace shows the applied temperature

- **WHEN** a generation completes with the entropy feature enabled
- **THEN** its trace carries the mean applied temperature together with the mean normalized entropy

#### Scenario: Neutral configuration is visible as neutral

- **WHEN** the feature is disabled or the gain is zero
- **THEN** telemetry reports the unmodulated temperature, so "the knob is doing nothing" is readable from the numbers
