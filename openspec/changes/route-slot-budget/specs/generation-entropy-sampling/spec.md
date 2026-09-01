# generation-entropy-sampling — delta (route-slot-budget)

## MODIFIED Requirements

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
