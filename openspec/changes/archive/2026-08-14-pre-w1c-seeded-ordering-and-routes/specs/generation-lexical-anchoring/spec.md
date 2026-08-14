## MODIFIED Requirements

### Requirement: Seeded candidates compete without priority

Seeded candidates SHALL join the best-of-N pool and be judged by the same
scorer as every other candidate, with no bonus for being seeded (ADR-008). The
share of the pool that is seeded SHALL be configurable.

«Без приоритета» SHALL означать и отсутствие гандикапа: seeded-кандидат
проходит **тот же** конвейер доводки и **те же** гейты формы, что кандидат
основного обхода, прежде чем попасть в пул. Ветка, которая собирает кандидата
мимо общей доводки, SHALL считаться дефектом, а не вариантом реализации: разница
в доводке проявляется как систематическая разница в оценке, и тогда гейт
продвижения измеряет полноту реализации ветки, а не идею, которую он проверяет.

#### Scenario: Seeded candidate wins on merit

- **WHEN** a seeded candidate and ordinary candidates are scored together
- **THEN** the seeded candidate wins only if its score is best, by the same rule applied to all candidates

#### Scenario: Seeded candidate is finalized like any other

- **WHEN** seeded-кандидат собран и готовится войти в пул
- **THEN** к нему применены те же шаги доводки и те же гейты формы, что к кандидату основного обхода, и по наблюдаемым признакам доводки (терминальная пунктуация, отсутствие ведущей пунктуации) он неотличим от него

#### Scenario: Malformed seeded candidate is rejected, not scored

- **WHEN** seeded-кандидат после доводки не проходит гейт формы, применяемый к кандидатам основного обхода
- **THEN** он отклоняется на том же основании, а не попадает в пул с заниженной оценкой
