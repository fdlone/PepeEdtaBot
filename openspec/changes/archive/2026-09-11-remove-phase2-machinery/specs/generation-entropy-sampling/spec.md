## REMOVED Requirements

### Requirement: Sampling temperature follows pool entropy

**Reason**: Гейт фазы 2 провален всеми шестью плечами 2026-08-12: энтропия
шага на корпусе бимодальна (78.8% шагов с одним продолжением, 20.9% почти
равномерных, 0.3% в середине), температуре не за что зацепиться. Решение
владельца 2026-09-11 (O22, группа 2): механизм удалён, не заморожен.

**Migration**: Ручки `markov_entropy_temp_gain` / `_pivot` / `_temp_min` /
`_temp_max` удалены; `randomness_strength` остаётся единственным масштабом
температуры шага. Переоткрытие — только с новыми данными и заново; код в
истории git до 2026-09-11.

### Requirement: Zero gain reproduces the frozen baseline bit-for-bit

**Reason**: Контракт «ноль = тождество» выполнялся и стал основанием
удаления: `generation_hash` не сдвинулся на обоих снимках.

**Migration**: Не требуется.

### Requirement: Entropy never overrides acceptance gates

**Reason**: Механизма больше нет; гейты приёмки кандидата не трогались.

**Migration**: Не требуется.

### Requirement: Candidate target follows observed branching

**Reason**: Ранняя остановка по ветвимости (M2R-110) торговала тематичность
(−0.040\*) за копии (−0.039\*) и гейтом отвергнута; при нуле не срабатывала.
Удалена вместе с `branching_aware_target`.

**Migration**: Ручки `markov_branching_degenerate_max` /
`_candidate_floor` удалены; цель пула — `CANDIDATE_TARGET` без поправок.

### Requirement: The shipped default is decided by a pre-registered gate

**Reason**: Гейт `phase2_entropy` вынес вердикт 2026-08-12 и снят вместе с
инструментом; блок порогов удалён из `eval_thresholds.yaml` как вывод из
строя закрытого гейта, не как смена порога.

**Migration**: Вердикт остаётся в `docs/v2/00_STATUS.md` и
`eval_2026-08-12_phase2-verdict.md`.

### Requirement: The applied temperature is observable

**Reason**: Применённая температура теперь тождественно равна базовой
степени шага; счётчик `mean_applied_temperature` остаётся в телеметрии как
диагностика M2R-010, но требование об отдельной наблюдаемости температуры
фазы 2 беспредметно.

**Migration**: Не требуется.
