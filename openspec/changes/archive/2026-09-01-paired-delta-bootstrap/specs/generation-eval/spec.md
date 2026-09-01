## MODIFIED Requirements

### Requirement: Statistical treatment

Every proportion metric SHALL carry a 95% bootstrap confidence interval (≥1000 resamples, pooled over the three protocol seeds). A delta between configurations SHALL be reported as significant only when the delta's interval excludes zero; the report SHALL always print the interval, not only the point estimate. Deltas below the protocol's resolution (~3 p.p. at n=500) SHALL be treated as "no effect".

Матрица конфигураций — **парный дизайн**: все армы проходят один и тот же
набор промптов под одними и теми же сидами, различаясь ровно одной ручкой.
Интервал дельты между армами SHALL учитывать эту парность: ресэмплироваться
SHALL наблюдения как пары, а не два арма независимо.

Независимый ресэмплинг двух армов SHALL считаться несоответствием, а не
допустимой реализацией: он оценивает `Var_A + Var_B` вместо дисперсии парной
разности и потому расширяет интервал тем сильнее, чем выше корреляция армов.
Ошибка направленная — она порождает ложные вердикты «эффекта нет», а не
ложные «эффект есть».

Наблюдение, у которого один из армов не дал ответа, SHALL исключаться из пары
целиком: полупара не является наблюдением разности.

Отчёт SHALL указывать, что интервал дельты парный, — иначе читатель сравнит
его с интервалами прежних отчётов, снятыми другой статистикой.

#### Scenario: Insignificant delta

- **WHEN** a configuration's metric delta interval covers zero
- **THEN** the report marks the delta as not significant regardless of the point estimate's sign

#### Scenario: Коррелированные армы

- **WHEN** два арма дают положительно коррелированные наблюдения на одних и
  тех же промптах
- **THEN** интервал парной дельты уже, чем интервал, полученный независимым
  ресэмплингом тех же данных
- **AND** точечная оценка дельты у обоих способов совпадает

#### Scenario: Арм не дал ответа на промпт

- **WHEN** на каком-то промпте один из армов не дал ответа
- **THEN** это наблюдение не участвует в оценке дельты ни одной своей половиной

#### Scenario: Разрыв сопоставимости объявлен

- **WHEN** отчёт напечатан после перехода на парную оценку
- **THEN** из него видно, что интервал дельты парный
- **AND** прошлые отчёты не пересчитываются и не сравниваются с новыми по
  ширине интервала
