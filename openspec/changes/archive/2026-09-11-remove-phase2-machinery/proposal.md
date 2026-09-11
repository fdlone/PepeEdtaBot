## Why

O22, группа 2 (решение владельца 2026-09-11): фаза 2 Markov 2.0R закрыта
2026-08-12 отрицательным вердиктом — энтропийная температура (M2R-100) и
ранняя остановка по ветвимости (M2R-110) провалили гейт всеми шестью плечами
(`eval_2026-08-12_phase2-verdict.md`: энтропия шага бимодальна, температуре
не за что зацепиться; остановка торгует тематичность). Код остался в `main`
за нулевыми дефолтами: `EntropySampling` протянут через 39 сигнатур и на
каждом шаге возвращает вход без изменений, `branching_aware_target` при нуле
не срабатывает. Шесть ручек в реестре (`markov_entropy_temp_gain`, `_pivot`,
`_temp_min`, `_temp_max`, `markov_branching_degenerate_max`,
`_candidate_floor`) — по переписи 11.09 weak-родитель и gated-дети —
включаемы только через `/set` и ведут к измеренно худшему результату.

## What Changes

- Удалены класс `EntropySampling` и его протяжка через сборщики и прогулку;
  `_step_power` оставляет только диагностику шага для телеметрии M2R-010
  (энтропия и «применённая температура» = базовая степень).
- Удалены `branching_aware_target`, накопитель ветвимости и ранняя остановка
  цикла; цель пула — `CANDIDATE_TARGET` без поправок.
- Удалены шесть ручек (реестр, `RuntimeTunables`, `.env.example`),
  гейт `phase2_entropy` в отчёте и его блок порогов, грид
  `matrix_phase2_grid.yaml`, дети из `knob_census.GATED_BY`,
  тесты фазы 2.
- **Поведение генерации не меняется:** `generation_hash` совпал с базлайном
  на обоих снимках (`hash-log.md`).

## Capabilities

### New Capabilities

_нет_

### Modified Capabilities
- `generation-entropy-sampling`: все требования сняты; capability
  упраздняется.

## Impact

- `app/core/markov.py`, `app/core/response_generator.py`,
  `app/config/{registry,settings}.py`, `.env.example`.
- `tools/eval/{report,knob_census}.py`, `tools/eval/eval_thresholds.yaml`,
  `tools/eval/matrix.yaml` (комментарий C1), удалён
  `tools/eval/matrix_phase2_grid.yaml`; `tools/eval_generation.py`.
- Тесты: удалён `tests/test_markov2r_phase2.py`, сняты классы и фикстуры в
  пяти файлах. Документы: карта §2.1/§4, пайплайн §4/§5.1, STATUS (фаза 2).
