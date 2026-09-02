## Why

`selection-knobs` дал ручки окна отбора и бонус различности за нейтральными
дефолтами. Вопрос M3R-100 — можно ли расширить окно отбора (главную метрику
M3R-011), не заплатив тематичностью, — остаётся неизмеренным. Гейт
регистрируется до прогона, грид — по осям, которые окно определяют.

## What Changes

- **Гейт `selection_window`** в `eval_thresholds.yaml`: coverage в форме,
  обещанной M3R-011, — доля входов с единственной траекторией в окне обязана
  упасть не меньше чем на 5 п.п.; must-improve escape окна (значимо);
  must-not-worsen affinity без копий, copy, repetition; ECB ≥ 4.0;
  латентность; связность по соло-протоколу в ctx; `requires_both_modes`.
- **Грид**: `matrix_selection_grid_1.yaml` (C0, C9m50, C9m80, C9w10) и
  `matrix_selection_grid_2.yaml` (C0, C9d20, C9d40, C9w13): запас окна 0.5 /
  0.8, вес тематичности 1.0 / 1.3 (потолок следует), бонус различности 0.2 /
  0.4. Прогон в обоих режимах на прод-копии 01.09, версия набора
  `bababb4b7693`; C0 обязан совпасть между файлами и с базлайном версии.
- Запись генерации получает `single_trajectory` (escape окна < 2) как
  вход coverage; отчёт печатает вердикт `selection_window[C9*]`.
- Дефолты не меняются; промоушен — отдельным решением по вердикту и раунду.

## Capabilities

### New Capabilities
<!-- нет -->

### Modified Capabilities
- `generation-eval`: гейт окна отбора пре-регистрирован; coverage — сдвиг
  доли входов с единственной траекторией.

## Impact

- `tools/eval/eval_thresholds.yaml`, `tools/eval/report.py`,
  `tools/eval/matrix_selection_grid_{1,2}.yaml`, тесты.
- `docs/eval_reports/eval_<дата>_selection-grid*_{ctx,noctx}.md` + вердикт;
  `docs/PRE_ROADMAP.md` (M3R-100), `docs/GENERATION_MAP.md`, `docs/CLOSED.md`.
- `generation_hash` не задет.
