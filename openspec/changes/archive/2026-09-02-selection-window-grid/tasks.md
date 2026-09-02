# Tasks — selection-window-grid (M3R-100, шаг 2)

## 1. До прогона

- [x] 1.1 `hash-log.md`: ожидание «сдвига нет» (правки только в `tools/eval`)
- [x] 1.2 Блок `selection_window` в `eval_thresholds.yaml` с обоснованием, до прогона
- [x] 1.3 `matrix_selection_grid_{1,2}.yaml`

## 2. Гейт

- [x] 2.1 `_selection_arm_verdict` (coverage по доле входов с одной траекторией,
      must-improve escape, must-not-worsen affinity/copy/repetition, ECB,
      латентность, связность в ctx) и строки в `evaluate_gates`
- [x] 2.2 Тесты: coverage ниже пола → insufficient при росте среднего;
      падение affinity при росте escape → fail; noctx без сдвигов — pass на
      уровне арма; без раунда → insufficient

## 3. Прогон и вердикт

- [x] 3.1 Оба файла в обоих режимах на прод-копии 01.09, C0 сверен между
      файлами и с базлайном версии
- [x] 3.2 Вердикт-заметка `eval_<дата>_selection-grid-verdict.md`

## 4. Приёмка

- [x] 4.1 `ruff`, `mypy app/`, полный сьют, покрытие
- [x] 4.2 `generation_hash --synthetic --check`, факт в `hash-log.md`
- [x] 4.3 Роадмап (M3R-100), карта, CLOSED/OPEN, README
