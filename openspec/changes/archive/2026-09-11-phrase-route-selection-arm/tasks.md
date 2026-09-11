# Tasks — phrase-route-selection-arm (O20)

## 1. Замер

- [x] 1.1 `tools/eval/matrix_route_selection.yaml` (C0, C11d, C11s) с ожиданиями в шапке до прогона; README `tools/eval`
- [x] 1.2 Прогон в обоих режимах на прод-копии 10.09 (`--label route-selection`), отчёты и JSON в `docs/eval_reports/`
- [x] 1.3 Вердикт-заметка `eval_2026-09-11_route-selection-verdict.md`: условия гейта по арму и контролю в обоих режимах, атрибуция маршрута поверх ручки (design D2), взвешивание по прод-доле; при взятой ctx-автоматике — список соло-раунда подготовлен (design D3)

## 2. Приёмка

- [x] 2.1 Документы: роадмап (M3R-220/O20), OPEN (O20), CLOSED, карта §2.1; `ruff`/`unittest` (гарды документов и ID) зелёные
