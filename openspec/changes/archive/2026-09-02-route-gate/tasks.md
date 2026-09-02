# Tasks — route-gate (M3R-220)

## 1. До прогона

- [x] 1.1 `hash-log.md`: ожидание «сдвига нет» (правки только в `tools/eval`)
- [x] 1.2 Блок `route_gate` в `eval_thresholds.yaml` — условия с ролями и
      обоснованием (design D2), до прогона

## 2. Инструмент

- [x] 2.1 `tools/eval/report.py`: маршрут под гейтом по данным (D1), вердикт
      `route_gate[<arm>]` для префикса `C11`, связность в `ctx`
- [x] 2.2 Тесты: маршрут выводится из пулов (ноль/два новых — insufficient);
      coverage-пол; must-improve по доле одиночных входов; must-not-worsen;
      без раунда — insufficient; одномодовый прогон — insufficient
- [x] 2.3 `matrix_route_gate.yaml` (C0, C11a20, C11a40)

## 3. Прогон

- [x] 3.1 Оба режима на прод-копии 01.09, C0 сверен с базлайном версии;
      заметка `eval_<дата>_route-gate-verdict.md` (автоматическая часть)
- [x] 3.2 Соло-раунд: **не готовился** — арм провален по must-improve, раунд
      вердикт изменить не может (вердикт-заметка, раздел «Что не сделано»)

## 4. Приёмка

- [x] 4.1 `ruff`, `mypy app/`, полный сьют, покрытие
- [x] 4.2 `generation_hash --synthetic --check`, факт в `hash-log.md`
- [x] 4.3 Роадмап (M3R-220), OPEN (раунд за владельцем), CLOSED, README
      `tools/eval`, CLAUDE.md §6 (список гейтов с `requires_both_modes`)
