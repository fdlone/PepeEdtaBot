# Tasks — pool-composition-sweep (M3R-110)

## 1. До прогона

- [x] 1.1 `hash-log.md`: ожидание «сдвига нет» (правки только в `tools/eval`)
- [x] 1.2 Зарегистрировать `pool_composition` в `eval_thresholds.yaml` с
      обоснованием каждого числа — **до** прогона
- [x] 1.3 `matrix_pool_grid_1.yaml` (C0, C8b30, C8b40, C8s40) и
      `matrix_pool_grid_2.yaml` (C0, C8a10, C8m); причины состава — в файлах

## 2. Харнесс

- [x] 2.1 Хуки дописки и мутации → карта «производный → исходный»,
      `start_source` победителя через неё (D2); тест: дописанный победитель
      сохраняет источник; захват вне таймера
- [x] 2.2 Гейт `pool_composition[C8*]` в `evaluate_gates`; тесты: сдвиг доли
      ниже пола → insufficient; рост affinity при падении escape → fail;
      без раунда → insufficient; noctx-арм с нулевыми дельтами — не fail

## 3. Прогон и вердикт

- [x] 3.1 Оба файла грида в `ctx` и `noctx` на прод-копии 01.09; сверка
      сводок C0 между файлами
- [x] 3.2 Вердикт-заметка `eval_2026-09-01_pool-grid-verdict.md`

## 4. Приёмка

- [x] 4.1 `ruff`, `mypy app/`, полный сьют, покрытие
- [x] 4.2 `generation_hash --synthetic --check`, факт в `hash-log.md`
- [x] 4.3 `docs/PRE_ROADMAP.md` (M3R-110), `docs/GENERATION_MAP.md` §1.4,
      `docs/CLOSED.md`, `docs/OPEN.md`, `tools/eval/README.md`
