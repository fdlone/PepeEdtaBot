# Design — honest-blend-telemetry

## Context

Мотивация — proposal. Дефект: `temporal.py::blend`, ветка «короткий слой
пуст» возвращает `BlendedPool(long_p, 0.0)` — log-сжатие уже сдвинуло
распределение относительно сырых `cnt`, но сдвиг меряется от `long_p` и
потому нулевой. Существующая пара метрик едет цепочкой
`BlendedPool.displacement → _WalkDiagnostics.note_blend →
GenerationTrace/телеметрия → /stats + gen-trace`.

## Decisions

- **D1 — третье поле `BlendedPool.raw_displacement` с дефолтом 0.0.**
  NamedTuple с дефолтом сохраняет два существующих позиционных конструктора
  (включая переиспользование в интерполяционной ветке `markov.py`, где
  понятие raw-сдвига смеси неприменимо и остаётся 0.0).
- **D2 — считать на каждом пути `blend()`.** `raw_p = cnt_i / Σcnt`;
  `raw_displacement = TV(final, raw_p)`. Пути: uniform (оба слоя пусты),
  «короткий пуст» (final = long_p — ключевой случай дефекта), «длинный пуст»
  (final = short_p), основной (final = blended). При `Σcnt ≤ 0` — 0.0
  (сравнивать не с чем).
- **D3 — та же проводка, что у пары M2R-210, без новых механизмов:**
  `_WalkDiagnostics.blend_raw_displacement_sum`, поле
  `mean_blend_raw_displacement` в обеих трассах, сумма в
  `GenerationTelemetry`, строка в `/stats` рядом с coverage/сдвигом,
  `raw_shift=` в gen-трассе.
- **D4 — нейтральность структурная:** при `alpha = 0` `blend()` по-прежнему
  возвращает `None` до любой арифметики — новых веток на нейтральном пути
  нет, хеш обязан совпасть байт-в-байт.

## Risks / Trade-offs

- [Пара лишних float-операций на шаг при включённой смеси] → смесь выключена
  в проде (alpha=0), путь не исполняется; при включённой — копейки на фоне
  самой смеси.

## Migration Plan

Обычный мерж; ничего не выкатывается отдельно — метрика оживёт вместе с
первым включением `markov_alpha_*` (M2R-215).
