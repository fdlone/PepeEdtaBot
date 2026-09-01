# Tasks — context-attempts-knob

## 1. До правок

- [x] 1.1 `hash-log.md`: ожидание «сдвига нет» (дефолт равен константе)

## 2. Ручка

- [x] 2.1 `FieldSpec("generation_attempts_with_context", ...)` в реестре рядом с
      `reply_context_start_bias`; поле в `RuntimeTunables`; `.env.example`
- [x] 2.2 `ResponseGenerator` читает ручку в трёх местах; константа — дефолт
- [x] 2.3 Тесты: дефолт реестра == константа; ручка 2 → третья попытка без
      контекста; ручка 0 → ни одной с контекстом, drop на первой

## 3. Приёмка

- [x] 3.1 `ruff`, `mypy app/`, полный сьют, покрытие
- [x] 3.2 `generation_hash` на обоих снимках, факт в `hash-log.md`
- [x] 3.3 `docs/GENERATION_MAP.md` §2.1/§2.2, `docs/GENERATION_PIPELINE.md` §4
