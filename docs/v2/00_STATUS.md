# Markov 2.0R — статус реализации

Трекер фаз по roadmap (`03_MARKOV_2_0R_IMPLEMENTATION_ROADMAP.md`). Обновляется
в конце каждой фазы. Решения по гейтам — только по отчётам в
`docs/eval_reports/` и порогам `tools/eval/eval_thresholds.yaml`
(предрегистрированы 2026-08-11, правка — отдельным коммитом с обоснованием).

| Фаза | Change | Статус | Итог |
|---|---|---|---|
| 0. Baseline + eval | `markov2r-phase0-baseline-eval` | **закрыта 2026-08-11** (PR #131, архив) | Аудит (`MARKOV_2_0R_PRE_IMPLEMENTATION_AUDIT.md`); раннер `tools/eval` (матрица C0–CF, промпты `308b7deaea0f`, bootstrap CI, CI-smoke); baseline C0 заморожен на `db_prod_copy` 2026-07-13: success 1.0, copy 0.222, affinity 0.280/0.209, циклы 0.0007; воспроизводимость бит-в-бит подтверждена дважды. Поправка протокола v1.0.1: temporal-метрики недоступны до Phase 3. |
| 1. Телеметрия + тень + кэш | `markov2r-phase1-telemetry-shadow-cache` | **закрыта 2026-08-11** (PR #132/#133, архив) | M2R-010: энтропия/ветвление/confidence пулов → трасса и `/stats`. M2R-020: теневой order-4 (estimator=window); **первые данные гейта Phase 7: 0.0% на 897 шагах** (вердикт с ≥1000). M2R-030: инкрементальная вкатка дельт в кэши вместо сброса (kill switch `markov_cache_incremental`); hit-rate 41% оффлайн. Нейтральность: generation_hash идентичен, eval-контент побайтно равен C0 (`eval_2026-08-11_phase1-gate.md`), p95 48 мс (бюджет 150). |
| 2. Entropy sampling | `markov2r-phase2-entropy-sampling` | **следующая** | M2R-100/110 (ТЗ §6): энтропия → температура, `GAIN=0` ⇒ 1.x бит-в-бит. Первая фаза, меняющая живое поведение: нужна конфигурация C1 в матрице и ablation-гейт (copy не растёт, distinct-2/3 растут, p95 в бюджете). Вход готов: `pool_diagnostics` из Phase 1. |
| 3. Temporal layer | `markov2r-phase3-*` (делить: core → calibration → gc) | не начата | Миграция M2R-200 добавит `first_seen/last_seen/s_value/s_updated_at`; исторические `first_seen` = дата миграции (аудит §8) — temporal-метрики после окна накопления. |
| 4. PMI + коллокации | `markov2r-phase4-pmi-collocations` | не начата | Скоринг-уровень (ADR-016), интеграция в hot n-grams. Гейт: ручная разметка топ-20 (нужны оценщики — владелец + участники чата). |
| 5. Lexical anchoring | `markov2r-phase5-*` (делить: reverse-index → seeded) | не начата | Эксперимент ADR-012; завершение = решение M2R-430 с цифрами. |
| 6. Anti-cycle + jumps | gated | ждёт данных | Гейт rate×harm; наблюдаемый rate 0.0007 при пороге 0.05 — пока и близко нет; harm требует ручной раунд (док 05 §5). |
| 7. Order-4 | gated | ждёт данных | Тень: 0.0% на 897 шагах при пороге 10% — ранний сигнал «закрыть без реализации»; вердикт при ≥1000 шагов. |
| 8. Реакции-бандит | эксперимент | не начата | Не заблокирована (aiogram 3.29 умеет реакции; нужен хендлер `message_reaction` + бот-админ в чате — аудит §5). |

## С чего начинать следующую сессию

1. `/opsx:propose markov2r-phase2-entropy-sampling` (M2R-100/110, ТЗ §6) →
   approval владельца → apply. Не забыть: конфигурация C1 в
   `tools/eval/matrix.yaml`, ablation-прогон против C0, отчёт в
   `docs/eval_reports/`, архив change после мержа.
2. Фоновая гигиена (вне 2.0R, по желанию): подъём yanked `aiogram 3.29.0`
   отдельным change; `actions/checkout@v4` предупреждает о Node 20.

## Инструменты

- Полный прогон: `python -m tools.eval --db db_prod_copy/markov.db --label <тег>`
- Гард рефакторингов: `python -m tools.generation_hash --db db_prod_copy/markov.db`
- Смок (как в CI): `python -m tools.eval --smoke`
