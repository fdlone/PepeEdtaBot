# Tasks — hot-channel-write-gate (O22, группа 4)

## 1. Замер

- [x] 1.1 Арм без легаси-розыгрыша против дефолта, noctx, копия 10.09 — `eval_2026-09-11_legacy-seed-check_noctx.md`: все дельты 0.000

## 2. Код

- [x] 2.1 Гейт записи горячего окна — `hot_ngram_slot_ratio > 0`; `_hot_ngram_seed`, ручка, поле `GenerationRequest.seed` удалены; краткий `/config` показывает ратио маршрута
- [x] 2.2 Харнесс: `draw_hot_seed`, `HOT_SEED_RNG_OFFSET`, `seed_drawn`; README и комментарий грида L1
- [x] 2.3 Тесты: гейт записи по ратио (вкл/выкл), затравочные тесты сняты, `seed=None` в тест-двойниках убран

## 3. Приёмка

- [x] 3.1 `ruff`, `mypy app/`, полный `unittest`; `generation_hash` на обоих снимках не сдвинут (`hash-log.md`)
- [x] 3.2 Документы: карта §1.1/§2.1, пайплайн §3.2; спеки синхронизированы (MODIFIED hot-route, REMOVED eval)
