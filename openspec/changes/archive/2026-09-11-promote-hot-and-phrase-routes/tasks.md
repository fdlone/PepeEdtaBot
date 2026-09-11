# Tasks — promote-hot-and-phrase-routes (O18 + O20)

## 1. До правок

- [x] 1.1 `hash-log.md`: хеш до правок на обоих снимках (значения после #195 известны), ожидание «сдвиг есть на обоих» с перечнем причин (семь ручек)

## 2. Дефолты

- [x] 2.1 Реестр: семь дефолтов; `.env.example` синхронно; `tests/test_selection_knobs.py` — оба бонуса 0.2 ниже запаса; `test_settings` зелёный
- [x] 2.2 Приёмочный замер (design D2): `matrix_promotion_check.yaml` (C0 = прежние дефолты оверрайдами, CP = новые), оба режима на копии 10.09; условия: ECB ≥ 4.0, copy/repetition не растут значимо, affinity без копий не падает значимо, p95 ≤ 150 — заметка `eval_2026-09-11_promotion-check.md`

## 3. Перезаморозка

- [x] 3.1 `tools/generation_hash_baseline.json`: новое значение синтетики, `revision`, `reason` с перечнем; `python -m tools.generation_hash --synthetic --check` зелёный; прод-копия после правок в `hash-log.md`
- [x] 3.2 `tools/generation_baseline.json` и case-preserved перегенерированы командой `eval_generation`; `tests/test_eval_generation.py` зелёный

## 4. Приёмка

- [x] 4.1 `ruff`, `mypy app/`, полный `unittest`, покрытие ≥ храповика, `python -m tools.eval --smoke`
- [x] 4.2 Документы: OPEN (O18, O20 → CLOSED), роадмап (M3R-230, M3R-210/220, приоритеты), карта §2.1 (семь строк), пайплайн §4/§10.1 (сдвиг базлайна), README харнесса (C0 после 2026-09-11), CLAUDE.md §5 (строка ECB — новый состав пула)
