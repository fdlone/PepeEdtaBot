# Tasks — l1-hot-channel-sweep (M3R-145)

## 1. До прогона

- [x] 1.1 `hash-log.md`: ожидание «сдвига нет» (правки только в `tools/eval`)
- [x] 1.2 Зарегистрировать `l1_hot_channel` в `eval_thresholds.yaml` с
      обоснованием каждого числа — **до** первого прогона грида
- [x] 1.3 `matrix_l1_grid.yaml`: C0, C7a (2/0.5), C7b (2/0.25), C7c (2/0.0);
      причина отсутствия `min_count = 1` записана в файле

## 2. Харнесс

- [x] 2.1 Затравка в `run_config_seed` по D1/D2; `GenRecord.seed_drawn`,
      `GenRecord.start_source`; тест: ctx никогда не сидирует; пустой пул даёт
      записи, равные прогону без затравки; детерминизм розыгрыша
- [x] 2.2 Счётчики `hot_ngram_draws` / `hot_ngram_empty_rate` в строке
      машинерии отчёта
- [x] 2.3 Гейт `l1_hot_channel[C7*]` в `evaluate_gates` (по режиму: coverage и
      must-improve — noctx; affinity — ctx; copy/repetition/latency — оба;
      связность — noctx); тесты: coverage ниже пола → insufficient, не fail;
      рост copy → fail; без раунда — insufficient

## 3. Прогон и вердикт

- [x] 3.1 Прогон грида в `noctx` и `ctx` на прод-копии 01.09, отчёты и JSON
      в `docs/eval_reports/`
- [x] 3.2 Вердикт-заметка `eval_2026-09-01_l1-grid-verdict.md`: числа по армам,
      coverage, оговорка D4, что остаётся владельцу (раунд связности,
      промоушен)

## 4. Приёмка

- [x] 4.1 `ruff`, `mypy app/`, полный сьют, покрытие не ниже храповика
- [x] 4.2 `generation_hash --synthetic --check`, факт в `hash-log.md`
- [x] 4.3 `docs/PRE_ROADMAP.md` (M3R-145), `docs/GENERATION_MAP.md` §3.2б,
      `docs/CLOSED.md`, `docs/OPEN.md` (что ждёт владельца),
      `tools/eval/README.md`
